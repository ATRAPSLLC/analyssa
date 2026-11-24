//! The [`SsaOp`] enum itself - every operation variant and its operands.
//!
//! Behaviour lives in the sibling modules: operand access in [`crate::ir::ops::visit`],
//! effect classification in [`crate::ir::ops::effects`], taxonomy and tokens in
//! [`crate::ir::ops::classify`], control-flow queries in [`crate::ir::ops::control`], and
//! rendering in [`crate::ir::ops::display`].

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    ir::{
        float::{
            FloatClassLayout, FloatExceptionLayout, FloatExceptions, FloatFormat, FloatRounding,
        },
        ops::{
            calls::{CallData, CallInput, CallOutput, ReturnTransfer},
            execution::Computation,
            kinds::{
                AccessQualifiers, AtomicAccessWidth, AtomicOrdering, AtomicRmwOp, AtomicStatus,
                BreakpointOp, CmpKind, CompareBlkData, FenceKind, FloatFunctionKind,
                FpuControlKind, KindedVecData, NativeKindedData, RcwProtection, RuntimeValueKind,
                SystemOpKind,
            },
            native::{CompareCondition, FlagCondition, FlagsMask},
            vector::{
                Contraction, ReduceGroups, VectorBinaryKind, VectorBitmaskKind, VectorCastKind,
                VectorCompareKind, VectorElement, VectorElementCountData, VectorExtendInLaneData,
                VectorFaultMode, VectorMaskMode, VectorPackKind, VectorPredicateWhileData,
                VectorReduceKind, VectorSegmentLayout, VectorShuffleSelector,
                VectorStructLoadReplicateData, VectorSveComputeData, VectorTernaryKind,
                VectorUnaryKind,
            },
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

mod emit;

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
    derive(Serialize, Deserialize),
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
        /// Whether the compare is signalling (IEEE 754 compareSignaling): a
        /// quiet NaN raises the invalid-operation exception as well as a
        /// signalling one.
        signaling: bool,
    },

    /// The floating-point exceptions an operation raises, laid out as a status
    /// register keeps them: `dest = raised(op)`, the bit `layout` keeps each
    /// flag of `raises` at set where computing `op` raises the flag, every
    /// other bit clear.
    ///
    /// `op` is the float operation on its own operands, which are this
    /// operation's uses; its definitions are `dest` and are not read. A
    /// status register's accrual reads it (`fflags = fflags | raised(fadd)`),
    /// and `raises` bounds what it can set: a round to an integral value that
    /// does not signal inexact (`frinti`) cannot set the inexact flag, and an
    /// operation sets no flag its architecture does not have. Where an
    /// instruction computes only some of `op`'s lanes (an AVX-512 opmask that
    /// masks the result), `lanes` names them, and the others raise nothing.
    FloatRaise {
        /// Destination SSA variable: the raised flags, at their register bits.
        dest: SsaVarId,
        /// The float operation whose exceptions these are.
        op: Box<SsaOp<T>>,
        /// The lanes the operation computes: one bit per lane, bit `i`
        /// governing lane `i` (an AVX-512 opmask register), or a vector of the
        /// operation's lanes, each all ones where the lane is active. `None`
        /// where it computes every lane.
        lanes: Option<SsaVarId>,
        /// The flags the operation can raise.
        raises: FloatExceptions,
        /// The bit each flag is kept at.
        layout: FloatExceptionLayout,
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

    /// Rotate left: `dest = value <<< amount`, by `amount` modulo the value's
    /// width - a rotation is periodic in the width, so every amount is defined.
    Rol {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Shift or rotate amount variable.
        amount: SsaVarId,
    },

    /// Rotate right: `dest = value >>> amount`, by `amount` modulo the value's
    /// width - a rotation is periodic in the width, so every amount is defined.
    Ror {
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

    /// Parity: `dest = parity(src)` - 1 if odd number of set bits, 0 if even
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
        /// Whether a float compare is signalling (IEEE 754 compareSignaling):
        /// a quiet NaN raises the invalid-operation exception as well as a
        /// signalling one. `false` for integers.
        signaling: bool,
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
        /// Whether a float compare is signalling (IEEE 754 compareSignaling):
        /// a quiet NaN raises the invalid-operation exception as well as a
        /// signalling one. `false` for integers.
        signaling: bool,
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
        /// Whether a float compare is signalling (IEEE 754 compareSignaling):
        /// a quiet NaN raises the invalid-operation exception as well as a
        /// signalling one. `false` for integers.
        signaling: bool,
    },

    /// Boolean conjunction: `dest = left && right`.
    ///
    /// The operands are truth values: any non-zero integer is true.
    BoolAnd {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean disjunction: `dest = left || right`.
    ///
    /// The operands are truth values: any non-zero integer is true.
    BoolOr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean exclusive-or: `dest = (left != 0) != (right != 0)`.
    ///
    /// The operands are truth values, compared as such: `2 xor 1` is false,
    /// because both are true, not true because their bit patterns differ.
    BoolXor {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
    },

    /// Boolean negation: `dest = !value`.
    ///
    /// The operand is a truth value: any non-zero integer is true.
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
    /// source), and equal-width reinterpretation - the physical widen/narrow is a
    /// consequence of the source and `target` widths, not a distinct operation,
    /// so it is derived on demand rather than committed at lift time (the lifter
    /// does not always know the source width). `unsigned` selects zero- vs
    /// sign-extension semantics; `overflow_check` is CIL `conv.ovf.*` (may throw).
    ///
    /// Its operand and target are integers or pointers: a float's bits move
    /// with [`SsaOp::Bitcast`], and a float's value converts with
    /// [`SsaOp::FloatToInt`], [`SsaOp::IntToFloat`] or [`SsaOp::FloatConv`].
    /// The verifier refuses a float or a vector on either side; a type the
    /// target classifies as neither is not judged.
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
        /// The rounding the conversion applies where its result is inexact, or
        /// `None` for the environment's current mode.
        rounding: Option<FloatRounding>,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
    },

    /// Floating-point→integer conversion: `dest = (target_int)operand`,
    /// rounded toward zero (`fptosi`/`fptoui`, x87 `fisttp`, CIL `conv.ovf.*`
    /// from a float; a conversion that rounds otherwise, x87 `fist`, rounds
    /// first). `unsigned` selects the destination interpretation, and agrees
    /// with an integer `target`'s sign (the verifier checks); `overflow_check`
    /// is the CIL checked form (may throw).
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
        /// The format of the float converted, where the producer states it.
        ///
        /// It decides how the operand is read once its own type is gone - a
        /// value stored and loaded back as an integer of its width - which the
        /// width alone does not at 16 bits, where binary16 and bfloat16 meet
        /// ([`SsaOp::restated_for_operand`]). `None` where nothing states it.
        source: Option<FloatFormat>,
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
        /// The rounding the conversion applies where its result is inexact, or
        /// `None` for the environment's current mode.
        rounding: Option<FloatRounding>,
        /// The format of the float converted, where the producer states it, as
        /// [`SsaOp::FloatToInt`]'s states it.
        source: Option<FloatFormat>,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
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

    /// Compose a flags value from two: the bits `mask` selects come from
    /// `update`, every other bit from `base`.
    ///
    /// A partial flag write. An instruction that defines some condition flags
    /// and leaves the rest as they were - x86 `inc` keeps the carry, an ARM
    /// `muls` sets only the sign and zero flags - produces the flags of its
    /// operation merged over the flags it inherited, so a later read of a bit
    /// it left alone reads the instruction that last set it.
    MergeFlags {
        /// Destination flags variable.
        dest: SsaVarId,
        /// The flags every bit outside `mask` comes from.
        base: SsaVarId,
        /// The flags every bit inside `mask` comes from.
        update: SsaVarId,
        /// The bits taken from `update`.
        mask: FlagsMask,
    },

    /// Set one flag of a flags value from a boolean: `dest` is `base` with the
    /// single bit `mask` names replaced by `bit`.
    ///
    /// The dual of [`Self::ReadFlags`], for a flag no operation's own flags
    /// describe: an ARM logical instruction's carry is its shifter's carry out,
    /// and a two-step add-with-carry's carry is the or of its steps' carries.
    InsertFlag {
        /// Destination flags variable.
        dest: SsaVarId,
        /// The flags every other bit comes from.
        base: SsaVarId,
        /// The boolean the bit takes.
        bit: SsaVarId,
        /// The single bit replaced.
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
        /// Lane element descriptor (class / width / scalar / signedness).
        element: VectorElement,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
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
        /// Lane element descriptor (class / width / scalar / signedness).
        element: VectorElement,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
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
        /// Lane element descriptor (class / width / scalar / signedness).
        element: VectorElement,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
    },

    /// Predicated lane-wise unary vector operation.
    VectorPredicatedUnary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorUnaryKind,
        /// Lane element descriptor (class / width / scalar / signedness).
        element: VectorElement,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
    },

    /// Predicated lane-wise binary vector operation.
    VectorPredicatedBinary {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorBinaryKind,
        /// Lane element descriptor (class / width / scalar / signedness).
        element: VectorElement,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
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
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector operation kind.
        kind: VectorTernaryKind,
        /// Lane element descriptor (class / width / scalar / signedness).
        element: VectorElement,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
    },

    /// Lane-wise vector comparison producing a vector mask: each lane all
    /// ones where the relation holds and zero where it does not, the lanes
    /// laid out as the operands' ([`Self::VectorBitmask`] gathers them into
    /// one bit per lane).
    VectorCompare {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Right operand variable.
        right: SsaVarId,
        /// Vector comparison kind.
        kind: VectorCompareKind,
        /// The lanes compared: integers ordered as their stated signedness
        /// reads them, or floats ordered as IEEE does, a NaN unordered with
        /// everything.
        element: VectorElement,
        /// Whether a float compare is signalling (IEEE 754 compareSignaling):
        /// a quiet NaN lane raises the invalid-operation exception as well as
        /// a signalling one. `false` for integer lanes.
        signaling: bool,
    },

    /// Vector load from memory.
    VectorLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Vector store to memory.
    VectorStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Predicated vector load from memory.
    VectorMaskedLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Predicated vector store to memory.
    VectorMaskedStore {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Loads one scalar value and broadcasts it to all vector lanes.
    VectorBroadcastLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Gathers vector lanes from memory using vector indices: lane `i` from
    /// `base` plus lane `i` of `indices`.
    VectorGather {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Base address operand variable.
        base: SsaVarId,
        /// The byte offset of each lane's address from `base`, lane `i` lane
        /// `i`'s, a signed integer of its lane's width: an index already
        /// extended and scaled.
        indices: SsaVarId,
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Optional passthrough vector for inactive lanes.
        passthrough: Option<SsaVarId>,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Inactive lane behavior.
        mode: VectorMaskMode,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Vector load with first-fault or fault-only-first behavior.
    VectorFaultingLoad {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// The elements the load read, one bit per element (bit `i` element
        /// `i`'s): every active element but those a fault suppressed.
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Loads multiple vector segments from memory.
    VectorSegmentLoad {
        /// Destination vector variables, one per segment.
        dests: Vec<SsaVarId>,
        /// Base address operand variable.
        base: SsaVarId,
        /// The structures it loads, where not every one: one bit per lane,
        /// bit `i` governing structure `i` (lane `i` of every segment), or a
        /// vector of the lanes, each all ones where active (an SVE predicate,
        /// `ld2w {z0.s, z1.s}, p0/z, [x0]`). An inactive structure reads no
        /// memory and loads zero into each segment's lane.
        mask: Option<SsaVarId>,
        /// Vector type metadata for each segment.
        vector_type: T::Type,
        /// Number of segments loaded.
        segments: u32,
        /// Segment memory layout.
        layout: VectorSegmentLayout,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Scatters vector lanes to memory using vector indices.
    VectorScatter {
        /// Base address operand variable.
        base: SsaVarId,
        /// The byte offset of each lane's address from `base`, as
        /// [`SsaOp::VectorGather`] states it.
        indices: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Stores multiple vector segments to memory.
    VectorSegmentStore {
        /// Base address operand variable.
        base: SsaVarId,
        /// Vector values to store, one per segment.
        values: Vec<SsaVarId>,
        /// The structures it stores, where not every one, as
        /// [`SsaOp::VectorSegmentLoad`]'s mask states them (`st2w {z0.s,
        /// z1.s}, p0, [x0]`): an inactive structure writes no memory.
        mask: Option<SsaVarId>,
        /// Vector type metadata for each segment.
        vector_type: T::Type,
        /// Number of segments stored.
        segments: u32,
        /// Segment memory layout.
        layout: VectorSegmentLayout,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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

    /// Selects lanes from one or two vector inputs laid end to end, the left
    /// lowest: by a mask the instruction fixes, or by an index vector it
    /// computes ([`VectorShuffleSelector`]). Several tables (`tbl` of four
    /// registers) are joined by mask into the two inputs first.
    VectorShuffle {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Left operand variable.
        left: SsaVarId,
        /// Optional second shuffle input vector.
        right: Option<SsaVarId>,
        /// Which input lane each result lane takes.
        selector: VectorShuffleSelector,
    },

    /// Converts each lane of a vector to another lane type: an integer of
    /// another width or signedness, a float, or a float of another format.
    VectorCast {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Target vector type metadata.
        target_type: T::Type,
        /// How the source lanes are read.
        from: VectorCastKind,
        /// How the destination lanes are written.
        to: VectorCastKind,
        /// The rounding a lane's conversion applies where its result is
        /// inexact (a narrowing between floats, an integer to a float), or
        /// `None` for the environment's current mode.
        rounding: Option<FloatRounding>,
        /// Whether an integer conversion clamps a lane to the range of the
        /// destination's lane, as `to` reads it, rather than keeping its low
        /// bits: a saturating narrowing (`sqxtn`, `sqxtun`, `packsswb`).
        saturate: bool,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
        /// The value that names, at run time, the encoding and scale of a
        /// format the instruction does not state: Arm's FP8 lanes, which FPMR
        /// selects. `None` where every lane's format is stated. Two
        /// conversions of one value under different ones are different
        /// values, so this is an input like any other.
        format: Option<SsaVarId>,
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
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
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
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Compresses active vector lanes and stores them contiguously to memory.
    VectorPackStore {
        /// Base address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// The mask: one bit per lane, bit `i` governing lane `i` (an
        /// AVX-512 opmask register), or a vector of the operation's lanes,
        /// each all ones where the lane is active (an SVE predicate).
        mask: SsaVarId,
        /// Vector type metadata.
        vector_type: T::Type,
        /// Lane element width in bits.
        element_bits: u32,
        /// Packing direction. Must be [`VectorPackKind::Compress`].
        kind: VectorPackKind,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Reduces vector lanes to one scalar result, or, grouped, each run of
    /// adjacent lanes to one lane of the result ([`ReduceGroups`]).
    VectorReduce {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Vector or mask operand variable.
        value: SsaVarId,
        /// Reduction operation kind.
        kind: VectorReduceKind,
        /// Lane element descriptor of the reduced operand: its class, width
        /// and, for [`VectorReduceKind::Min`] and [`VectorReduceKind::Max`] over
        /// integers, its signedness.
        element: VectorElement,
        /// The scalar of the lane type the reduction starts from, combined
        /// before every lane: SVE `fadda`'s accumulator, RISC-V's `vs1[0]`.
        /// `None` reduces the lanes alone.
        initial: Option<SsaVarId>,
        /// Whether the lanes combine one at a time in lane order, the initial
        /// value first (SVE `fadda`, RISC-V `vfredosum`), each step rounded,
        /// rather than in an order the operation does not state. Only float
        /// lanes tell the two apart.
        ordered: bool,
        /// The runs of lanes each lane of a vector result reduces, and that
        /// lane's width; `None` reduces every lane to one scalar.
        groups: Option<ReduceGroups>,
        /// The rounding mode of the floating-point environment the operation
        /// rounds by, where it rounds by the environment's mode: an integer in
        /// [`VectorRounding::FloatingEnvironment`]'s encoding (0 nearest even,
        /// 1 toward zero, 2 down, 3 up, 4 nearest away), which the host reads
        /// from the unit's control register. `None` where the operation states
        /// its rounding, rounds nothing, or the mode is the target's own
        /// ([`Target::environment_rounding`]).
        ///
        /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
        environment: Option<SsaVarId>,
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
        /// The machine state the branch hands to code the lift has not seen:
        /// each register's value at the branch, under a key the host defines.
        /// An unresolved branch continues where any register may be read, so
        /// each is an ordinary use; the host binds them to the entry values
        /// of the code it lifts there. The last occurrence of a key supplies
        /// it.
        #[cfg_attr(feature = "serde", serde(default))]
        state: Box<[CallInput]>,
    },

    /// Return from method with optional value, reading the processor state
    /// a caller observes after the return.
    ///
    /// `state` holds the values of that state the return leaves to its
    /// caller beside the value: the sticky status flags the function accrued
    /// (x87's and `MXCSR`'s exception flags, Arm's `FPSR` and `Q`, RISC-V's
    /// `fflags`, MIPS's `DSPControl` flags), which a caller's `fetestexcept`
    /// or `rddsp` reads after the call. Read here, an accrual stays live to
    /// the return as the effect it is ([`Self::returns_with`]).
    Return {
        /// Optional return value variable.
        value: Option<SsaVarId>,
        /// The processor state a caller observes after the return.
        state: Box<[SsaVarId]>,
        /// The values the return hands its caller beside `value`, each under
        /// the key a call receives it by ([`CallData::results`]): a second
        /// register the function leaves a value in, which its callers read.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "<[CallInput]>::is_empty")
        )]
        results: Box<[CallInput]>,
        /// What the return states for static execution alone: the value it
        /// transfers control through and the registers it hands its caller
        /// ([`ReturnTransfer`]). `None` where it states neither.
        #[cfg_attr(feature = "serde", serde(default))]
        transfer: Option<Box<ReturnTransfer>>,
    },

    /// Load instance field: `dest = object.field`
    LoadField {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
        /// How the access is made ([`AccessQualifiers`]): CIL `volatile.` and
        /// `unaligned.` state it.
        access: AccessQualifiers,
    },

    /// Store instance field: `object.field = value`
    StoreField {
        /// Object operand variable.
        object: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
        /// Value operand variable.
        value: SsaVarId,
        /// How the access is made ([`AccessQualifiers`]): CIL `volatile.` and
        /// `unaligned.` state it.
        access: AccessQualifiers,
    },

    /// Load static field: `dest = ClassName.field`
    LoadStaticField {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Field reference metadata.
        field: T::FieldRef,
        /// How the access is made ([`AccessQualifiers`]): CIL `volatile.` and
        /// `unaligned.` state it.
        access: AccessQualifiers,
    },

    /// Store static field: `ClassName.field = value`
    StoreStaticField {
        /// Field reference metadata.
        field: T::FieldRef,
        /// Value operand variable.
        value: SsaVarId,
        /// How the access is made ([`AccessQualifiers`]): CIL `volatile.` and
        /// `unaligned.` state it.
        access: AccessQualifiers,
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
        /// Whether the address is never written through (CIL `readonly.`),
        /// which spares the runtime the element type check of an array of
        /// references.
        readonly: bool,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Store through pointer: `*ptr = value`
    StoreIndirect {
        /// Address operand variable.
        addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::Type,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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

    /// The arguments the current call received past its declared parameters:
    /// `dest = __arglist`, a C `va_list` or a CIL `arglist` handle.
    ///
    /// What it designates is the current call's own argument area, so the
    /// value is fixed for the activation and reads no program state.
    ArgList {
        /// Destination SSA variable.
        dest: SsaVarId,
    },

    /// Reads the next variadic argument and moves the list past it:
    /// `dest = va_arg(*list, ty)`.
    ///
    /// `list` is the address of the list object. How the list locates and
    /// steps past an argument is its ABI's, so the operation reads and writes
    /// that object and reads wherever the argument lies: the save areas or
    /// the stack.
    ArgNext {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Address of the list object.
        list: SsaVarId,
        /// The type of the argument read.
        ty: T::Type,
    },

    /// Reads the next variadic argument of a list held as a value, and states
    /// the list past it: `dest = va_arg(list, ty)`, `next` the moved list.
    ///
    /// The value form of [`SsaOp::ArgNext`], for a list that is the address of
    /// the next argument where no list object exists: an optimiser keeps
    /// that address in a register. It reads the argument and writes nothing.
    ArgStep {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// The list past the argument.
        next: SsaVarId,
        /// The list, the address of the argument read.
        list: SsaVarId,
        /// The type of the argument read.
        ty: T::Type,
    },

    /// Make a typed reference to the value at an address (CIL `mkrefany`):
    /// `dest = __makeref(*address)`, typed `value_type`.
    MakeTypedRef {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// The address the reference designates.
        address: SsaVarId,
        /// The type of the value at the address.
        value_type: T::TypeRef,
    },

    /// The address a typed reference designates (CIL `refanyval`):
    /// `dest = &__refvalue(typed_ref, Type)`. Throws when the reference is not
    /// of `value_type`.
    TypedRefValue {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// The typed reference.
        typed_ref: SsaVarId,
        /// The type the reference must have.
        value_type: T::TypeRef,
    },

    /// The type of a typed reference, as a runtime type handle (CIL
    /// `refanytype`): `dest = __reftype(typed_ref).TypeHandle`.
    TypedRefType {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// The typed reference.
        typed_ref: SsaVarId,
    },

    /// A call ([`CallData`]): the code it runs and how it reaches it
    /// ([`Callee`](crate::ir::ops::Callee): direct, virtual, constrained or
    /// through a pointer), its arguments and result, and what it does beyond
    /// computing it ([`CallEffects`](crate::ir::ops::CallEffects)). Boxed: a call carries more than any other
    /// operation, and every operation is as large as the largest.
    Call(Box<CallData<T>>),

    /// Load function pointer: `dest = &method`
    LoadFunctionPtr {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Method reference metadata.
        method: T::MethodRef,
        /// The type whose implementation of a static virtual `method` the
        /// pointer names (CIL `constrained. ldftn`); `None` for the method
        /// itself.
        constraint: Option<T::TypeRef>,
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

    /// A value the execution context supplies: `dest = runtime(kind)`.
    ///
    /// No computation produces `dest`. For the exception object a catch
    /// handler or a filter is entered with, and the state a handler is entered
    /// holding, entering the block is what makes it hold, so the operation is
    /// legal only as a leading instruction of a block control can enter that
    /// way, and the verifier says so. A value the specification leaves to the
    /// implementation ([`crate::ir::ops::RuntimeValueKind::Unspecified`])
    /// holds where control reaches it, anywhere in a block.
    ///
    /// It is fixed where it stands. Hoisting it, copying it into a duplicated
    /// block or numbering two of them equal would each assert something the
    /// runtime never promised, which its
    /// [`RuntimeDefined`](crate::ir::ops::SsaEffectKind::RuntimeDefined) effect class
    /// prevents. Deleting it when `dest` is dead is fine, and dead-code
    /// elimination does.
    RuntimeValue {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Which value the context supplies.
        kind: RuntimeValueKind,
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

    /// Fills a block of memory with copies of a value: `size` bytes at
    /// `dest_addr`, element after element using the value type's attested
    /// storage representation. This is a byte for CIL `initblk` and `memset`,
    /// and the accumulator's storage width for x86 `rep stosw`/`stosd`/`stosq`.
    /// Logical value width alone does not establish a storage extent.
    InitBlk {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// The element written using its declared storage representation.
        value: SsaVarId,
        /// The number of bytes filled.
        size: SsaVarId,
        /// The direction flag the fill reads (x86 `DF`): set, it proceeds
        /// from high to low addresses (`rep stos` after `std`); `None` for a
        /// fill that proceeds upward by definition (CIL `initblk`). Downward
        /// fills require a whole number of storage elements. An upward fill
        /// may end after a prefix of its last element. A zero size accesses
        /// no address and needs no element representation.
        backward: Option<SsaVarId>,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Copies a block of memory as if through a temporary buffer, so that
    /// overlapping regions copy as the source held them before the copy:
    /// `memmove`, which AArch64 FEAT_MOPS `cpyp`/`cpym`/`cpye` perform (they
    /// choose the direction by the overlap). A zero size accesses neither
    /// address and needs no source bytes.
    MoveBlk {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Size operand variable, in bytes.
        size: SsaVarId,
        /// How the read of the source is made ([`AccessQualifiers`]).
        src_access: AccessQualifiers,
        /// How the write of the destination is made ([`AccessQualifiers`]).
        dest_access: AccessQualifiers,
    },

    /// Copies complete elements in stream order, reading each element before writing it.
    ///
    /// Addresses name the first element. `size` is a byte count divisible by
    /// `element_bytes`; reverse streams subtract the element width between steps.
    /// A zero size accesses neither address. The element width remains nonzero.
    CopyBlk {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Source address operand variable.
        src_addr: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
        /// Nonzero bytes read together before writing one complete element.
        ///
        /// One for byte copies; x86 `rep movsw`/`movsd`/`movsq` use two/four/eight.
        element_bytes: u32,
        /// The direction flag the copy reads (x86 `DF`): set, it proceeds
        /// from high to low addresses (`rep movs` after `std`), which matters
        /// for overlapping regions; `None` for a copy that proceeds upward by
        /// definition (CIL `cpblk`).
        backward: Option<SsaVarId>,
        /// How the read of the source is made ([`AccessQualifiers`]).
        src_access: AccessQualifiers,
        /// How the write of the destination is made ([`AccessQualifiers`]).
        dest_access: AccessQualifiers,
    },

    /// Memory fence / barrier
    Fence {
        /// Fence ordering kind.
        kind: FenceKind,
    },

    /// Typed native **system / privileged** operation (`cpuid`, time-stamp reads,
    /// system/control-register access, syscalls, traps, cache/TLB maintenance,
    /// privileged state ops). A structured [`SystemOpKind`] identity drives a
    /// precise effect summary and a distinct similarity class. The payload is boxed (see [`NativeKindedData`]).
    SystemOp(Box<NativeKindedData<SystemOpKind>>),

    /// A **contraction**: products of lanes summed into lanes, as its
    /// [`Contraction`] states (dot products of every width and sign, the
    /// bfloat16 dots, the 52-bit multiply-adds, the masked float dot, matrix
    /// and outer products, and the complex products and rotations). Pure. The payload is boxed (see [`KindedVecData`]).
    Contract(Box<KindedVecData<Contraction>>),

    /// Predicate-generation from scalar loop bounds (see [`VectorPredicateWhileData`]).
    VectorPredicateWhile(Box<VectorPredicateWhileData>),
    /// Sign/zero-extend the low field of each lane in place (see [`VectorExtendInLaneData`]).
    VectorExtendInLane(Box<VectorExtendInLaneData>),
    /// Read the symbolic element count `VL / element_bits` into a scalar (see
    /// [`VectorElementCountData`]).
    VectorElementCount(Box<VectorElementCountData>),
    /// Load-and-replicate a 2/3/4-element structure (see [`VectorStructLoadReplicateData`]).
    VectorStructLoadReplicate(Box<VectorStructLoadReplicateData>),
    /// SVE2/NEON compute op named by its kind (see [`VectorSveComputeData`]).
    VectorSveCompute(Box<VectorSveComputeData>),

    /// Compares two blocks element by element, from `left` and `right`,
    /// while the elements agree as its data says and fewer than `count`
    /// were compared (x86 `repe cmps`, `repne cmps`). Each compare sets the
    /// flags as subtracting the second block's element from the first's.
    /// The payload is boxed (see [`CompareBlkData`]).
    CompareBlk(Box<CompareBlkData>),

    /// Compares a value with a block's elements, from `address`, while they
    /// agree as `while_equal` says and fewer than `count` were compared (x86
    /// `repe scas`, `repne scas`). Each compare sets the flags as subtracting
    /// the element from the value.
    ScanBlk {
        /// The elements left unscanned when the scan stopped.
        remaining: SsaVarId,
        /// The address after the last element scanned.
        next: SsaVarId,
        /// The flags of the last compare, or `flags_in` where none was.
        flags: SsaVarId,
        /// The block's first element.
        address: SsaVarId,
        /// The value each element is compared with.
        value: SsaVarId,
        /// The most elements scanned.
        count: SsaVarId,
        /// The flags an empty scan leaves; `None` where `count` is never
        /// zero, so no scan is empty and nothing reads them.
        flags_in: Option<SsaVarId>,
        /// The direction flag the scan reads (x86 `DF`): set, it walks from
        /// high to low addresses.
        backward: Option<SsaVarId>,
        /// The width of an element, in bytes.
        element_bytes: u32,
        /// Whether the scan goes on while the elements equal the value
        /// (`repe`) rather than while they differ (`repne`).
        while_equal: bool,
        /// How the block is read ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Reads a block's `count` elements in turn, from `address`, keeping the
    /// last (x86 `rep lods`): `value` is the last element read, or
    /// `value_in` where `count` is zero.
    LoadBlk {
        /// The last element read, or `value_in`.
        value: SsaVarId,
        /// The address after the last element read.
        next: SsaVarId,
        /// The block's first element.
        address: SsaVarId,
        /// The elements read.
        count: SsaVarId,
        /// The value an empty read leaves; `None` where `count` is never
        /// zero, so no read is empty and nothing reads it.
        value_in: Option<SsaVarId>,
        /// The direction flag the read reads (x86 `DF`): set, it walks from
        /// high to low addresses.
        backward: Option<SsaVarId>,
        /// The width of an element, in bytes.
        element_bytes: u32,
        /// How the block is read ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Typed native **call-clobber** marker - defines fresh, undefined values
    /// for the caller-saved registers a preceding [`Self::Call`] clobbers (the
    /// `Call` op carries a single `dest`, so the remaining clobbered registers
    /// need an owning def for the verifier). It is pure (the call already
    /// happened) and freely eliminable when its outputs are unread.
    CallClobber {
        /// The caller-saved register values invalidated by the call.
        outputs: Vec<SsaVarId>,
        /// Physical keys bound to fresh output definitions by their list positions.
        ///
        /// Missing entries remain unknown. A key never rewrites a pre-call SSA value.
        #[cfg_attr(feature = "serde", serde(default))]
        machine: Option<Box<[CallOutput]>>,
    },

    /// Native atomic load with explicit ordering, width, and volatility: an
    /// ordered load (AArch64 `ldar`, RISC-V `lw.aq`), or, where it
    /// [`reserves`](Self::AtomicLoad::reserves), an exclusive load that opens
    /// the reservation a store-conditional tests (`ldxr`, `lr.w`, `ll`).
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
        /// Whether the load opens a reservation on its address, which a later
        /// store-conditional to it tests: an exclusive load.
        reserves: bool,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Native store-conditional with explicit status output and ordering.
    AtomicStoreConditional {
        /// Store status output variable.
        status: SsaVarId,
        /// Native success/failure encoding, independently of the stored value.
        #[cfg_attr(feature = "serde", serde(default))]
        status_encoding: AtomicStatus,
        /// Whether every completion consumes the reservation. A missing
        /// contract leaves a possible reservation live after the operation.
        #[cfg_attr(feature = "serde", serde(default))]
        consumes_reservation: bool,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Native exclusive pair load with explicit ordering, width, and volatility.
    ///
    /// The first result reads the lower address and the second reads the next
    /// half-width span. Like a reserving [`Self::AtomicLoad`], it permits a
    /// subsequent conditional store to succeed without guaranteeing success.
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Native pair store-conditional with one shared status output.
    AtomicPairStoreConditional {
        /// Store status output variable.
        status: SsaVarId,
        /// Native success/failure encoding, independently of the stored value.
        #[cfg_attr(feature = "serde", serde(default))]
        status_encoding: AtomicStatus,
        /// Whether every completion consumes the reservation. A missing
        /// contract leaves a possible reservation live after the operation.
        #[cfg_attr(feature = "serde", serde(default))]
        consumes_reservation: bool,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Native atomic conditional read-modify-write: `old = *addr; if
    /// condition(old, compare) { *addr = op(old, value) }; return old`, as one
    /// atomic access (x86 `cmp<cc>xadd`). The condition is a relation of `old`
    /// and `compare`, or one flag of `old - compare` ([`CompareCondition`]).
    AtomicCondRmw {
        /// Old memory value output variable.
        old: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// The value the old one is compared with.
        compare: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Atomic read-modify-write operation, applied where the condition holds.
        op: AtomicRmwOp,
        /// What the comparison of the old value with `compare` must satisfy.
        condition: CompareCondition,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width.
        width: AtomicAccessWidth,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Native atomic read-modify-write of a register pair as one access of
    /// twice a register's width (AArch64 FEAT_LSE128 `swpp`, `ldsetp`,
    /// `ldclrp`): memory becomes `op` of its old value and the pair, and the
    /// old pair is returned.
    ///
    /// The first register is the half at the lower address, as in the other
    /// pair operations: the low-order half of a little-endian access and the
    /// high-order half of a big-endian one. An operation that carries between
    /// the halves (`Add`, `Sub`) reads the pair as one value in the access's
    /// byte order.
    AtomicPairRmw {
        /// Old memory value of the half at the lower address.
        old_first: SsaVarId,
        /// Old memory value of the half at the higher address.
        old_second: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// Half of the value stored at the lower address.
        value_first: SsaVarId,
        /// Half of the value stored at the higher address.
        value_second: SsaVarId,
        /// Atomic read-modify-write operation.
        op: AtomicRmwOp,
        /// Memory ordering.
        ordering: AtomicOrdering,
        /// Total atomic memory access width.
        width: AtomicAccessWidth,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
    },

    /// Native atomic read-check-write (AArch64 FEAT_THE `rcwcas`, `rcwclr`,
    /// `rcwset`, `rcwswp`, their `rcws` soft forms and `p` pairs), one
    /// single-copy-atomic access as Arm's `MemAtomicRCW` makes it.
    ///
    /// The new value is `op` of the old one and `value` (`AndNot` clears the
    /// bits `value` sets, `Or` sets them, `Xchg` replaces the old value).
    /// With a `compare` it is a compare-and-swap: where the old value differs
    /// from `compare` the compare fails, memory keeps the old value and
    /// `status` is `0b1010`. Otherwise the change from the old value to the
    /// new one is checked (`RCWCheck`): the read-check-write check against
    /// `rcw_mask`, the effective mask of `RCWMASK_EL1`, wherever the
    /// protected bit (52 of 64 bits, 114 of 128) is enabled, and for a soft
    /// form the soft check against `rcws_mask` (`RCWSMASK_EL1`) as well.
    /// `status` is then `0`, `rcw_fail`, `!rcws_fail`, `0` as NZCV, so
    /// `0b0010` where both pass, and the new value is written where it is.
    /// Where a check fails the architecture leaves it to the implementation
    /// whether the new value is written anyway
    /// (`Unpredictable_WRITEFAILEDRCWCAS`), so memory then holds the old value
    /// or the new one. `old` is the old value, or, after a compare that
    /// passed, the old value or `compare` (`Unpredictable_CASRETURNOLDVALUE`),
    /// which are equal.
    ///
    /// A pair form accesses 128 bits as one value: its registers' halves are
    /// the frontend's to join and split, in the access's byte order.
    AtomicCheckedRmw {
        /// Old memory value output variable.
        old: SsaVarId,
        /// The check's NZCV, bit 3 `N` to bit 0 `V`.
        status: SsaVarId,
        /// Address operand variable.
        addr: SsaVarId,
        /// The value a compare-and-swap expects; `None` for any other form.
        compare: Option<SsaVarId>,
        /// The operand `op` combines with the old value.
        value: SsaVarId,
        /// What the new value is of the old one and `value`: `AndNot`, `Or`
        /// or `Xchg`.
        op: AtomicRmwOp,
        /// `RCWMASK_EL1`, the bits a protected value may change.
        rcw_mask: SsaVarId,
        /// `RCWSMASK_EL1`, the bits a soft-protected value may change: the
        /// soft forms' second check; `None` for a form that makes only the
        /// first.
        rcws_mask: Option<SsaVarId>,
        /// Whether the protected bit is enabled ([`RcwProtection`]).
        protection: RcwProtection,
        /// Atomic memory ordering.
        ordering: AtomicOrdering,
        /// Atomic memory access width: 64 bits, or 128 for a pair.
        width: AtomicAccessWidth,
        /// How the access is made ([`AccessQualifiers`]).
        access: AccessQualifiers,
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
        /// How the access is made ([`AccessQualifiers`]): CIL `volatile.` and
        /// `unaligned.` state it.
        access: AccessQualifiers,
    },

    /// Store object (value type copy): `*dest = value`
    StoreObj {
        /// Destination address operand variable.
        dest_addr: SsaVarId,
        /// Value operand variable.
        value: SsaVarId,
        /// Value type metadata.
        value_type: T::TypeRef,
        /// How the access is made ([`AccessQualifiers`]): CIL `volatile.` and
        /// `unaligned.` state it.
        access: AccessQualifiers,
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

    /// Classify a floating-point value, producing the class of the operand's
    /// IEEE-754 category in the encoding `layout` states: a mask of one bit
    /// (RISC-V `fclass`, MIPS `class.fmt`) or x87's condition codes (`fxam`).
    FpClassify {
        /// Destination SSA variable (the class).
        dest: SsaVarId,
        /// Floating-point operand variable.
        operand: SsaVarId,
        /// The encoding of the class.
        layout: FloatClassLayout,
    },

    /// A floating-point function the machine computes as one operation, named
    /// by its [`FloatFunctionKind`]: the transcendentals (x87 `fsin`, `fcos`,
    /// `fpatan`, `f2xm1`, `fyl2x`), which no closed form of the primitive
    /// operations states and which do not fold, and the exact functions
    /// `fscale`, `fxtract`, `fprem` and `fprem1`, which fold
    /// ([`crate::analysis::Machine::evaluate_results`]). Pure: its condition
    /// codes are outputs of its own. The payload ([`KindedVecData`]) is boxed
    /// to keep `SsaOp` compact.
    FloatFunction(Box<KindedVecData<FloatFunctionKind>>),

    /// Floating-point unit control / state op (x87 `fldcw`/`fnstcw`/`fnstsw`/
    /// `fldenv`/`fnsave`/`frstor`/`fnclex`/`fdecstp`/`ffree`/`fxsave`/…). The
    /// payload ([`KindedVecData`]) is boxed to keep `SsaOp` compact.
    FpuControl(Box<KindedVecData<FpuControlKind>>),

    /// Allocates `size` bytes of the function's own frame, released when it
    /// returns, and defines their address: CIL `localloc`, C's `alloca`, a
    /// variable-length array, which a native frame lowers to moving the stack
    /// pointer.
    LocalAlloc {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Size operand variable.
        size: SsaVarId,
        /// Whether the bytes hold zero when allocated (CIL `localloc` in a
        /// method that zeroes its locals); otherwise they hold whatever the
        /// stack did.
        zeroed: bool,
    },

    /// Phi node: merges values from different predecessors.
    ///
    /// This is placed at the beginning of blocks with multiple predecessors.
    Phi {
        /// Destination SSA variable.
        dest: SsaVarId,
        /// Incoming phi operands by predecessor block.
        operands: Vec<(usize, SsaVarId)>,
    },

    /// Extracts a runtime-positioned bit range into the target layout.
    ///
    /// Result bit `i` is source bit `offset + i`, with bits outside the source
    /// zero. Both the unsigned offset and length are explicit bit counts.
    /// The target layout must agree with that length, including when the width
    /// scales at runtime. The bit count survives representations without type
    /// metadata.
    VectorSlice {
        /// Destination variable.
        dest: SsaVarId,
        /// Source storage value.
        vector: SsaVarId,
        /// Unsigned bit offset supplied at runtime.
        offset: SsaVarId,
        /// Unsigned result bit length supplied at runtime.
        length: SsaVarId,
        /// Result layout, with total width equal to `length`.
        target_type: T::Type,
    },

    /// Replaces a runtime-positioned bit range while preserving every other bit.
    ///
    /// Bit `offset + i` takes bit `i` of `value` when it is inside the base
    /// storage. Bits beyond the base are discarded. The unsigned offset is in
    /// bits, and the result width equals the base width at runtime.
    VectorInsertSlice {
        /// Destination variable.
        dest: SsaVarId,
        /// Base storage whose other bits are preserved.
        vector: SsaVarId,
        /// Unsigned bit offset supplied at runtime.
        offset: SsaVarId,
        /// Bits inserted at the offset.
        value: SsaVarId,
        /// Result layout, with the same total width as the base.
        target_type: T::Type,
    },

    /// Concatenates two whole fixed or scalable storage values, low part first.
    ///
    /// The result width is the sum of the operand widths at runtime. Larger
    /// register groups are built by composing this operation.
    VectorConcat {
        /// Destination variable.
        dest: SsaVarId,
        /// Low bits of the result.
        left: SsaVarId,
        /// High bits of the result.
        right: SsaVarId,
        /// Result layout, with the sum of the input widths.
        target_type: T::Type,
    },

    /// A computation the machine performs as one operation ([`Computation`]):
    /// a [`Kernel`] over a whole value (a checksum step, a bit deposit, a
    /// decimal adjust, an AES round, a ternary logic function, a packed string
    /// comparison) or over the lanes of a vector under an explicit restart
    /// index, active length and mask (RISC-V's vector extension).
    ///
    /// Inactive elements never execute, access memory or produce exceptions.
    /// The computation defines its complete runtime interface, independent of
    /// any machine instruction encoding, and retains each inactive policy. The
    /// payload ([`KindedVecData`]) is boxed to keep `SsaOp` compact.
    ///
    /// [`Kernel`]: crate::ir::ops::Kernel
    Compute(Box<KindedVecData<Computation>>),
}
