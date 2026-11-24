//! What a lane-wise operation computes in each lane ([`Kernel`]).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::{
    VectorConst,
    integer::BitWidth,
    ops::{
        BcdAdjustKind, CarrylessPart, ControlledKind, CrcPolynomial, FlagsMask, FpHelperKind,
        MemoryTagKind, OpKindTable, PacKind, ScalarCipherKind, VectorBinaryKind, VectorCastKind,
        VectorCompareKind, VectorCryptoKind, VectorFaultMode, VectorMaskBinaryKind,
        VectorReduceKind, VectorUnaryKind,
        execution::{BitfieldPosition, VectorFixedKind, VectorMaskPrefix},
    },
};

/// A computation the machine performs as one operation, named by what it
/// computes, never by the instruction that computes it.
///
/// Under [`LaneControl`](crate::ir::ops::execution::LaneControl) a kernel runs for the selected elements of a vector
/// (RISC-V's vector extension); without it, it computes a whole value: a
/// scalar (`crc32`, `pdep`, `daa`) or a vector register at once (`aesenc`,
/// `vpternlogd`, `pcmpistri`). The kernels the generic operations also state
/// (`Move` through `Estimate7`) exist for the lane-controlled form, which no
/// generic operation states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum Kernel {
    /// Copies one source value, broadcasting a scalar source when necessary.
    Move,
    /// Applies a unary operation to each selected element.
    Unary(VectorUnaryKind),
    /// Applies a binary operation to each selected element.
    ///
    /// A scalar input is broadcast to the active elements without evaluating
    /// any inactive element. Vector inputs supply their matching element.
    /// Floating inputs carry their own formats. Narrower inputs promote exactly
    /// to result precision before arithmetic; integer inputs retain their width.
    Binary(VectorBinaryKind),
    /// Divides integer elements with total quotient and remainder semantics.
    ///
    /// Division by zero returns all ones for a quotient and the dividend for
    /// a remainder. Signed minimum divided by negative one returns the minimum
    /// quotient and zero remainder. Other signed quotients truncate toward zero.
    IntegerDivide {
        /// Whether to return the remainder instead of the quotient.
        remainder: bool,
        /// Whether to read both operands as unsigned integers.
        unsigned: bool,
    },
    /// Shifts or rotates with a count reduced modulo the element bit width.
    ///
    /// The width must be a power of two. Only left shift, right shift, left
    /// rotate and right rotate are valid; right shift follows element signedness.
    ModuloShift {
        /// The shift or rotate applied after count reduction.
        kind: VectorBinaryKind,
    },
    /// Returns the high half of a double-width integer product.
    ///
    /// Mixed products require a signed left operand and an unsigned right operand.
    /// The element's signedness agrees with the left operand.
    MultiplyHigh {
        /// Whether to sign-extend the left operand before multiplication.
        signed_left: bool,
        /// Whether to sign-extend the right operand before multiplication.
        signed_right: bool,
    },
    /// Compares two elements and writes a predicate bit per result.
    Compare {
        /// The comparison relation.
        relation: VectorCompareKind,
        /// Whether quiet as well as signaling NaNs raise the invalid exception.
        signaling: bool,
    },
    /// Converts each selected element between explicit numeric domains.
    Cast {
        /// The source element's numeric domain and signedness.
        from: VectorCastKind,
        /// The result element's numeric domain and signedness.
        to: VectorCastKind,
    },
    /// Extends the source element to the result width, preserving its sign when set.
    Extend {
        /// Whether the source element is signed.
        signed: bool,
    },
    /// Extends narrow inputs before a binary operation at double their width.
    ///
    /// Narrow source elements have half the result width. When `wide_left` is
    /// set, the left source already has the result width and is not extended.
    /// Logical left shifts instead require two unsigned narrow inputs and reduce
    /// each shift count modulo the result width.
    WidenBinary {
        /// Whether the left source already has the result width.
        #[cfg_attr(feature = "serde", serde(default))]
        wide_left: bool,
        /// The arithmetic operation applied after extension.
        kind: VectorBinaryKind,
        /// Whether to sign-extend the left input.
        signed_left: bool,
        /// Whether to sign-extend the right input.
        signed_right: bool,
    },
    /// Fuses a product and addend with one rounding at the result precision.
    ///
    /// Floating inputs carry their own formats. Narrower factors and addends
    /// promote exactly to result precision before the single fused rounding.
    FusedMultiplyAdd {
        /// Whether the product is negated before addition.
        negate_product: bool,
        /// Whether the addend is negated before addition.
        negate_addend: bool,
    },
    /// Reads two integer factors followed by an addend, returning addend plus or minus product.
    ///
    /// The result wraps at the element width. All inputs have that width.
    IntegerMultiplyAdd {
        /// Whether to subtract the product from the addend.
        subtract_product: bool,
    },
    /// Reduces active source elements with an initial scalar element.
    ///
    /// The seed is a scalar value or element zero of a vector input.
    /// The initial element participates even if the execution mask is empty.
    /// An empty source selection copies its exact bits without NaN quieting,
    /// canonicalization or exceptions. An ordered FP sum rounds after each add.
    /// An unordered FP sum uses a deterministic tree for the same runtime shape
    /// and length; node precision and exponent range are at least the result's.
    /// Such a tree may include an additive identity: positive zero for downward
    /// rounding, negative zero otherwise. Analysis may fold only results common
    /// to every allowed tree. `FloatBits::reduce_rounded` implements
    /// the ordered case and one legal unordered tree.
    /// A zero length retains the destination, including its tail. The reduction
    /// writes element zero; the remaining result elements follow the tail policy.
    Reduce {
        /// The reduction operation.
        kind: VectorReduceKind,
        /// Whether floating additions execute in increasing element order.
        ordered: bool,
    },
    /// Reads an indexed element, returning zero beyond the source's capacity.
    ///
    /// An index may be scalar or per-element. Indices are not narrowed to the
    /// data element width. Source capacity, rather than active length, is the bound.
    Gather,
    /// Slides elements with a scalar offset or a single scalar fill element.
    Slide {
        /// Whether source elements move to higher destination indices.
        up: bool,
        /// Whether the second input fills the vacated element of a one-element slide.
        scalar_fill: bool,
    },
    /// Writes the zero-based element index.
    Index,
    /// Loads or stores interleaved fields through explicit byte addresses.
    ///
    /// Address inputs are a base and either a signed byte stride or a vector
    /// of unsigned byte offsets. Offsets are zero-extended or truncated to the
    /// base address width; all address arithmetic wraps at that width. Field
    /// addresses add the field index times the memory access byte width.
    /// Only active elements access memory or report memory exceptions.
    Memory {
        /// Whether to load, rather than store, the fields.
        load: bool,
        /// Width of each memory access, no larger than the result element width.
        access_bits: u16,
        /// Whether loads sign-extend a narrower memory element.
        sign_extend: bool,
        /// Whether bytes retain their address order in the result's low-to-high bits.
        ///
        /// Byte-stream transfers preserve register bytes independently of the
        /// function's numeric memory byte order. Numeric transfers use that
        /// native byte order within each memory element.
        byte_stream: bool,
        /// Whether repeated accesses at an identical address may be coalesced.
        ///
        /// False requires one access per active element even when the stride
        /// happens to be zero. This field affects observable device accesses.
        coalescible: bool,
        /// Whether the second address operand is a vector of byte offsets.
        indexed: bool,
        /// Whether accesses occur in increasing element order.
        ordered: bool,
        /// Number of interleaved fields, which must be nonzero.
        fields: u8,
        /// Normal or first-fault behavior.
        ///
        /// Fault-only-first adds a shortened-length result. First-fault adds a
        /// completion predicate result. Stores require normal fault behavior.
        /// Fault-only-first may also shorten without a fault, making progress
        /// when restart is zero and length is nonzero. Elements between the
        /// shortened length and original length may receive unspecified data,
        /// even under undisturbed policies; they do not access memory again.
        fault: VectorFaultMode,
    },
    /// Copies a scalar into element zero and applies the tail policy to other elements.
    ///
    /// The original restart and active length govern whether any update occurs.
    /// Prestart elements are preserved, including element zero when restart is nonzero.
    InsertScalar,
    /// Shifts a double-width integer source right, then truncates to the result width.
    ///
    /// Counts use only the low log2(2 * result_bits) bits. A count may be scalar
    /// or supplied by the corresponding element of an integer vector.
    NarrowShift {
        /// Whether to shift in zeroes rather than copies of the source sign bit.
        unsigned: bool,
    },
    /// Applies integer scaling and rounding before optional saturation.
    ///
    /// Elements are 8, 16, 32 or 64 bits. The explicit environment uses
    /// [`VectorRounding::FixedEnvironment`](crate::ir::ops::execution::VectorRounding::FixedEnvironment); fractional multiplication is signed.
    /// [`VectorFixedKind::saturates`] selects a required saturation result.
    Fixed {
        /// The arithmetic performed before rounding and destination truncation.
        kind: VectorFixedKind,
    },
    /// Accumulates a full product of independently extended half-width factors.
    ///
    /// The inputs are a left factor, right factor and full-width addend. The
    /// result wraps at the element width, which is twice each factor's width.
    WidenMultiplyAdd {
        /// Whether to sign-extend the left factor.
        signed_left: bool,
        /// Whether to sign-extend the right factor.
        signed_right: bool,
    },
    /// Combines two predicates at each corresponding logical mask bit.
    ///
    /// Packed masks use one-bit elements. Byte-granular predicates use eight-bit
    /// element spacing, so every physical predicate bit participates.
    MaskBinary(VectorMaskBinaryKind),
    /// Adds or subtracts two integers and an explicit packed carry or borrow input.
    ///
    /// All source values are unsigned bit patterns. The third data input supplies
    /// one carry bit per element, or a scalar bit broadcast to every element.
    /// Every selected body element computes even when its carry bit is zero.
    IntegerCarry {
        /// Whether to subtract the second source and incoming borrow.
        subtract: bool,
        /// Whether to return the outgoing carry or borrow mask instead of the data.
        mask_result: bool,
    },
    /// Selects the right data element for a set selector bit, or the left otherwise.
    ///
    /// The three data inputs are left, right and an explicit predicate value.
    /// A scalar right input broadcasts. The predicate layout describes selector
    /// bits, independently of any governing execution predicate.
    Select,
    /// Produces a predicate prefix from one source predicate and the execution mask.
    ///
    /// Restart must be zero. Disabled source positions never terminate the prefix.
    MaskPrefix(VectorMaskPrefix),
    /// Produces the exclusive prefix sum of selected source predicate bits.
    ///
    /// Restart must be zero. Counts wrap to the unsigned result element width.
    MaskIota,
    /// Summarizes selected source predicate bits into one unsigned scalar result.
    ///
    /// Restart must be zero. The result is written even for zero active length.
    /// Neither tail policy nor the predicate at output position zero suppresses it.
    /// A first-index result with no selected set bit is all ones at result width.
    MaskSummary {
        /// Whether to return the first index rather than the population count.
        first: bool,
    },
    /// Reduces half-width source elements into a full-width initial value.
    ///
    /// Integer sources extend according to result signedness before reduction.
    /// Floating sources promote exactly to result precision before addition and
    /// obey the ordered, unordered-tree and empty-selection rules of `Reduce`.
    /// The seed is a scalar value or element zero of a vector input. The output
    /// is element zero of the destination; restart must be zero.
    ReduceWide {
        /// The reduction relation.
        kind: VectorReduceKind,
        /// Whether floating additions execute in increasing source order.
        ordered: bool,
    },
    /// Packs source elements selected by an explicit predicate into a low prefix.
    ///
    /// The two data inputs are a source vector and selector predicate. The
    /// execution length bounds source positions. Remaining destination elements
    /// follow the tail policy. Restart must be zero; there is no governing mask.
    Compress,
    /// Replaces a float's sign while preserving every magnitude and NaN payload bit.
    ///
    /// Reads the magnitude source followed by the sign source. Negation and
    /// exclusive-or are mutually exclusive. No floating exceptions are raised.
    SignInject {
        /// Whether to complement the source sign before inserting it.
        negate: bool,
        /// Whether to XOR the source sign with the magnitude source's sign.
        xor: bool,
    },
    /// Returns the IEEE classification mask in an integer result element.
    ///
    /// Reads one floating-point element of its own format and produces the ten
    /// class bits documented by [`crate::ir::FloatBits::classify`], with zero
    /// high bits and no exceptions, including for signaling NaNs.
    FloatClassify,
    /// Converts a float to a bounded integer with explicit NaN clipping policy.
    ///
    /// The input's type states its format and precision. Invalid results clamp
    /// to the signed or unsigned endpoint and raise only invalid. Valid rounded
    /// fractional inputs raise inexact. See [`crate::ir::FloatBits::to_integer_clamped`].
    FloatToInteger {
        /// Whether to use unsigned result endpoints.
        unsigned: bool,
        /// Whether every NaN becomes the maximum, rather than zero.
        nan_maximum: bool,
    },
    /// Computes the specified seven-bit midpoint-table reciprocal estimate.
    ///
    /// See [`crate::ir::FloatBits::estimate7`] for the exact normalized table,
    /// exceptional classes and subnormal truncation. The square-root estimate
    /// ignores rounding; reciprocal overflow follows the selected rounding mode.
    Estimate7 {
        /// Whether to estimate the reciprocal square root rather than reciprocal.
        square_root: bool,
    },
    /// Multiplies integer bit polynomials and returns the part of their
    /// product [`CarrylessPart`] names.
    ///
    /// Each source has the element width, with a scalar second input
    /// broadcast. Coefficient addition is XOR, so carries never propagate. A
    /// slice of the product has the element width; the whole product twice
    /// it (x86 `pclmulqdq` of the two quadwords its control selects, Arm
    /// `pmull`).
    CarrylessMultiply {
        /// The part of the product returned.
        part: CarrylessPart,
    },
    /// Applies a pure cryptographic primitive to complete element groups.
    ///
    /// `element.bits` is the complete 128- or 256-bit group width, rather than
    /// an internal word width. Runtime start, length and predicates count groups.
    /// Inputs have the primitive's specified order. A scalar block broadcasts;
    /// round selectors are scalar integers and never vector-lane predicates.
    Crypto(VectorCryptoKind),
    /// Deposits the low bits of the source, in order, at the positions the
    /// mask sets, every other bit clear (x86 `pdep`, AArch64 `bdep`). Inputs
    /// are the source and the mask.
    BitDeposit,
    /// Gathers the source bits the mask selects, in order, into the low bits
    /// (x86 `pext`, AArch64 `bext`). Inputs are the source and the mask.
    BitExtract,
    /// Separates the source bits by the mask, each group in order: those the
    /// mask selects into the low bits, the others above them (AArch64
    /// `bgrp`): the extract by the mask, then the extract by its complement
    /// shifted past it. Inputs are the source and the mask.
    BitGroup,
    /// One step of a CRC-32 accumulation by [`CrcPolynomial`] (`crc32`,
    /// `crc32c`). Inputs are the checksum and the data, as wide as the step
    /// reads.
    Checksum(CrcPolynomial),
    /// A value the hardware's random source draws (`rdrand`, `rdseed`), with
    /// whether it succeeded in the flags it writes. Constant inputs never
    /// determine it.
    Random {
        /// `true` for `rdseed` (conditioned entropy), `false` for `rdrand`.
        from_entropy: bool,
    },
    /// An Arm pointer-authentication step ([`PacKind`]: signing,
    /// authenticating or stripping a pointer, or a generic MAC), keyed by
    /// processor state.
    PointerAuth(PacKind),
    /// A scalar block-cipher step, AES or SM4 (RISC-V Zkne, Zknd, Zksed),
    /// reading what [`ScalarCipherKind::evaluate`] reads.
    ScalarCipher(ScalarCipherKind),
    /// A float converted to a signed integer of the result's width modulo its
    /// range, JavaScript's `ToInt32`: the value truncated toward zero and the
    /// low bits of that unbounded integer kept, an infinity or NaN giving zero
    /// (AArch64 `fjcvtzs`, RISC-V `fcvtmod.w.d`).
    FloatToIntModular,
    /// An Arm Memory Tagging Extension allocation-tag computation on an
    /// address ([`MemoryTagKind`]); choosing a random tag reads a generator.
    MemoryTag(MemoryTagKind),
    /// Whether the hardware supports a scalable vector configuration: inputs
    /// are the element width in bits and the group multiplier's numerator and
    /// denominator; the result is a boolean, a property of the
    /// implementation.
    VectorShapeSupported,
    /// The strip-mining length chosen for an application count and a maximum
    /// element count: the count when no greater than the maximum, the maximum
    /// when the count is at least twice it, and an implementation-defined
    /// value between half the count, rounded up, and the maximum otherwise. A
    /// zero maximum gives zero.
    VectorLengthChoice,
    /// Whether the hardware supports a vector float element format: inputs
    /// are the bit width and a [`VectorFloatCapability`](crate::ir::ops::execution::VectorFloatCapability) selector.
    VectorFloatSupported,
    /// Whether the hardware supports a cryptographic word precision and
    /// algorithm: inputs are the word width and a selector (zero AES, one
    /// GHASH, two SHA-2, three SM3, four SM4, five polynomial multiply).
    VectorCryptoSupported,
    /// A decimal or ASCII adjust of the accumulator after arithmetic, by its
    /// flags (x86 `daa`, `das`, `aaa`, `aas`), or of `AX` around a multiply
    /// or divide in radix `base` (`aam`, `aad`). Inputs are the accumulator
    /// and, for the adjusts after arithmetic, the flags (the carry and adjust
    /// flags are the ones read); results the adjusted accumulator and the
    /// flags.
    DecimalAdjust {
        /// The adjust.
        kind: BcdAdjustKind,
        /// The radix of `aam` and `aad`, 10 unless the encoding states
        /// another; 10 for the others, which read none.
        base: u8,
    },
    /// An SVE float helper step ([`FpHelperKind`]: `frecpx`, `fexpa`,
    /// `ftmad`, `ftsmul`, `ftssel`), lane by lane.
    FloatHelper(FpHelperKind),
    /// Any bitwise function of three sources, bit by bit: the result bit is
    /// bit `4a + 2b + c` of `table` for source bits `a`, `b` and `c` (x86
    /// `vpternlogd`, `vpternlogq`).
    TernaryLogic {
        /// The truth table.
        table: u8,
    },
    /// Sums of absolute differences of unsigned bytes over a sliding window
    /// (x86 `mpsadbw`): for each 128-bit lane, `offsets` selects the four
    /// bytes of the second source and the eleven of the first the eight
    /// sums slide over. Inputs are the two sources.
    MultiSad {
        /// The block-offset control.
        offsets: u8,
    },
    /// Sums of absolute differences of unsigned bytes in quadword blocks
    /// (x86 `vdbpsadbw`): the second source's doublewords shuffled within
    /// each 128-bit lane by `selector`, two bits each, then four words per
    /// quadword, the first source's low four bytes against the shuffled
    /// bytes from offsets 0 and 1 and its high four against those from 2
    /// and 3. Inputs are the two sources.
    DoubleBlockSad {
        /// The doubleword shuffle control.
        selector: u8,
    },
    /// A packed string comparison (x86 `pcmpistri`, `pcmpistrm`,
    /// `pcmpestri`, `pcmpestrm`): `control` selects the element format,
    /// the aggregation and the polarity. Inputs are the two strings, then
    /// their lengths for the explicit-length forms; results the index or
    /// mask and the flags.
    StringCompare {
        /// The format, aggregation and polarity control.
        control: u8,
        /// Whether the lengths are inputs (`pcmpestr*`) rather than each
        /// string ending at its first zero element (`pcmpistr*`).
        explicit_length: bool,
        /// Whether the result is the index (`*stri`) rather than the mask
        /// (`*strm`).
        result_index: bool,
    },
    /// A bit field of a quadword extracted to the low bits (SSE4a `extrq`)
    /// or inserted from the low bits of a second quadword (`insertq`), at
    /// the position the instruction or a control quadword after the data
    /// states. The instructions leave their destination's upper quadword
    /// undefined, which is no part of this kernel.
    Bitfield {
        /// Whether the field is inserted rather than extracted.
        insert: bool,
        /// Where the field's position comes from.
        position: BitfieldPosition,
    },
    /// The masks of the elements of each of two sources equal to some element
    /// of the other (x86 `vp2intersectd`, `vp2intersectq`): two results.
    Intersect,
    /// For each byte of the control, the bit of the source quadword it lies
    /// in that the byte's low six bits select, as a mask (x86
    /// `vpshufbitqmb`). Inputs are the source and the control.
    ShuffleBits,
    /// The smallest unsigned 16-bit element of the source and its index, the
    /// lowest one of equal minima, in the low two words of a result otherwise
    /// zero (x86 `phminposuw`).
    HorizontalMinPos,
    /// A float lane operation its control immediate selects
    /// ([`ControlledKind`]: x86 `vrndscale*`, `vreduce*`, `vgetmant*`,
    /// `vrange*`, `vfixupimm*`, `vfpclass*`).
    FloatControlled {
        /// The operation.
        kind: ControlledKind,
        /// The control immediate.
        control: u8,
    },
    /// The part one register holds of the predicate a predicate-as-counter
    /// expands to (Arm's `CounterToPredicate`, SVE2.1 and SME2). A counter's
    /// low 16 bits are an element-size marker in bits 3 to 0 (`xxx1` bytes,
    /// `xx10` halfwords, `x100` words, `1000` doublewords, `0000` no element),
    /// above it a count of elements up to bit `log2(4 * PL)`, and an invert bit
    /// at 15. The expansion is the predicate of four registers, one bit per
    /// byte, setting the first byte of each of the first `count` elements, or
    /// of each other one where inverted. Input the counter, one register's
    /// predicate wide (`PL` bits, an eighth of the vector length); result
    /// part `part` of the expansion, `PL` bits.
    CounterPredicate {
        /// The register of the four whose part is the result.
        part: u8,
    },
    /// The predicate-as-counter of a count of active elements (Arm's
    /// `EncodePredCount`): zero for none, otherwise the element-size marker
    /// of `element_bits`, the count above it and the invert bit at 15. An
    /// inverted counter holds the elements less the count, and a count of
    /// every element is written as an inverted zero. Inputs the count, at
    /// most the elements, the elements the counter spans, and whether the
    /// active elements are the last rather than the first (nonzero); result
    /// a predicate the result's width, at least 16 bits.
    PredicateCounter {
        /// The width of the elements counted: 8, 16, 32 or 64.
        element_bits: u8,
    },
}

impl Kernel {
    /// Whether the data result is a predicate, one bit per element: a
    /// comparison's, a mask combination's or prefix's, and the carry or
    /// borrow out of an add or subtract with carry that writes its mask.
    #[must_use]
    pub const fn writes_predicate(self) -> bool {
        matches!(
            self,
            Self::Compare { .. }
                | Self::MaskBinary(_)
                | Self::MaskPrefix(_)
                | Self::IntegerCarry {
                    mask_result: true,
                    ..
                }
        )
    }

    /// Whether an execution computes every element of its body, so a reader of
    /// the body reads only what it computed. Not an upward slide's holes, the
    /// elements past a compression's count, every element but the first of an
    /// insertion, reduction or summary, or a first-fault load's elements past
    /// the one that faults.
    #[must_use]
    pub const fn executes_its_body(self) -> bool {
        match self {
            Self::Memory { fault, .. } => matches!(fault, VectorFaultMode::Normal),
            Self::Slide {
                up: true,
                scalar_fill: false,
            }
            | Self::Compress
            | Self::InsertScalar
            | Self::Reduce { .. }
            | Self::ReduceWide { .. }
            | Self::MaskSummary { .. } => false,
            _ => true,
        }
    }

    /// Returns the width of the integer a whole-value kernel computes on,
    /// where the kernel fixes it: a pointer authentication's and a
    /// memory-tag step's 64-bit address, a modular conversion's 32-bit
    /// result. A random draw's is the width of the register it fills.
    #[must_use]
    pub const fn fixed_integer_bits(self) -> Option<u32> {
        match self {
            Self::PointerAuth(_) | Self::MemoryTag(_) => Some(64),
            Self::FloatToIntModular => Some(32),
            _ => None,
        }
    }

    /// Returns the kernel's IR name, the words of the operation it is, such
    /// as `crc32c`, `bcd.adjust.add` or `aes.enc`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Move => "move",
            Self::Unary(_) => "unary",
            Self::Binary(_) => "binary",
            Self::IntegerDivide {
                remainder: false, ..
            } => "div",
            Self::IntegerDivide {
                remainder: true, ..
            } => "rem",
            Self::ModuloShift { .. } => "shift.modulo",
            Self::MultiplyHigh { .. } => "mul.high",
            Self::Compare { .. } => "cmp",
            Self::Cast { .. } => "convert",
            Self::Extend { .. } => "extend",
            Self::WidenBinary { .. } => "binary.widen",
            Self::FusedMultiplyAdd { .. } => "fma",
            Self::IntegerMultiplyAdd { .. } => "mul.add",
            Self::Reduce { .. } => "reduce",
            Self::Gather => "gather",
            Self::Slide { up: true, .. } => "slide.up",
            Self::Slide { up: false, .. } => "slide.down",
            Self::Index => "index",
            Self::Memory { load: true, .. } => "load",
            Self::Memory { load: false, .. } => "store",
            Self::InsertScalar => "insert.scalar",
            Self::NarrowShift { .. } => "shift.narrow",
            Self::Fixed { .. } => "fixed",
            Self::WidenMultiplyAdd { .. } => "mul.add.widen",
            Self::MaskBinary(_) => "mask.binary",
            Self::IntegerCarry { .. } => "carry",
            Self::Select => "select",
            Self::MaskPrefix(_) => "mask.prefix",
            Self::MaskIota => "mask.iota",
            Self::MaskSummary { first: false } => "mask.count",
            Self::MaskSummary { first: true } => "mask.first",
            Self::ReduceWide { .. } => "reduce.widen",
            Self::Compress => "compress",
            Self::SignInject { .. } => "sign.inject",
            Self::FloatClassify => "classify",
            Self::FloatToInteger { .. } => "convert.f2i.clamp",
            Self::Estimate7 { square_root: false } => "recip.estimate",
            Self::Estimate7 { square_root: true } => "rsqrt.estimate",
            Self::CarrylessMultiply { part } => part.name(),
            Self::Crypto(kind) => kind.name(),
            Self::BitDeposit => "bit.deposit",
            Self::BitExtract => "bit.extract",
            Self::BitGroup => "bit.group",
            Self::Checksum(CrcPolynomial::Ieee) => "crc32.ieee",
            Self::Checksum(CrcPolynomial::Castagnoli) => "crc32.castagnoli",
            Self::Random {
                from_entropy: false,
            } => "random",
            Self::Random { from_entropy: true } => "random.seed",
            Self::PointerAuth(PacKind::Sign) => "ptr.sign",
            Self::PointerAuth(PacKind::Authenticate) => "ptr.auth",
            Self::PointerAuth(PacKind::Strip) => "ptr.strip",
            Self::PointerAuth(PacKind::GenericMac) => "ptr.mac",
            Self::PointerAuth(PacKind::SignDiversified) => "ptr.sign.diversified",
            Self::PointerAuth(PacKind::AuthenticateDiversified) => "ptr.auth.diversified",
            Self::ScalarCipher(ScalarCipherKind::Aes32EncryptByte) => "aes.enc.byte",
            Self::ScalarCipher(ScalarCipherKind::Aes32EncryptByteMix) => "aes.enc.byte.mix",
            Self::ScalarCipher(ScalarCipherKind::Aes32DecryptByte) => "aes.dec.byte",
            Self::ScalarCipher(ScalarCipherKind::Aes32DecryptByteMix) => "aes.dec.byte.mix",
            Self::ScalarCipher(ScalarCipherKind::Aes64EncryptHalf) => "aes.enc.half",
            Self::ScalarCipher(ScalarCipherKind::Aes64EncryptHalfMix) => "aes.enc.half.mix",
            Self::ScalarCipher(ScalarCipherKind::Aes64DecryptHalf) => "aes.dec.half",
            Self::ScalarCipher(ScalarCipherKind::Aes64DecryptHalfMix) => "aes.dec.half.mix",
            Self::ScalarCipher(ScalarCipherKind::Aes64InverseMix) => "aes.mix.inverse",
            Self::ScalarCipher(ScalarCipherKind::Aes64KeyStep1) => "aes.key.step1",
            Self::ScalarCipher(ScalarCipherKind::Aes64KeyStep2) => "aes.key.step2",
            Self::ScalarCipher(ScalarCipherKind::Sm4EncryptByte) => "sm4.enc.byte",
            Self::ScalarCipher(ScalarCipherKind::Sm4KeyByte) => "sm4.key.byte",
            Self::FloatToIntModular => "convert.f2i.mod",
            Self::MemoryTag(kind) => kind.mnemonic(),
            Self::VectorShapeSupported => "vector.shape.supported",
            Self::VectorLengthChoice => "vector.length.choice",
            Self::VectorFloatSupported => "vector.float.supported",
            Self::VectorCryptoSupported => "vector.crypto.supported",
            Self::DecimalAdjust { kind, .. } => kind.name(),
            Self::FloatHelper(kind) => kind.name(),
            Self::TernaryLogic { .. } => "logic3",
            Self::MultiSad { .. } => "sad.multi",
            Self::DoubleBlockSad { .. } => "sad.double",
            Self::StringCompare {
                explicit_length: false,
                result_index: true,
                ..
            } => "string.cmp.nul.index",
            Self::StringCompare {
                explicit_length: false,
                result_index: false,
                ..
            } => "string.cmp.nul.mask",
            Self::StringCompare {
                explicit_length: true,
                result_index: true,
                ..
            } => "string.cmp.len.index",
            Self::StringCompare {
                explicit_length: true,
                result_index: false,
                ..
            } => "string.cmp.len.mask",
            Self::Bitfield { insert: false, .. } => "field.extract",
            Self::Bitfield { insert: true, .. } => "field.insert",
            Self::Intersect => "intersect",
            Self::ShuffleBits => "shuffle.bits",
            Self::HorizontalMinPos => "min.pos",
            Self::CounterPredicate { .. } => "mask.from_counter",
            Self::PredicateCounter { .. } => "mask.counter.of",
            // A round or a remainder signals inexact unless the control's bit
            // 3 suppresses it, and is then the exact one.
            Self::FloatControlled {
                kind: ControlledKind::RoundScale,
                control,
            } if control & 0b1000 == 0 => "round.scale.exact",
            Self::FloatControlled {
                kind: ControlledKind::Reduce,
                control,
            } if control & 0b1000 == 0 => "round.remainder.exact",
            Self::FloatControlled { kind, .. } => kind.name(),
        }
    }

    /// Returns the stable fingerprint key for this kernel: the key of the
    /// operation it names (`compute.pdep`, `crypto.aesenc`, `bcd.daa`), one
    /// per identity a payload enum names, and one per kernel of a lane
    /// computation, whose parameters are data.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::BitDeposit => "compute.pdep",
            Self::BitExtract => "compute.pext",
            Self::BitGroup => "compute.bgrp",
            Self::Checksum(polynomial) => polynomial.kind_str(),
            Self::Random {
                from_entropy: false,
            } => "compute.rdrand",
            Self::Random { from_entropy: true } => "compute.rdseed",
            // The sub-operation is identity, not data: signing, authenticating
            // and stripping are three instructions, and the difference between
            // the first two is the whole security property.
            Self::PointerAuth(op) => op.kind_str(),
            Self::CarrylessMultiply { part } => part.kind_str(),
            Self::ScalarCipher(kind) => kind.kind_str(),
            Self::FloatToIntModular => "compute.f2i_modular",
            Self::MemoryTag(kind) => kind.kind_str(),
            Self::VectorShapeSupported => "compute.vector.shape.supported",
            Self::VectorLengthChoice => "compute.vector.length.choice",
            Self::VectorFloatSupported => "compute.vector.float.supported",
            Self::VectorCryptoSupported => "compute.vector.crypto.supported",
            Self::Crypto(kind) => kind.kind_str(),
            Self::DecimalAdjust { kind, .. } => kind.kind_str(),
            Self::FloatHelper(kind) => kind.kind_str(),
            Self::FloatControlled { kind, .. } => kind.kind_str(),
            Self::TernaryLogic { .. } => "compute.ternlog",
            Self::MultiSad { .. } => "compute.mpsadbw",
            Self::DoubleBlockSad { .. } => "compute.dbpsadbw",
            Self::StringCompare { .. } => "compute.pcmpstr",
            Self::Bitfield { .. } => "compute.bitfield",
            Self::Intersect => "compute.p2intersect",
            Self::ShuffleBits => "compute.shufbitqmb",
            Self::HorizontalMinPos => "compute.phminposuw",
            Self::CounterPredicate { .. } => "compute.counter.expand",
            Self::PredicateCounter { .. } => "compute.counter.encode",
            Self::Move => "lanes.move",
            Self::Unary(_) => "lanes.unary",
            Self::Binary(_) => "lanes.binary",
            Self::IntegerDivide { .. } => "lanes.divide",
            Self::ModuloShift { .. } => "lanes.shift.modulo",
            Self::MultiplyHigh { .. } => "lanes.mul.high",
            Self::Compare { .. } => "lanes.compare",
            Self::Cast { .. } => "lanes.cast",
            Self::Extend { .. } => "lanes.extend",
            Self::WidenBinary { .. } => "lanes.binary.widen",
            Self::FusedMultiplyAdd { .. } => "lanes.fma",
            Self::IntegerMultiplyAdd { .. } => "lanes.mul.add",
            Self::Reduce { .. } => "lanes.reduce",
            Self::Gather => "lanes.gather",
            Self::Slide { .. } => "lanes.slide",
            Self::Index => "lanes.index",
            Self::Memory { .. } => "lanes.memory",
            Self::InsertScalar => "lanes.insert.scalar",
            Self::NarrowShift { .. } => "lanes.shift.narrow",
            Self::Fixed { .. } => "lanes.fixed",
            Self::WidenMultiplyAdd { .. } => "lanes.mul.add.widen",
            Self::MaskBinary(_) => "lanes.mask.binary",
            Self::IntegerCarry { .. } => "lanes.carry",
            Self::Select => "lanes.select",
            Self::MaskPrefix(_) => "lanes.mask.prefix",
            Self::MaskIota => "lanes.mask.iota",
            Self::MaskSummary { .. } => "lanes.mask.summary",
            Self::ReduceWide { .. } => "lanes.reduce.widen",
            Self::Compress => "lanes.compress",
            Self::SignInject { .. } => "lanes.sign.inject",
            Self::FloatClassify => "lanes.classify",
            Self::FloatToInteger { .. } => "lanes.f2i.clamp",
            Self::Estimate7 { .. } => "lanes.estimate7",
        }
    }

    /// Returns every identity of the whole-value kernels, which
    /// [`Self::kind_str`] keys apart: each payload enum expanded through its
    /// own variants, data fields (a truth table, a control immediate, a
    /// radix) at a canonical placeholder.
    #[must_use]
    pub fn identities() -> Vec<Self> {
        let mut all = vec![
            Self::BitDeposit,
            Self::BitExtract,
            Self::BitGroup,
            Self::Random {
                from_entropy: false,
            },
            Self::Random { from_entropy: true },
            Self::FloatToIntModular,
            Self::VectorShapeSupported,
            Self::VectorLengthChoice,
            Self::VectorFloatSupported,
            Self::VectorCryptoSupported,
            Self::TernaryLogic { table: 0 },
            Self::MultiSad { offsets: 0 },
            Self::DoubleBlockSad { selector: 0 },
            Self::StringCompare {
                control: 0,
                explicit_length: false,
                result_index: false,
            },
            Self::Bitfield {
                insert: false,
                position: BitfieldPosition::Register,
            },
            Self::Intersect,
            Self::ShuffleBits,
            Self::HorizontalMinPos,
            Self::CounterPredicate { part: 0 },
            Self::PredicateCounter { element_bits: 8 },
        ];
        all.extend(CrcPolynomial::all().map(Self::Checksum));
        all.extend(PacKind::all().map(Self::PointerAuth));
        all.extend(CarrylessPart::all().map(|part| Self::CarrylessMultiply { part }));
        all.extend(ScalarCipherKind::all().map(Self::ScalarCipher));
        all.extend(MemoryTagKind::all().map(Self::MemoryTag));
        all.extend(BcdAdjustKind::all().map(|kind| Self::DecimalAdjust { kind, base: 10 }));
        all.extend(
            (0..u16::MAX)
                .map_while(|index| VectorCryptoKind::try_from(index).ok())
                .map(Self::Crypto),
        );
        all.extend(
            (0..u16::MAX)
                .map_while(|index| FpHelperKind::try_from(index).ok())
                .map(Self::FloatHelper),
        );
        all.extend(
            (0..u16::MAX)
                .map_while(|index| ControlledKind::try_from(index).ok())
                .map(|kind| Self::FloatControlled { kind, control: 0 }),
        );
        all
    }

    /// The integer this kernel gives for integer inputs, at `bits` wide.
    ///
    /// Each input is its bits, in the low bits, and its width, in the order
    /// the kernel reads them; `bits` is the width of the result. Inputs:
    ///
    /// - [`Self::BitDeposit`], [`Self::BitExtract`] and [`Self::BitGroup`]:
    ///   the source and the mask, both `bits` wide.
    /// - [`Self::Checksum`]: the checksum, its low 32 bits read, and the data,
    ///   as wide as the step reads (8, 16, 32 or 64 bits); the checksum
    ///   written is zero-extended to `bits`.
    /// - [`Self::CarrylessMultiply`]: the two factors, `bits` wide for a
    ///   slice of the product and half of it for the whole product.
    /// - [`Self::ScalarCipher`]: what [`ScalarCipherKind::evaluate`] reads.
    /// - [`Self::Bitfield`] of quadwords: the source (`extrq`), or the
    ///   destination and the source (`insertq`), then for a register form
    ///   the control, the field's length in bits 5 to 0 and its index in
    ///   bits 13 to 8. A length of zero is 64, and a field past bit 63 is
    ///   undefined and folds nothing.
    /// - [`Self::CounterPredicate`]: the counter, `bits` wide (`PL`, at most
    ///   64): the part of its expansion `bits` wide.
    /// - [`Self::PredicateCounter`]: the count, the elements and whether it is
    ///   inverted; a count past the elements folds nothing.
    /// - [`Self::VectorLengthChoice`]: the application count and the maximum.
    ///   The choice is the count when it is no greater than the maximum and
    ///   the maximum when the count is at least twice it; between the two the
    ///   implementation chooses, and nothing folds.
    ///
    /// # Returns
    ///
    /// The result, or `None` for inputs of another shape and for a
    /// computation constant inputs do not determine: a random source, a
    /// pointer authentication (its key is processor state), a memory tag (the
    /// generator, and whether tags are enabled at all, are processor state),
    /// a hardware support query (a property of the implementation), and a
    /// kernel whose inputs are no integers of at most 64 bits.
    #[must_use]
    pub fn evaluate_integers(self, inputs: &[(u64, u32)], bits: u32) -> Option<u64> {
        let mask = BitWidth::new(bits).checked_mask()?;
        let result = match (self, inputs) {
            (Self::Bitfield { insert, position }, inputs) if bits == 64 => {
                let quadwords = inputs
                    .iter()
                    .map(|(value, width)| (*width == 64).then_some(*value))
                    .collect::<Option<Vec<u64>>>()?;
                // The field's source, after the destination for an insert,
                // then the control of a register form.
                let data = if insert { 2 } else { 1 };
                let (length, index) = match (position, quadwords.get(data..)?) {
                    (BitfieldPosition::Immediate { index, length }, []) => {
                        (u64::from(length & 0x3f), u64::from(index & 0x3f))
                    }
                    (BitfieldPosition::Register, [control]) => {
                        (control & 0x3f, control.checked_shr(8)? & 0x3f)
                    }
                    _ => return None,
                };
                let length = if length == 0 { 64 } else { length };
                // A field past the quadword's top is undefined.
                if index.checked_add(length)? > 64 {
                    return None;
                }
                let field = BitWidth::new(u32::try_from(length).ok()?).checked_mask()?;
                let shift = u32::try_from(index).ok()?;
                match quadwords.as_slice() {
                    [destination, source, ..] if insert => {
                        destination & !field.checked_shl(shift)?
                            | (source & field).checked_shl(shift)?
                    }
                    [source, ..] => source.checked_shr(shift)? & field,
                    [] => return None,
                }
            }
            (Self::BitDeposit | Self::BitExtract, [(source, width), (select, other)])
                if *width == bits && *other == bits =>
            {
                let (mut result, mut next) = (0u64, 0u32);
                let selected =
                    (0..bits).filter(|bit| select.checked_shr(*bit).is_some_and(|b| b & 1 != 0));
                for bit in selected {
                    if self == Self::BitDeposit {
                        result |= (source.checked_shr(next)? & 1).checked_shl(bit)?;
                    } else {
                        result |= (source.checked_shr(bit)? & 1).checked_shl(next)?;
                    }
                    next = next.checked_add(1)?;
                }
                result
            }
            (Self::BitGroup, [(source, width), (select, other)])
                if *width == bits && *other == bits =>
            {
                let extract = |select: u64| {
                    Self::BitExtract.evaluate_integers(&[(*source, bits), (select, bits)], bits)
                };
                let selected = extract(*select)?;
                let others = extract(!*select & mask)?;
                selected | others.checked_shl(select.count_ones()).unwrap_or(0)
            }
            // The checksum is its low word: truncation is the point.
            #[allow(clippy::cast_possible_truncation)]
            (Self::Checksum(polynomial), [(crc, _), (data, width)]) => {
                u64::from(polynomial.step(*crc as u32, *data, *width)?)
            }
            (Self::CarrylessMultiply { part }, [(left, width), (right, other)])
                if *width == bits && *other == bits && part != CarrylessPart::Full =>
            {
                u64::try_from(part.product((*left).into(), (*right).into(), bits)?).ok()?
            }
            (
                Self::CarrylessMultiply {
                    part: CarrylessPart::Full,
                },
                [(left, width), (right, other)],
            ) if *width == *other && width.checked_mul(2) == Some(bits) => u64::try_from(
                CarrylessPart::Full.product((*left).into(), (*right).into(), *width)?,
            )
            .ok()?,
            (Self::ScalarCipher(kind), _) => {
                let values: Vec<u64> = inputs.iter().map(|(value, _)| *value).collect();
                kind.evaluate(&values)?
            }
            (Self::CounterPredicate { part }, [(counter, width)]) if *width == bits => {
                Self::counter_part(*counter, bits, part)?
            }
            (Self::PredicateCounter { element_bits }, [(count, _), (elements, _), (invert, _)])
                if bits >= 16 =>
            {
                Self::counter_encoding(element_bits, *elements, *count, *invert != 0)?
            }
            (Self::VectorLengthChoice, [(requested, _), (maximum, _)]) => {
                if requested <= maximum {
                    *requested
                } else if requested / 2 >= *maximum {
                    *maximum
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        Some(result & mask)
    }

    /// Part `part` of the predicate `counter` expands to, `predicate_bits`
    /// wide ([`Self::CounterPredicate`]): the bits from `part * PL` up to
    /// `(part + 1) * PL` of Arm's `CounterToPredicate`, where `PL` is
    /// `predicate_bits` and the vector `8 * PL` bits. `None` past four parts
    /// and for a counter wider than 64 bits.
    fn counter_part(counter: u64, predicate_bits: u32, part: u8) -> Option<u64> {
        if part >= 4 || predicate_bits == 0 || predicate_bits > 64 {
            return None;
        }
        let pattern = counter & 0xffff;
        let invert = pattern & 0x8000 != 0;
        let marker = pattern & 0xf;
        if marker == 0 {
            return Some(0);
        }
        // The marker's lowest set bit is the element's size: `1 << shift`
        // bytes.
        let shift = marker.trailing_zeros();
        let span = predicate_bits.checked_mul(4)?;
        // The count runs from the bit above the marker to `log2(4 * PL)`.
        let top = span.ilog2();
        let count_bits = top.checked_sub(shift)?;
        let field = BitWidth::new(count_bits).mask();
        let count = pattern.checked_shr(shift.checked_add(1)?)? & field;
        let element_bytes = 1u32.checked_shl(shift)?;
        let elements = span.checked_div(element_bytes)?;
        let first = predicate_bits.checked_mul(u32::from(part))?;
        let mut result = 0u64;
        for element in 0..elements {
            let active = (u64::from(element) < count) != invert;
            let bit = element.checked_mul(element_bytes)?;
            if active && (first..first.checked_add(predicate_bits)?).contains(&bit) {
                result |= 1u64.checked_shl(bit.checked_sub(first)?)?;
            }
        }
        Some(result)
    }

    /// The predicate-as-counter of `count` active elements of `element_bits`
    /// among `elements` ([`Self::PredicateCounter`], Arm's `EncodePredCount`).
    /// `None` for a count past the elements and an element width no marker
    /// names.
    fn counter_encoding(element_bits: u8, elements: u64, count: u64, invert: bool) -> Option<u64> {
        if count > elements {
            return None;
        }
        let shift = match element_bits {
            8 => 0u32,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => return None,
        };
        if count == 0 {
            return Some(0);
        }
        let (count, invert) = if invert {
            (elements.checked_sub(count)?, true)
        } else if count == elements {
            (0, true)
        } else {
            (count, false)
        };
        // The count field is the 14, 13, 12 or 11 bits under the invert bit.
        let width = 14u32.checked_sub(shift)?;
        let field = BitWidth::new(width).low_bits(count);
        let marker = 1u64.checked_shl(shift)?;
        Some(u64::from(invert) << 15 | field.checked_shl(shift.checked_add(1)?)? | marker)
    }

    /// The lanes this kernel computes of vector inputs, element by element:
    /// each lane of the result from the lanes of the inputs at its index,
    /// the inputs read as lanes `element_bits` wide.
    ///
    /// The element-wise kernels are [`Self::CarrylessMultiply`], whose whole
    /// product has lanes twice the element's width, and [`Self::BitDeposit`],
    /// [`Self::BitExtract`] and [`Self::BitGroup`] (SVE2 `bdep`, `bext`,
    /// `bgrp`).
    ///
    /// # Returns
    ///
    /// The result's lanes, or `None` for another kernel, inputs of another
    /// number of lanes, and an element width the kernel does not compute.
    #[must_use]
    pub fn evaluate_lanes(self, inputs: &[VectorConst], element_bits: u32) -> Option<VectorConst> {
        let [first, second] = inputs else {
            return None;
        };
        let (first, second) = (first.relaned(element_bits)?, second.relaned(element_bits)?);
        if first.lanes() != second.lanes() {
            return None;
        }
        let result_bits = match self {
            Self::CarrylessMultiply {
                part: CarrylessPart::Full,
            } => element_bits.checked_mul(2)?,
            Self::CarrylessMultiply { .. }
            | Self::BitDeposit
            | Self::BitExtract
            | Self::BitGroup => element_bits,
            _ => return None,
        };
        let lanes = first
            .lane_values()
            .zip(second.lane_values())
            .map(|(left, right)| match self {
                Self::CarrylessMultiply { part } => part.product(left, right, element_bits),
                _ => self
                    .evaluate_integers(
                        &[
                            (u64::try_from(left).ok()?, element_bits),
                            (u64::try_from(right).ok()?, element_bits),
                        ],
                        element_bits,
                    )
                    .map(u128::from),
            })
            .collect::<Option<Vec<_>>>()?;
        VectorConst::from_lanes(result_bits, lanes)
    }

    /// The results a kernel over a whole register computes of vector
    /// constants, one for each of its outputs in order, its elements
    /// `element_bits` wide where it reads them as elements.
    ///
    /// - [`Self::TernaryLogic`] of `[a, b, c]`: each bit the table's bit at
    ///   `a << 2 | b << 1 | c` of the inputs' bits there (`vpternlogd`).
    /// - [`Self::HorizontalMinPos`] of eight words: the least, the index of
    ///   its first occurrence above it, and the other words zero
    ///   (`phminposuw`).
    /// - [`Self::MultiSad`] of two byte vectors: in each 128-bit half, eight
    ///   words, word `i` the sum of the absolute differences of the four bytes
    ///   of the first source from byte `i` of the block the control's bit 2
    ///   (the upper half's bit 5) selects by the four bytes of the block of the
    ///   second its bits 1 and 0 (bits 4 and 3) select (`mpsadbw`).
    /// - [`Self::DoubleBlockSad`] of two byte vectors: the second's
    ///   doublewords shuffled within each 128-bit lane, doubleword `d` the
    ///   lane's doubleword the selector's bits `2d + 1` and `2d` name, then in
    ///   each quadword four words, each the sum of the absolute differences
    ///   of four bytes of the first, its low four for words 0 and 1 and its
    ///   high four for words 2 and 3, from the four shuffled bytes from the
    ///   quadword's byte `w` for word `w` (`vdbpsadbw`).
    /// - [`Self::ShuffleBits`] of a source and a control: mask bit `8 i + j`
    ///   the bit of the source's quadword `i` that byte `j` of the control's
    ///   quadword `i` names, by its low six bits (`vpshufbitqmb`).
    /// - [`Self::Intersect`] of two vectors: the mask of the first's elements
    ///   that occur in the second, and the mask of the second's that occur in
    ///   the first (`vp2intersectd`).
    /// - [`Self::StringCompare`] of two strings, and for the explicit-length
    ///   forms their lengths: the index or the mask, then the flags, the
    ///   formats, aggregations and polarities the SDM states.
    /// - [`Self::Crypto`]: what [`VectorCryptoKind::evaluate`] computes.
    /// - [`Self::DecimalAdjust`] of the accumulator, then the flags where the
    ///   adjust reads them: the adjusted accumulator, `element_bits` wide, and
    ///   the flags as a 32-bit [`FlagsMask`] word
    ///   ([`BcdAdjustKind::evaluate`]).
    ///
    /// A mask is written whole to its 64-bit register, its bits above the
    /// elements clear.
    ///
    /// # Returns
    ///
    /// The results, or `None` for another kernel and for inputs of another
    /// shape.
    #[must_use]
    pub fn evaluate_registers(
        self,
        inputs: &[VectorConst],
        element_bits: u32,
    ) -> Option<Vec<VectorConst>> {
        let mask = |bits: u64| VectorConst::from_lanes(64, [u128::from(bits)]);
        Some(match (self, inputs) {
            (Self::TernaryLogic { table }, [a, b, c]) => {
                let (width, lane_bits) = (a.bits(), a.lane_bits());
                if b.bits() != width || c.bits() != width || width % 64 != 0 {
                    return None;
                }
                let (a, b, c) = (a.relaned(64)?, b.relaned(64)?, c.relaned(64)?);
                let words = (0..a.lanes()).map(|word| {
                    let (a, b, c) = (a.lane(word), b.lane(word), c.lane(word));
                    (0..8u32)
                        .filter(|index| table >> index & 1 != 0)
                        .fold(0u128, |bits, index| {
                            let pick = |value: u128, set: bool| if set { value } else { !value };
                            bits | pick(a, index & 4 != 0)
                                & pick(b, index & 2 != 0)
                                & pick(c, index & 1 != 0)
                        })
                        & u128::from(u64::MAX)
                });
                vec![VectorConst::from_lanes(64, words.collect::<Vec<_>>())?.relaned(lane_bits)?]
            }
            (Self::HorizontalMinPos, [source]) => {
                let words = source.relaned(16)?;
                if words.lanes() != 8 {
                    return None;
                }
                let (index, least) = (0..8u32)
                    .map(|index| (index, words.lane(index)))
                    .min_by_key(|(index, value)| (*value, *index))?;
                vec![VectorConst::from_lanes(
                    16,
                    [least, u128::from(index), 0, 0, 0, 0, 0, 0],
                )?]
            }
            (Self::MultiSad { offsets }, [first, second]) => {
                let (first, second) = (first.relaned(8)?, second.relaned(8)?);
                if first.lanes() != second.lanes() || first.lanes() % 16 != 0 {
                    return None;
                }
                let mut words = Vec::new();
                for half in 0..first.lanes() / 16 {
                    let control = u32::from(offsets)
                        .checked_shr(half.saturating_mul(3))
                        .unwrap_or(0);
                    let base = half.saturating_mul(16);
                    let first_block = base.saturating_add((control >> 2 & 1).saturating_mul(4));
                    let second_block = base.saturating_add((control & 3).saturating_mul(4));
                    for word in 0..8u32 {
                        words.push(
                            (0..4u32)
                                .map(|byte| {
                                    first
                                        .lane(first_block.saturating_add(word).saturating_add(byte))
                                        .abs_diff(second.lane(second_block.saturating_add(byte)))
                                })
                                .fold(0u128, u128::saturating_add),
                        );
                    }
                }
                vec![VectorConst::from_lanes(16, words)?]
            }
            (Self::DoubleBlockSad { selector }, [first, second]) => {
                let (first, second) = (first.relaned(8)?, second.relaned(8)?);
                if first.lanes() != second.lanes() || first.lanes() % 16 != 0 {
                    return None;
                }
                // The second source's doublewords, shuffled within each lane.
                let shuffled: Vec<u128> = (0..second.lanes())
                    .map(|byte| {
                        let (lane, dword, within) = (byte / 16, byte / 4 % 4, byte % 4);
                        let chosen = u32::from(selector) >> dword.saturating_mul(2) & 3;
                        second.lane(
                            lane.saturating_mul(16)
                                .saturating_add(chosen.saturating_mul(4))
                                .saturating_add(within),
                        )
                    })
                    .collect();
                let mut words = Vec::new();
                for quadword in 0..first.lanes() / 8 {
                    let base = quadword.saturating_mul(8);
                    for word in 0..4u32 {
                        let stationary = base.saturating_add((word / 2).saturating_mul(4));
                        let sliding = base.saturating_add(word);
                        words.push(
                            (0..4u32)
                                .map(|byte| -> Option<u128> {
                                    let at = usize::try_from(sliding.saturating_add(byte)).ok()?;
                                    Some(
                                        first
                                            .lane(stationary.saturating_add(byte))
                                            .abs_diff(*shuffled.get(at)?),
                                    )
                                })
                                .sum::<Option<u128>>()?,
                        );
                    }
                }
                vec![VectorConst::from_lanes(16, words)?]
            }
            (Self::ShuffleBits, [source, control]) => {
                let (source, control) = (source.relaned(64)?, control.relaned(8)?);
                if control.lanes() != source.lanes().saturating_mul(8) || control.lanes() > 64 {
                    return None;
                }
                let bits = (0..control.lanes()).fold(0u64, |bits, byte| {
                    let word = source.lane(byte / 8);
                    let selected = (word >> (control.lane(byte) & 63)) & 1;
                    bits | u64::from(selected == 1) << byte
                });
                vec![mask(bits)?]
            }
            (
                Self::StringCompare {
                    control,
                    explicit_length,
                    result_index,
                },
                inputs,
            ) => Self::string_compare(control, explicit_length, result_index, inputs)?,
            (Self::Crypto(primitive), inputs) => primitive.evaluate(inputs, element_bits)?,
            // The accumulator, then the flags where the adjust reads them.
            (Self::DecimalAdjust { kind, base }, [accumulator, rest @ ..]) => {
                // The flags an add or a subtract adjust reads are its second
                // input; without them it is no computation of known flags.
                let flags = match (kind.reads_flags(), rest) {
                    (false, []) => 0,
                    (true, [flags]) => u16::try_from(flags.lane(0) & 0xffff).ok()?,
                    _ => return None,
                };
                let accumulator = u16::try_from(accumulator.lane(0) & 0xffff).ok()?;
                let (value, flags) =
                    kind.evaluate(base, accumulator, FlagsMask::from_bits(flags))?;
                vec![
                    VectorConst::from_lanes(element_bits, [u128::from(value)])?,
                    VectorConst::from_lanes(32, [u128::from(flags.bits())])?,
                ]
            }
            (Self::Intersect, [first, second]) if matches!(element_bits, 32 | 64) => {
                let (first, second) = (first.relaned(element_bits)?, second.relaned(element_bits)?);
                if first.lanes() != second.lanes() || first.lanes() > 64 {
                    return None;
                }
                let occurs = |value: u128, among: &VectorConst| {
                    among.lane_values().any(|other| other == value)
                };
                let bits = |of: &VectorConst, among: &VectorConst| {
                    of.lane_values()
                        .enumerate()
                        .filter(|(_, value)| occurs(*value, among))
                        .fold(0u64, |bits, (index, _)| bits | 1 << index)
                };
                vec![mask(bits(&first, &second))?, mask(bits(&second, &first))?]
            }
            _ => return None,
        })
    }

    /// The index or mask and the flags of an SSE4.2 packed string compare
    /// (`pcmpistri`, `pcmpistrm`, `pcmpestri`, `pcmpestrm`) by `control`, of
    /// the strings `[a, b]`, and for `explicit_length` their lengths after
    /// them.
    ///
    /// The control's bits 1 and 0 read the strings as sixteen bytes or eight
    /// words, signed or unsigned; a string's length is the absolute value of
    /// its length register, read signed at its width, or its first zero
    /// element's index, at most the element count. Bits 3 and 2 aggregate the
    /// comparisons of the elements within both lengths: equal any, ranges
    /// (the first string's elements in pairs, low then high), equal each (two
    /// elements past both lengths equal) and equal ordered (the first string
    /// found at each element of the second, an element past its own length
    /// matching anything). Bits 5 and 4 keep the result, negate it, or
    /// negate it within the second string's length. Bit 6 names the most
    /// significant set element rather than the least (the element count where
    /// none is set, a 32-bit index), or the mask with an element of ones for
    /// each set one rather than a bit (128 bits). The flags set the carry for a
    /// result not zero, the zero flag for a second string shorter than the
    /// register, the sign flag for a first one, and the overflow flag for the
    /// result's bit 0, as a 32-bit [`FlagsMask`].
    fn string_compare(
        control: u8,
        explicit_length: bool,
        result_index: bool,
        inputs: &[VectorConst],
    ) -> Option<Vec<VectorConst>> {
        let (bits, signed) = (if control & 1 == 0 { 8u32 } else { 16 }, control & 2 != 0);
        let count = 128u32.checked_div(bits)?;
        let (a, b, lengths) = match inputs {
            [a, b] if !explicit_length => (a, b, None),
            [a, b, first, second] if explicit_length => (a, b, Some((first, second))),
            _ => return None,
        };
        let (a, b) = (a.relaned(bits)?, b.relaned(bits)?);
        if a.lanes() != count || b.lanes() != count {
            return None;
        }
        let element = |string: &VectorConst, index: u32| -> i128 {
            if signed {
                string.signed_lane(index)
            } else {
                i128::try_from(string.lane(index)).unwrap_or(i128::MAX)
            }
        };
        let stated = |register: &VectorConst| -> Option<u32> {
            let length = register.signed_lane(0).unsigned_abs();
            u32::try_from(length.min(u128::from(count))).ok()
        };
        let terminated = |string: &VectorConst| {
            (0..count)
                .find(|index| string.lane(*index) == 0)
                .unwrap_or(count)
        };
        let (first_length, second_length) = match lengths {
            Some((first, second)) => (stated(first)?, stated(second)?),
            None => (terminated(&a), terminated(&b)),
        };
        let in_a = |index: u32| index < first_length;
        let in_b = |index: u32| index < second_length;
        let equal = |i: u32, j: u32| element(&a, i) == element(&b, j);
        let mut matched = 0u32;
        for j in 0..count {
            let set = match control >> 2 & 3 {
                0 => (0..count).any(|i| in_a(i) && in_b(j) && equal(i, j)),
                1 => (0..count / 2).any(|pair| {
                    let (low, high) = (
                        pair.saturating_mul(2),
                        pair.saturating_mul(2).saturating_add(1),
                    );
                    in_a(low)
                        && in_a(high)
                        && in_b(j)
                        && element(&b, j) >= element(&a, low)
                        && element(&b, j) <= element(&a, high)
                }),
                2 => match (in_a(j), in_b(j)) {
                    (false, false) => true,
                    (true, true) => equal(j, j),
                    _ => false,
                },
                _ => (0..count.saturating_sub(j)).all(|k| {
                    let at = j.saturating_add(k);
                    match (in_a(k), in_b(at)) {
                        (false, _) => true,
                        (true, false) => false,
                        (true, true) => equal(k, at),
                    }
                }),
            };
            matched |= u32::from(set) << j;
        }
        let all = u32::MAX
            .checked_shr(32u32.saturating_sub(count))
            .unwrap_or(0);
        let within_b = all
            .checked_shr(count.saturating_sub(second_length))
            .unwrap_or(0);
        let result = match control >> 4 & 3 {
            0 | 2 => matched,
            1 => !matched & all,
            _ => (matched ^ within_b) & all,
        };
        let most = control & 0x40 != 0;
        let output = if result_index {
            let index = match (result, most) {
                (0, _) => count,
                (_, true) => 31u32.saturating_sub(result.leading_zeros()),
                (_, false) => result.trailing_zeros(),
            };
            VectorConst::from_lanes(32, [u128::from(index)])?
        } else if most {
            let ones = BitWidth::new(bits).mask_u128();
            VectorConst::from_lanes(
                bits,
                (0..count).map(|j| if result >> j & 1 != 0 { ones } else { 0 }),
            )?
            .relaned(64)?
        } else {
            VectorConst::from_lanes(64, [u128::from(result), 0])?
        };
        let mut flags = 0u16;
        for (set, flag) in [
            (result != 0, FlagsMask::CARRY),
            (second_length < count, FlagsMask::ZERO),
            (first_length < count, FlagsMask::SIGN),
            (result & 1 != 0, FlagsMask::OVERFLOW),
        ] {
            if set {
                flags |= flag.bits();
            }
        }
        Some(vec![
            output,
            VectorConst::from_lanes(32, [u128::from(flags)])?,
        ])
    }
}
