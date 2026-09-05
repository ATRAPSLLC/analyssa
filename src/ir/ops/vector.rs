//! Vector and SIMD descriptors — lane kinds, shuffle and permute payloads,
//! SVE / SME / AMX-tile operands, and the vector operation kind taxonomies.
//!
//! Split from [`super::kinds`] because the vector surface is large enough to
//! read on its own and is consulted as a unit when adding lane semantics.

use crate::ir::{
    ops::{
        effects::{SsaEffectKind, SsaEffects},
        native::FlagsMask,
        operands::impl_kinded_payload,
    },
    variable::SsaVarId,
};

/// Structured identity of a hardware **vector cryptographic** operation
/// ([`SsaOp::VectorCrypto`]) — the first-class, named replacement for the
/// AES / SHA / SM3 / SM4 / GF(2^8) / carry-less-multiply round and message
/// primitives. Each variant names the exact hardware operation (the LLVM-
/// intrinsic model): the lifter never decomposes a round into primitive bit
/// ops nor carries it as an opaque blob. Every variant is a pure function of
/// its vector inputs.
///
/// [`SsaOp::VectorCrypto`]: crate::ir::ops::def::SsaOp::VectorCrypto
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum VectorCryptoKind {
    /// AES encryption round (`aesenc`).
    AesEncrypt,
    /// AES last encryption round (`aesenclast`).
    AesEncryptLast,
    /// AES decryption round (`aesdec`).
    AesDecrypt,
    /// AES last decryption round (`aesdeclast`).
    AesDecryptLast,
    /// AES inverse mix-columns (`aesimc`).
    AesInvMixColumns,
    /// AES round-key generation assist (`aeskeygenassist`).
    AesKeygenAssist,
    /// AES single encryption round without mix-columns: AddRoundKey then
    /// SubBytes and ShiftRows (AArch64 `aese`).
    AesEncryptRound,
    /// AES single decryption round without inverse mix-columns: AddRoundKey
    /// then inverse SubBytes and ShiftRows (AArch64 `aesd`).
    AesDecryptRound,
    /// AES forward mix-columns (AArch64 `aesmc`).
    AesMixColumns,
    /// SHA-1 four rounds (`sha1rnds4`).
    Sha1Rounds4,
    /// SHA-1 next `E` value (`sha1nexte`).
    Sha1NextE,
    /// SHA-1 message schedule step 1 (`sha1msg1`).
    Sha1Msg1,
    /// SHA-1 message schedule step 2 (`sha1msg2`).
    Sha1Msg2,
    /// SHA-256 two rounds (`sha256rnds2`).
    Sha256Rounds2,
    /// SHA-256 message schedule step 1 (`sha256msg1`).
    Sha256Msg1,
    /// SHA-256 message schedule step 2 (`sha256msg2`).
    Sha256Msg2,
    /// SHA-512 two rounds (`vsha512rnds2`).
    Sha512Rounds2,
    /// SHA-512 message schedule step 1 (`vsha512msg1`).
    Sha512Msg1,
    /// SHA-512 message schedule step 2 (`vsha512msg2`).
    Sha512Msg2,
    /// SM3 message schedule step 1 (`vsm3msg1`).
    Sm3Msg1,
    /// SM3 message schedule step 2 (`vsm3msg2`).
    Sm3Msg2,
    /// SM3 two rounds (`vsm3rnds2`).
    Sm3Rounds2,
    /// SM4 key expansion (`vsm4key4`).
    Sm4Key,
    /// SM4 four rounds (`vsm4rnds4`).
    Sm4Rounds,
    /// GF(2^8) affine transform (`gf2p8affineqb`).
    Gf2p8Affine,
    /// GF(2^8) inverse affine transform (`gf2p8affineinvqb`).
    Gf2p8AffineInv,
    /// GF(2^8) multiply (`gf2p8mulb`).
    Gf2p8Mul,
    /// Carry-less (polynomial) multiply (`pclmulqdq`).
    CarrylessMul,
    /// AES key-locker encryption using a wrapped key handle
    /// (`aesenc128kl`/`aesenc256kl` and their wide `aesencwide*kl` peers).
    AesEncryptKeyLocker,
    /// AES key-locker decryption using a wrapped key handle
    /// (`aesdec128kl`/`aesdec256kl` and their wide `aesdecwide*kl` peers).
    AesDecryptKeyLocker,
    /// SHA-1 hash update using the choose function (AArch64 `sha1c`).
    Sha1HashChoose,
    /// SHA-1 hash update using the majority function (AArch64 `sha1m`).
    Sha1HashMajority,
    /// SHA-1 hash update using the parity function (AArch64 `sha1p`).
    Sha1HashParity,
    /// SHA-1 fixed rotate of the working variable (AArch64 `sha1h`).
    Sha1FixedRotate,
    /// SHA-1 message schedule update step 0 (AArch64 `sha1su0`).
    Sha1ScheduleUpdate0,
    /// SHA-1 message schedule update step 1 (AArch64 `sha1su1`).
    Sha1ScheduleUpdate1,
    /// SHA-256 hash update part 1 (AArch64 `sha256h`).
    Sha256Hash,
    /// SHA-256 hash update part 2 (AArch64 `sha256h2`).
    Sha256Hash2,
    /// SHA-256 message schedule update step 0 (AArch64 `sha256su0`).
    Sha256ScheduleUpdate0,
    /// SHA-256 message schedule update step 1 (AArch64 `sha256su1`).
    Sha256ScheduleUpdate1,
    /// SM3 message schedule part 1 (AArch64 `sm3partw1`).
    Sm3PartW1,
    /// SM3 message schedule part 2 (AArch64 `sm3partw2`).
    Sm3PartW2,
    /// SM3 hash update step 1 helper (AArch64 `sm3ss1`).
    Sm3SS1,
    /// SM3 hash update T1 variant A (AArch64 `sm3tt1a`).
    Sm3TT1A,
    /// SM3 hash update T1 variant B (AArch64 `sm3tt1b`).
    Sm3TT1B,
    /// SM3 hash update T2 variant A (AArch64 `sm3tt2a`).
    Sm3TT2A,
    /// SM3 hash update T2 variant B (AArch64 `sm3tt2b`).
    Sm3TT2B,
    /// Polynomial (carry-less) multiply long over GF(2) (AArch64 `pmull`).
    PolynomialMultiply,
}

impl VectorCryptoKind {
    /// Effect summary — every vector-crypto primitive is a pure function of its
    /// inputs (no memory, no trap, no nondeterminism).
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        SsaEffects::new(SsaEffectKind::Pure, false)
    }

    /// Stable display / fingerprint key.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::AesEncrypt => "crypto.aesenc",
            Self::AesEncryptLast => "crypto.aesenclast",
            Self::AesDecrypt => "crypto.aesdec",
            Self::AesDecryptLast => "crypto.aesdeclast",
            Self::AesInvMixColumns => "crypto.aesimc",
            Self::AesKeygenAssist => "crypto.aeskeygenassist",
            Self::AesEncryptRound => "crypto.aese",
            Self::AesDecryptRound => "crypto.aesd",
            Self::AesMixColumns => "crypto.aesmc",
            Self::Sha1Rounds4 => "crypto.sha1rnds4",
            Self::Sha1NextE => "crypto.sha1nexte",
            Self::Sha1Msg1 => "crypto.sha1msg1",
            Self::Sha1Msg2 => "crypto.sha1msg2",
            Self::Sha256Rounds2 => "crypto.sha256rnds2",
            Self::Sha256Msg1 => "crypto.sha256msg1",
            Self::Sha256Msg2 => "crypto.sha256msg2",
            Self::Sha512Rounds2 => "crypto.sha512rnds2",
            Self::Sha512Msg1 => "crypto.sha512msg1",
            Self::Sha512Msg2 => "crypto.sha512msg2",
            Self::Sm3Msg1 => "crypto.sm3msg1",
            Self::Sm3Msg2 => "crypto.sm3msg2",
            Self::Sm3Rounds2 => "crypto.sm3rnds2",
            Self::Sm4Key => "crypto.sm4key",
            Self::Sm4Rounds => "crypto.sm4rnds",
            Self::Gf2p8Affine => "crypto.gf2p8affine",
            Self::Gf2p8AffineInv => "crypto.gf2p8affineinv",
            Self::Gf2p8Mul => "crypto.gf2p8mul",
            Self::CarrylessMul => "crypto.pclmulqdq",
            Self::AesEncryptKeyLocker => "crypto.aesenckl",
            Self::AesDecryptKeyLocker => "crypto.aesdeckl",
            Self::Sha1HashChoose => "crypto.sha1c",
            Self::Sha1HashMajority => "crypto.sha1m",
            Self::Sha1HashParity => "crypto.sha1p",
            Self::Sha1FixedRotate => "crypto.sha1h",
            Self::Sha1ScheduleUpdate0 => "crypto.sha1su0",
            Self::Sha1ScheduleUpdate1 => "crypto.sha1su1",
            Self::Sha256Hash => "crypto.sha256h",
            Self::Sha256Hash2 => "crypto.sha256h2",
            Self::Sha256ScheduleUpdate0 => "crypto.sha256su0",
            Self::Sha256ScheduleUpdate1 => "crypto.sha256su1",
            Self::Sm3PartW1 => "crypto.sm3partw1",
            Self::Sm3PartW2 => "crypto.sm3partw2",
            Self::Sm3SS1 => "crypto.sm3ss1",
            Self::Sm3TT1A => "crypto.sm3tt1a",
            Self::Sm3TT1B => "crypto.sm3tt1b",
            Self::Sm3TT2A => "crypto.sm3tt2a",
            Self::Sm3TT2B => "crypto.sm3tt2b",
            Self::PolynomialMultiply => "crypto.pmull",
        }
    }
}

/// The element interpretation of an AMX tile dot-product ([`TileOpKind`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TileDotKind {
    /// Signed × signed byte dot product (`tdpbssd`).
    Int8SignedSigned,
    /// Signed × unsigned byte dot product (`tdpbsud`).
    Int8SignedUnsigned,
    /// Unsigned × signed byte dot product (`tdpbusd`).
    Int8UnsignedSigned,
    /// Unsigned × unsigned byte dot product (`tdpbuud`).
    Int8UnsignedUnsigned,
    /// BF16 dot product accumulating into f32 (`tdpbf16ps`).
    Bf16,
    /// FP16 dot product accumulating into f32 (`tdpfp16ps`).
    Fp16,
    /// Complex FP16 matrix multiply, real part, accumulating into f32
    /// (`tcmmrlfp16ps`).
    ComplexFp16Real,
    /// Complex FP16 matrix multiply, imaginary part, accumulating into f32
    /// (`tcmmimfp16ps`).
    ComplexFp16Imaginary,
}

/// Structured identity of an AMX (Advanced Matrix Extensions) **tile**
/// operation ([`SsaOp::TileOp`]). Tiles are 2-D matrix registers, distinct
/// from 1-D SIMD vectors; the kind names the exact tile operation and drives a
/// precise effect summary.
///
/// [`SsaOp::TileOp`]: crate::ir::ops::def::SsaOp::TileOp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TileOpKind {
    /// Tile matrix multiply-accumulate (`tdp*`); the [`TileDotKind`] names the
    /// element interpretation. Pure function of the source tiles.
    DotProduct(TileDotKind),
    /// Load a tile from memory (`tileloadd`/`tileloaddt1`).
    Load,
    /// Store a tile to memory (`tilestored`).
    Store,
    /// Zero a tile register (`tilezero`). Pure.
    Zero,
    /// Load the tile configuration from memory (`ldtilecfg`).
    LoadConfig,
    /// Store the tile configuration to memory (`sttilecfg`).
    StoreConfig,
    /// Release all tile state (`tilerelease`).
    Release,
}

impl TileOpKind {
    /// Precise effect summary: dot-product / zero are pure; loads read memory,
    /// stores write memory; `tilerelease` mutates architectural tile state.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::DotProduct(_) | Self::Zero => SsaEffects::new(SsaEffectKind::Pure, false),
            Self::Load | Self::LoadConfig => SsaEffects::new(SsaEffectKind::Read, false),
            Self::Store | Self::StoreConfig | Self::Release => {
                SsaEffects::new(SsaEffectKind::Write, false)
            }
        }
    }

    /// Stable display / fingerprint key.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::DotProduct(TileDotKind::Int8SignedSigned) => "tile.dp.bssd",
            Self::DotProduct(TileDotKind::Int8SignedUnsigned) => "tile.dp.bsud",
            Self::DotProduct(TileDotKind::Int8UnsignedSigned) => "tile.dp.busd",
            Self::DotProduct(TileDotKind::Int8UnsignedUnsigned) => "tile.dp.buud",
            Self::DotProduct(TileDotKind::Bf16) => "tile.dp.bf16ps",
            Self::DotProduct(TileDotKind::Fp16) => "tile.dp.fp16ps",
            Self::DotProduct(TileDotKind::ComplexFp16Real) => "tile.cmm.rlfp16ps",
            Self::DotProduct(TileDotKind::ComplexFp16Imaginary) => "tile.cmm.imfp16ps",
            Self::Load => "tile.load",
            Self::Store => "tile.store",
            Self::Zero => "tile.zero",
            Self::LoadConfig => "tile.ldcfg",
            Self::StoreConfig => "tile.stcfg",
            Self::Release => "tile.release",
        }
    }
}

/// Boxed payload for [`SsaOp::VectorPermute`] — a lane permute whose selector
/// is a **runtime index vector** (unlike [`SsaOp::VectorShuffle`], whose mask is
/// static). The `inputs` are the data source vector(s) followed by the index
/// vector (`[src, index]` for single-source `vpermd`/`vpshufb`, `[src1, src2,
/// index]` for the two-source `vpermt2*`/`vpermi2*`). Pure.
///
/// [`SsaOp::VectorPermute`]: crate::ir::ops::def::SsaOp::VectorPermute
/// [`SsaOp::VectorShuffle`]: crate::ir::ops::def::SsaOp::VectorShuffle
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPermuteData {
    /// Explicit SSA outputs defined by the operation (the permuted vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the data source vector(s) then the index vector.
    pub inputs: Vec<SsaVarId>,
}

/// The specific fused multiply-then-horizontal-add operation named by a
/// [`SsaOp::VectorMultiplyAdd`].
///
/// [`SsaOp::VectorMultiplyAdd`]: crate::ir::ops::def::SsaOp::VectorMultiplyAdd
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum VectorMaddKind {
    /// `pmaddwd`: signed 16×16→32 products, adjacent pairs summed.
    MultiplyAddS16,
    /// `pmaddubsw`: unsigned×signed 8×8→16 products, adjacent pairs summed with
    /// signed saturation.
    MultiplyAddU8S8Sat,
    /// `vpdpbusd`: unsigned×signed byte products, groups of four summed into a
    /// 32-bit accumulator (the destination).
    DotProductU8S8,
    /// `vpdpbusds`: as [`Self::DotProductU8S8`] with signed-saturating
    /// accumulation.
    DotProductU8S8Sat,
    /// `vpdpwssd`: signed word products, pairs summed into a 32-bit accumulator.
    DotProductS16,
    /// `vpdpwssds`: as [`Self::DotProductS16`] with signed-saturating
    /// accumulation.
    DotProductS16Sat,
    /// `vpmadd52luq`: low 52 bits of unsigned 52×52 products added to a 64-bit
    /// accumulator.
    MultiplyAdd52Lo,
    /// `vpmadd52huq`: high 52 bits of unsigned 52×52 products added to a 64-bit
    /// accumulator.
    MultiplyAdd52Hi,
    /// `vdpbf16ps`: bf16 pairs multiplied and summed into a 32-bit float
    /// accumulator.
    DotProductBf16,
    /// AMD XOP `vpmacs*`: per-lane signed multiply added to a third
    /// accumulator source.
    MultiplyAccumulate,
    /// AMD XOP `vpmacss*`: as [`Self::MultiplyAccumulate`] with signed-saturating
    /// accumulation.
    MultiplyAccumulateSat,
    /// AMD XOP `vpmadcswd`: signed word products summed in adjacent pairs and
    /// added to a third accumulator source.
    MultiplyAccumulatePairs,
    /// AMD XOP `vpmadcsswd`: as [`Self::MultiplyAccumulatePairs`] with
    /// signed-saturating accumulation.
    MultiplyAccumulatePairsSat,
    /// `vpdpbssd[s]`: signed×signed byte dot-product into a 32-bit accumulator.
    DotProductS8S8,
    /// `vpdpbssds`: saturating signed×signed byte dot-product.
    DotProductS8S8Sat,
    /// `vpdpbsud[s]`: signed×unsigned byte dot-product.
    DotProductS8U8,
    /// `vpdpbsuds`: saturating signed×unsigned byte dot-product.
    DotProductS8U8Sat,
    /// `vpdpbuud[s]`: unsigned×unsigned byte dot-product.
    DotProductU8U8,
    /// `vpdpbuuds`: saturating unsigned×unsigned byte dot-product.
    DotProductU8U8Sat,
    /// `vpdpwsud[s]`: signed×unsigned word dot-product.
    DotProductS16U16,
    /// `vpdpwsuds`: saturating signed×unsigned word dot-product.
    DotProductS16U16Sat,
    /// `vpdpwusd[s]`: unsigned×signed word dot-product.
    DotProductU16S16,
    /// `vpdpwusds`: saturating unsigned×signed word dot-product.
    DotProductU16S16Sat,
    /// `vpdpwuud[s]`: unsigned×unsigned word dot-product.
    DotProductU16U16,
    /// `vpdpwuuds`: saturating unsigned×unsigned word dot-product.
    DotProductU16U16Sat,
}

/// The specific complex-number floating-point multiply named by a
/// [`SsaOp::VectorComplexMul`]. Each lane pair holds a `(real, imag)` complex
/// value.
///
/// [`SsaOp::VectorComplexMul`]: crate::ir::ops::def::SsaOp::VectorComplexMul
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum ComplexMulKind {
    /// `vfmulcph`/`vfmulcsh`: complex multiply `a * b`.
    Multiply,
    /// `vfcmulcph`/`vfcmulcsh`: complex multiply by the conjugate `a * conj(b)`.
    ConjugateMultiply,
    /// `vfmaddcph`/`vfmaddcsh`: complex multiply-accumulate `acc + a * b`.
    MultiplyAdd,
    /// `vfcmaddcph`/`vfcmaddcsh`: conjugate multiply-accumulate
    /// `acc + a * conj(b)`.
    ConjugateMultiplyAdd,
}

impl ComplexMulKind {
    /// Returns the stable textual identity used in similarity / display.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Multiply => "vector.cmul",
            Self::ConjugateMultiply => "vector.cmul.conj",
            Self::MultiplyAdd => "vector.cmadd",
            Self::ConjugateMultiplyAdd => "vector.cmadd.conj",
        }
    }
}

/// Boxed payload for [`SsaOp::VectorDotProduct`] — a masked floating-point dot
/// product (x86 SSE4.1 `dpps`/`dppd` and VEX `vdpps`/`vdppd`). The 8-bit
/// immediate selects which source lanes participate in the sum (high nibble)
/// and which destination lanes receive the broadcast result (low nibble);
/// `element_bits` is the lane width (32 for `*ps`, 64 for `*pd`). The `inputs`
/// are the two source vectors `[a, b]` (the destination doubles as `a`). Pure.
///
/// [`SsaOp::VectorDotProduct`]: crate::ir::ops::def::SsaOp::VectorDotProduct
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorDotProductData {
    /// The 8-bit lane-participation / result-broadcast control immediate.
    pub imm8: u8,
    /// Lane width in bits (32 for `*ps`, 64 for `*pd`).
    pub element_bits: u16,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two source vectors `[a, b]`.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorIntDotProduct`] — an integer grouped
/// dot-product-accumulate (ARM/AArch64 `sdot`/`udot`/`usdot`): each output lane
/// sums the products of a contiguous group of `dest_bits / source_bits`
/// source-element pairs (widened from `source_bits`), added into the destination
/// lane. `signed_a`/`signed_b` give the per-operand sign-extension (they differ
/// only for the mixed `usdot`). The `inputs` are `[acc, a, b]` (the destination
/// accumulator and the two source vectors). Pure.
///
/// [`SsaOp::VectorIntDotProduct`]: crate::ir::ops::def::SsaOp::VectorIntDotProduct
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorIntDotProductData {
    /// `true` when the first source's elements are sign-extended.
    pub signed_a: bool,
    /// `true` when the second source's elements are sign-extended.
    pub signed_b: bool,
    /// Source element width in bits (the group size is `dest_bits / source_bits`).
    pub source_bits: u16,
    /// Destination (accumulator) lane width in bits.
    pub dest_bits: u16,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: `[acc, a, b]`.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorShuffleBits`] — an AVX-512 VBMI2
/// `vpshufbitqmb`: for each byte it gathers the source bit at the position
/// named by the corresponding control byte into a mask. The `outputs` are the
/// result mask; the `inputs` are the source and the control vectors. Pure.
///
/// [`SsaOp::VectorShuffleBits`]: crate::ir::ops::def::SsaOp::VectorShuffleBits
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorShuffleBitsData {
    /// Explicit SSA outputs: the result mask.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source and the control vectors.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorIntersect`] — an AVX-512 VP2INTERSECT
/// (`vp2intersectd`/`vp2intersectq`). It computes a pair of masks: the first
/// marks which elements of the first source also appear in the second, the
/// second marks the converse. The `outputs` are the two mask registers; the
/// `inputs` are the two source vectors. Pure.
///
/// [`SsaOp::VectorIntersect`]: crate::ir::ops::def::SsaOp::VectorIntersect
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorIntersectData {
    /// Explicit SSA outputs: the two result masks `[m1, m2]`.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two source vectors `[a, b]`.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorBitfield`] — an SSE4a bit-field extract /
/// insert over the low 64 bits of a vector register (`extrq`/`insertq`).
/// `insert` selects insert (`insertq`) vs extract (`extrq`); `index` and
/// `length` are the bit position and width for the immediate-controlled forms
/// (both zero for the register-controlled forms, whose control vector is an
/// additional input). Pure.
///
/// [`SsaOp::VectorBitfield`]: crate::ir::ops::def::SsaOp::VectorBitfield
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorBitfieldData {
    /// `true` for `insertq` (insert), `false` for `extrq` (extract).
    pub insert: bool,
    /// Bit position of the field for the immediate-controlled forms.
    pub index: u8,
    /// Bit width of the field for the immediate-controlled forms.
    pub length: u8,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the destination vector, the source / control
    /// vector, and (for the register-controlled forms) the control vector.
    pub inputs: Vec<SsaVarId>,
}

/// Per-byte condition selecting which lanes a [`SsaOp::VectorConditionalMove`]
/// updates (Cyrix EMMI `pmv*` family). The exact test reference is part of the
/// underdocumented EMMI semantics; the variant names the documented predicate.
///
/// [`SsaOp::VectorConditionalMove`]: crate::ir::ops::def::SsaOp::VectorConditionalMove
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ByteMoveCondition {
    /// Move where the tested byte is zero (`pmvzb`).
    Zero,
    /// Move where the tested byte is non-zero (`pmvnzb`).
    NonZero,
    /// Move where the tested byte is negative (`pmvlzb`).
    Negative,
    /// Move where the tested byte is non-negative (`pmvgezb`).
    NonNegative,
}

impl ByteMoveCondition {
    /// Stable display / fingerprint key.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Zero => "z",
            Self::NonZero => "nz",
            Self::Negative => "lz",
            Self::NonNegative => "gez",
        }
    }
}

/// Boxed payload for [`SsaOp::VectorConditionalMove`] — a Cyrix EMMI per-byte
/// conditional move (`pmvzb`/`pmvnzb`/`pmvlzb`/`pmvgezb`): selected bytes of the
/// source replace the destination under the [`ByteMoveCondition`]. The `inputs`
/// are the destination and source vectors. Pure.
///
/// [`SsaOp::VectorConditionalMove`]: crate::ir::ops::def::SsaOp::VectorConditionalMove
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorConditionalMoveData {
    /// The per-byte move condition.
    pub condition: ByteMoveCondition,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the destination and source vectors.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorStringCompare`] — an SSE4.2 packed string
/// comparison (`pcmpestri`/`pcmpestrm`/`pcmpistri`/`pcmpistrm` and their VEX
/// peers). The 8-bit immediate selects the data format, aggregation function
/// and polarity; `explicit_length` distinguishes the explicit-length `e*`
/// forms (which consume the two index registers) from the implicit
/// null-terminated `i*` forms; `result_index` distinguishes the index-result
/// `*stri` forms from the mask-result `*strm` forms. The `inputs` are the two
/// string vectors (plus the two length registers for the explicit forms); the
/// `outputs` are the result register and the comparison flags. Pure.
///
/// [`SsaOp::VectorStringCompare`]: crate::ir::ops::def::SsaOp::VectorStringCompare
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorStringCompareData {
    /// The 8-bit format / aggregation / polarity control immediate.
    pub imm8: u8,
    /// `true` for the explicit-length `pcmpestr*` forms (consume two index
    /// registers); `false` for the implicit null-terminated `pcmpistr*` forms.
    pub explicit_length: bool,
    /// `true` for the index-result `*stri` forms; `false` for the mask-result
    /// `*strm` forms.
    pub result_index: bool,
    /// Explicit SSA outputs: the result register and the comparison flags.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two string vectors (plus the two length
    /// registers for the explicit-length forms).
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorHorizontalMinPos`] — the SSE4.1
/// `phminposuw` horizontal minimum: it finds the smallest unsigned 16-bit lane
/// of the single source, placing that minimum in the low word of the result
/// and its source index in the next field (remaining lanes zeroed). The single
/// input is the source vector. Pure.
///
/// [`SsaOp::VectorHorizontalMinPos`]: crate::ir::ops::def::SsaOp::VectorHorizontalMinPos
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorHorizontalMinPosData {
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorHorizontalReduce`] — a grouped, widening
/// horizontal add / subtract (AMD XOP `vphadd*`/`vphsub*`): each output lane is
/// the sum (or difference) of a contiguous group of `dest_bits / source_bits`
/// input lanes, widened from `source_bits` to `dest_bits`. The single input is
/// the source vector. Pure.
///
/// [`SsaOp::VectorHorizontalReduce`]: crate::ir::ops::def::SsaOp::VectorHorizontalReduce
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorHorizontalReduceData {
    /// `true` for horizontal subtract (`vphsub*`); `false` for add.
    pub subtract: bool,
    /// `true` when the source lanes are zero-extended; `false` sign-extended.
    pub unsigned: bool,
    /// Source lane width in bits.
    pub source_bits: u16,
    /// Destination (widened) lane width in bits.
    pub dest_bits: u16,
    /// Explicit SSA outputs defined by the operation (the result vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorPackNarrow`] — a two-source saturating
/// narrowing pack (`packsswb`/`packssdw`/`packuswb`/`packusdw`). Each source's
/// lanes are narrowed to half width with signed or unsigned saturation, the
/// first source filling the low half of each 128-bit granule and the second the
/// high half. The `inputs` are `[src1, src2]`. Pure.
///
/// [`SsaOp::VectorPackNarrow`]: crate::ir::ops::def::SsaOp::VectorPackNarrow
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPackNarrowData {
    /// `true` for unsigned saturation (`packuswb`/`packusdw`); `false` for
    /// signed (`packsswb`/`packssdw`).
    pub unsigned: bool,
    /// Explicit SSA outputs defined by the operation (the packed vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two source vectors.
    pub inputs: Vec<SsaVarId>,
}

/// The SVE data-movement operation for [`SsaOp::VectorSvePermute`].
///
/// [`SsaOp::VectorSvePermute`]: crate::ir::ops::def::SsaOp::VectorSvePermute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SvePermuteKind {
    /// Generate a monotonically increasing index vector (`index`).
    Index,
    /// Insert a scalar at the bottom, shifting the vector up (`insr`).
    InsertShift,
    /// Pack the active elements into the low lanes (`compact`).
    Compact,
    /// Splice two vectors at the last active predicate element (`splice`).
    Splice,
    /// Extract the last active element to a scalar (`lasta`).
    ExtractLastActive,
    /// Extract the element before the last active to a scalar (`lastb`).
    ExtractLastBefore,
    /// Copy the last active element across the destination lanes (`clasta`).
    CopyLastActive,
    /// Copy the element before the last active across the lanes (`clastb`).
    CopyLastBefore,
}

/// The floating-point helper operation for [`SsaOp::VectorFpHelper`] (SVE
/// transcendental-acceleration primitives).
///
/// [`SsaOp::VectorFpHelper`]: crate::ir::ops::def::SsaOp::VectorFpHelper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FpHelperKind {
    /// Reciprocal exponent (`frecpx`).
    ReciprocalExponent,
    /// Extract the biased exponent (`flogb`).
    ExtractExponent,
    /// Exponential acceleration table lookup (`fexpa`).
    ExpAccelerate,
    /// Trigonometric multiply-add (`ftmad`).
    TrigMulAdd,
    /// Trigonometric select-multiply (`ftsmul`).
    TrigSelectMul,
    /// Trigonometric select (`ftssel`).
    TrigSelect,
}

/// The predicate-generating operation for [`SsaOp::VectorPredicateGen`].
///
/// [`SsaOp::VectorPredicateGen`]: crate::ir::ops::def::SsaOp::VectorPredicateGen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PredicateGenKind {
    /// All-true predicate, optionally restricted by a count pattern (`ptrue`).
    True,
    /// All-false predicate (`pfalse`).
    False,
    /// Advance to the next active element (`pnext`).
    Next,
    /// Select the first active element (`pfirst`).
    First,
    /// Read the first-fault register into a predicate (`rdffr`).
    ReadFfr,
    /// Unpack and widen the high half of a predicate (`punpkhi`).
    UnpackHi,
    /// Unpack and widen the low half of a predicate (`punpklo`).
    UnpackLo,
    /// Predicate-wise select between two source predicates (`psel`).
    Select,
    /// Read-after-write address-hazard predicate (`whilerw`).
    HazardRw,
    /// Write-after-read address-hazard predicate (`whilewr`).
    HazardWr,
}

/// Boxed payload for [`SsaOp::VectorSmeOuterProduct`] — an SME outer-product
/// accumulate into a ZA tile (AArch64 `smopa`/`umopa`/`fmopa`/`bfmopa`/`usmopa`/
/// `sumopa` and the subtracting `*mops`). The ZA tile is the accumulator
/// (`inputs[0]`), followed by the governing predicates and the two source
/// vectors; the result is the updated tile. Pure.
///
/// [`SsaOp::VectorSmeOuterProduct`]: crate::ir::ops::def::SsaOp::VectorSmeOuterProduct
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSmeOuterProductData {
    /// `true` for the subtracting `*mops` forms.
    pub subtract: bool,
    /// `true` when the first source is signed.
    pub signed_a: bool,
    /// `true` when the second source is signed.
    pub signed_b: bool,
    /// `true` for the floating-point forms (`fmopa`/`bfmopa`).
    pub float: bool,
    /// Explicit SSA outputs (the updated ZA tile).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the ZA accumulator, governing predicates, and sources.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorMatrixMulAcc`] — an integer/floating matrix
/// multiply-accumulate over vector registers (AArch64 `smmla`/`ummla`/`usmmla`/
/// `fmmla`/`bfmmla`): two source matrices are multiplied and accumulated into the
/// destination. The `inputs` are `[acc, a, b]`. Pure.
///
/// [`SsaOp::VectorMatrixMulAcc`]: crate::ir::ops::def::SsaOp::VectorMatrixMulAcc
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorMatrixMulAccData {
    /// `true` when the first source matrix is signed.
    pub signed_a: bool,
    /// `true` when the second source matrix is signed.
    pub signed_b: bool,
    /// `true` for the floating-point forms (`fmmla`/`bfmmla`).
    pub float: bool,
    /// Explicit SSA outputs (the accumulated result).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the accumulator and the two source matrices.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorReverseChunks`] — reverses the order of
/// `chunk_bits`-sized chunks within each vector element (AArch64 SVE
/// `revb`/`revh`/`revw`/`revd`: byte/halfword/word/doubleword order reversal). The
/// element width comes from the operand register shape. The `inputs` are `[src]`.
/// Pure.
///
/// [`SsaOp::VectorReverseChunks`]: crate::ir::ops::def::SsaOp::VectorReverseChunks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorReverseChunksData {
    /// Size in bits of each reversed chunk (8/16/32/64).
    pub chunk_bits: u16,
    /// Explicit SSA outputs (the reversed vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorCountAdjust`] — adjusts a register by an
/// implicit count of vector elements (`VL / element_bits`) or active predicate
/// lanes, optionally saturating (AArch64 SVE `sqincb`/`uqincd`/`sqincp`/`incp`/
/// `decp`/…). The count is symbolic (the vector length is runtime-defined); the op
/// names the adjustment so no concrete materialization is needed. `inputs[0]` is
/// the value being adjusted, optionally followed by the governing predicate. Pure.
///
/// [`SsaOp::VectorCountAdjust`]: crate::ir::ops::def::SsaOp::VectorCountAdjust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorCountAdjustData {
    /// `true` to subtract the count (`*dec*`), `false` to add (`*inc*`).
    pub decrement: bool,
    /// `true` for the saturating `sq*`/`uq*` forms.
    pub saturate: bool,
    /// `true` for signed saturation (`sq*`), `false` for unsigned (`uq*`).
    pub signed: bool,
    /// `true` when the count is the active-predicate population (`incp`/`sqincp`/…),
    /// `false` when it is the element count `VL / element_bits`.
    pub by_predicate: bool,
    /// Element width selecting the count granularity (`8`/`16`/`32`/`64`).
    pub element_bits: u16,
    /// Explicit SSA outputs (the adjusted value).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the adjusted value, then any governing predicate.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorExtendInLane`] — sign- or zero-extends the
/// low `source_bits` of each `element_bits`-wide lane to the full lane width
/// (AArch64 SVE predicated `sxtb`/`uxtb`/`sxth`/`uxth`/`sxtw`/`uxtw`, and the
/// `sxtw`/`uxtw` index extension performed by SVE `adr`). `inputs[0]` is the
/// source vector, optionally followed by a governing predicate. Pure.
///
/// [`SsaOp::VectorExtendInLane`]: crate::ir::ops::def::SsaOp::VectorExtendInLane
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorExtendInLaneData {
    /// `true` to sign-extend (`sxt*`), `false` to zero-extend (`uxt*`).
    pub signed: bool,
    /// Width in bits of the low field that is extended (8/16/32).
    pub source_bits: u16,
    /// Full lane width in bits (16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the extended vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector, then any governing predicate.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorElementCount`] — computes the element count
/// `VL / element_bits` of the current vector length, scaled by `multiplier`, into
/// a scalar register (AArch64 SVE `cntb`/`cnth`/`cntw`/`cntd`). The count is
/// symbolic (the vector length is runtime-defined); the op names the granularity
/// so no concrete materialization is needed. Pure.
///
/// [`SsaOp::VectorElementCount`]: crate::ir::ops::def::SsaOp::VectorElementCount
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorElementCountData {
    /// Element width selecting the count granularity (`8`/`16`/`32`/`64`).
    pub element_bits: u16,
    /// Constant multiplier from the `mul #imm` form (`1` when absent).
    pub multiplier: u32,
    /// Explicit SSA outputs (the scalar count).
    pub outputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorSveAddressGen`] — SVE vector address
/// generation (`adr`): each destination lane is `base + (extend(index) << shift)`.
/// `inputs[0]` is the base vector, `inputs[1]` the index vector. The optional
/// 32→64 index extension and the left-shift amount are named on the op so the
/// per-lane address computation stays lossless. Pure.
///
/// [`SsaOp::VectorSveAddressGen`]: crate::ir::ops::def::SsaOp::VectorSveAddressGen
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSveAddressGenData {
    /// `Some(true)` for the `sxtw` (signed) 32→64 index extension, `Some(false)`
    /// for `uxtw` (unsigned), `None` for the plain `lsl` form (no extension).
    pub signed_extend: Option<bool>,
    /// Left-shift amount applied to the index lanes (`0..=3`).
    pub shift: u8,
    /// Explicit SSA outputs (the address vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the base vector, then the index vector.
    pub inputs: Vec<SsaVarId>,
}

/// The SVE2/NEON compute operation named by [`VectorSveComputeData`]. One
/// variant per hardware operation keeps the lift lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SveComputeKind {
    /// `adclb`/`adclt`: add with carry into the bottom/top long lanes.
    AddCarryBottom,
    /// Add with carry into the top long lanes.
    AddCarryTop,
    /// `sbclb`/`sbclt`: subtract with carry into the bottom/top long lanes.
    SubCarryBottom,
    /// Subtract with carry into the top long lanes.
    SubCarryTop,
    /// `bdep`: gather selected bits (PDEP-like).
    BitDeposit,
    /// `bext`: scatter selected bits (PEXT-like).
    BitExtract,
    /// `bgrp`: group bits by a mask.
    BitGroup,
    /// `histcnt`: per-element histogram count.
    Histogram,
    /// `histseg`: segmented histogram match count.
    HistogramSegment,
    /// `match`: per-element membership predicate.
    MatchElements,
    /// `nmatch`: per-element non-membership predicate.
    NoMatchElements,
    /// `sclamp`: clamp each lane to a signed `[min,max]` range.
    ClampSigned,
    /// `uclamp`: clamp each lane to an unsigned `[min,max]` range.
    ClampUnsigned,
    /// `sdivr`: reversed signed divide (`b / a`).
    DivideReversedSigned,
    /// `udivr`: reversed unsigned divide (`b / a`).
    DivideReversedUnsigned,
    /// `eorbt`: interleaving exclusive-or, bottom from top.
    InterleaveXorBottomTop,
    /// `eortb`: interleaving exclusive-or, top from bottom.
    InterleaveXorTopBottom,
    /// `sqabs`: saturating absolute value.
    SaturatingAbs,
    /// `sqneg`: saturating negate.
    SaturatingNeg,
    /// `cnot`: logical NOT per lane (`x == 0 ? 1 : 0`).
    LogicalNot,
    /// `cdot`: complex integer dot product with rotation.
    ComplexDotProduct,
    /// `sqrdcmlah`: saturating rounding doubling complex multiply-accumulate.
    ComplexMulAddRounding,
}

/// Boxed payload for [`SsaOp::VectorSveCompute`] — an SVE2/NEON compute op named
/// precisely by its [`SveComputeKind`]. `rotation` carries the complex-op rotation
/// in degrees (0/90/180/270; 0 otherwise); `element_bits` is the lane width. Pure.
///
/// [`SsaOp::VectorSveCompute`]: crate::ir::ops::def::SsaOp::VectorSveCompute
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSveComputeData {
    /// The exact hardware operation.
    pub op: SveComputeKind,
    /// Lane width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Complex-op rotation in degrees (0/90/180/270); 0 when not applicable.
    pub rotation: u16,
    /// Explicit SSA outputs.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs.
    pub inputs: Vec<SsaVarId>,
}

/// The predicate/FFR operation named by [`VectorPredicateOpData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PredicateOpKind {
    /// `cntp`: count active predicate elements into a scalar.
    CountActive,
    /// `ptest`: set condition flags from a predicate.
    Test,
    /// `setffr`: initialise the first-fault register to all-true.
    SetFirstFault,
    /// `wrffr`: write a predicate into the first-fault register.
    WriteFirstFault,
}

/// Boxed payload for [`SsaOp::VectorPredicateOp`] — an SVE predicate/first-fault
/// register operation named by its [`PredicateOpKind`]. Pure.
///
/// [`SsaOp::VectorPredicateOp`]: crate::ir::ops::def::SsaOp::VectorPredicateOp
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPredicateOpData {
    /// The exact predicate/FFR operation.
    pub op: PredicateOpKind,
    /// Governing element width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (scalar count / flags / FFR; may be empty).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (source predicate(s)).
    pub inputs: Vec<SsaVarId>,
}

/// The SME tile operation named by [`VectorSmeMiscData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SmeMiscKind {
    /// `addha`: horizontal add of vector elements into a ZA tile slice.
    AddHorizontal,
    /// `addva`: vertical add of vector elements into a ZA tile slice.
    AddVertical,
    /// `zero`: zero a list of ZA tiles.
    ZeroTiles,
}

/// Boxed payload for [`SsaOp::VectorSmeMisc`] — an SME ZA-tile accumulate/zero
/// operation named by its [`SmeMiscKind`]. Pure (models ZA as explicit operands).
///
/// [`SsaOp::VectorSmeMisc`]: crate::ir::ops::def::SsaOp::VectorSmeMisc
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorSmeMiscData {
    /// The exact SME operation.
    pub op: SmeMiscKind,
    /// Element width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the updated ZA tile).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (governing predicates and source vectors).
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorStructLoadReplicate`] — AArch64 `ld2r`/`ld3r`/
/// `ld4r`: load a single 2/3/4-element structure from memory and replicate each
/// element across all lanes of its destination register.
///
/// Reads memory through its address base and may fault, so it classifies as
/// [`SsaEffectKind::Read`] with [`TrapClass::MemoryFault`] — never pure.
///
/// [`SsaOp::VectorStructLoadReplicate`]: crate::ir::ops::def::SsaOp::VectorStructLoadReplicate
/// [`TrapClass::MemoryFault`]: crate::ir::ops::effects::TrapClass::MemoryFault
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorStructLoadReplicateData {
    /// Number of structure elements / destination registers (2, 3, or 4).
    pub count: u8,
    /// Element width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the replicated destination registers).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (the address base register).
    pub inputs: Vec<SsaVarId>,
}

/// The condition-flag (PSTATE) manipulation named by [`KindedVecData`]. Mirrors
/// the cleaned-IR `FlagAdjustKind`.
///
/// **Partition rule against [`MachineStateOp`].** This enum is the home for an
/// operation whose *entire* effect is writing bits of the flags register
/// [`FlagsMask`] models. IF is EFLAGS bit 9 exactly as DF is bit 10 and AC is
/// bit 18, so `cli` and `sti` belong here beside `cld`/`std` and `clac`/`stac`.
/// Everything else is a [`MachineStateOp`]: `clui`/`stui` toggle UIF, which is
/// separate user-interrupt state [`FlagsMask`] cannot express, and MIPS
/// `di`/`ei` optionally return the prior status register.
///
/// [`MachineStateOp`]: super::MachineStateOp
///
/// [`KindedVecData`]: crate::ir::ops::kinds::KindedVecData
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum FlagAdjustKind {
    /// `cfinv` — invert the carry flag.
    InvertCarry,
    /// `rmif` — rotate a register, mask, and insert into NZCV.
    RotateMaskInsert,
    /// `setf8` — set N and Z from the low 8 bits of a register.
    SetNzFrom8,
    /// `setf16` — set N and Z from the low 16 bits of a register.
    SetNzFrom16,
    /// `axflag` — convert PSTATE flags to the alternative (FP) format.
    ConvertToFpFlags,
    /// `xaflag` — convert alternative (FP) flags back to PSTATE format.
    ConvertFromFpFlags,
    /// Clear the carry flag (`clc`).
    ClearCarry,
    /// Set the carry flag (`stc`).
    SetCarry,
    /// Clear the string-direction flag (`cld`) — string operations advance.
    ClearDirection,
    /// Set the string-direction flag (`std`) — string operations retreat.
    SetDirection,
    /// Clear the alignment-check flag (`clac`).
    ClearAlignCheck,
    /// Set the alignment-check flag (`stac`).
    SetAlignCheck,
    /// Clear the interrupt-enable flag (`cli`) — maskable interrupts are held
    /// off until IF is set again.
    ClearInterrupt,
    /// Set the interrupt-enable flag (`sti`).
    SetInterrupt,
    /// Load the low status-flag byte from a register (`sahf`) — the input
    /// register supplies `SF`, `ZF`, `AF`, `PF` and `CF`; the output is the
    /// updated flags value.
    LoadStatusFromReg,
    /// Store the low status-flag byte into a register (`lahf`) — the input is
    /// the flags value and the output is the written register, so this is the
    /// one adjustment whose result is not a flags value.
    StoreStatusToReg,
    /// Set every bit of a register from the carry flag (`salc`) — the input is
    /// the flags value and the output is the written register.
    SetRegFromCarry,
}

impl FlagAdjustKind {
    /// Returns the architectural flag bits this adjustment writes.
    ///
    /// Callers use it to decide whether a pending comparison survives the
    /// adjustment: an op that writes only control flags (`cld`, `cli`, `stac`)
    /// leaves a `cmp`/`jcc` pair foldable, while one that touches the status
    /// flags does not.
    #[must_use]
    pub const fn flags_written(self) -> FlagsMask {
        match self {
            Self::InvertCarry | Self::ClearCarry | Self::SetCarry => FlagsMask::CARRY,
            Self::ClearDirection | Self::SetDirection => FlagsMask::DIRECTION,
            Self::ClearAlignCheck | Self::SetAlignCheck => FlagsMask::ALIGN_CHECK,
            Self::ClearInterrupt | Self::SetInterrupt => FlagsMask::INTERRUPT,
            // `sahf` restores SF/ZF/AF/PF/CF; OF is not in the byte it loads.
            Self::LoadStatusFromReg => FlagsMask::SIGN
                .union(FlagsMask::ZERO)
                .union(FlagsMask::ADJUST)
                .union(FlagsMask::PARITY)
                .union(FlagsMask::CARRY),
            // AArch64 `setf8`/`setf16` (FEAT_FlagM) write N, Z **and V**, and
            // leave C alone. V is the point of the instruction: it is set from
            // `op<msb+1> EOR op<msb>`, which is how a narrow add's overflow is
            // recovered. Omitting it here reports a stale V as still live, so a
            // caller folds a `b.vs` against a flag `setf8` had overwritten.
            Self::SetNzFrom8 | Self::SetNzFrom16 => FlagsMask::SIGN
                .union(FlagsMask::ZERO)
                .union(FlagsMask::OVERFLOW),
            Self::RotateMaskInsert | Self::ConvertToFpFlags | Self::ConvertFromFpFlags => {
                FlagsMask::x86_status()
            }
            // Both read the flags and write a register instead.
            Self::StoreStatusToReg | Self::SetRegFromCarry => FlagsMask::from_bits(0),
        }
    }

    /// Returns `true` when the adjustment's result is a general-purpose
    /// register rather than a flags value (`lahf`, `salc`).
    #[must_use]
    pub const fn defines_register(self) -> bool {
        matches!(self, Self::StoreStatusToReg | Self::SetRegFromCarry)
    }

    /// What SSA value, if any, captures this adjustment's architectural write.
    ///
    /// Exhaustive on purpose: a new variant does not compile until its author
    /// decides whether the IR can represent what it writes. That decision is
    /// what [`Self::effects`] is derived from, so getting it wrong is a visible
    /// choice rather than an inherited default.
    #[must_use]
    pub const fn result(self) -> FlagAdjustResult {
        match self {
            // Status flags: the SSA result *is* the whole write.
            Self::InvertCarry
            | Self::ClearCarry
            | Self::SetCarry
            | Self::RotateMaskInsert
            | Self::SetNzFrom8
            | Self::SetNzFrom16
            | Self::ConvertToFpFlags
            | Self::ConvertFromFpFlags
            | Self::LoadStatusFromReg => FlagAdjustResult::Flags,
            // A general-purpose register (`lahf`, `salc`).
            Self::StoreStatusToReg | Self::SetRegFromCarry => FlagAdjustResult::Register,
            // Control flags. DF, AC and IF are architectural state no SSA
            // operand models, so nothing downstream can observe the write and
            // nothing can be forwarded from it.
            Self::ClearDirection
            | Self::SetDirection
            | Self::ClearAlignCheck
            | Self::SetAlignCheck
            | Self::ClearInterrupt
            | Self::SetInterrupt => FlagAdjustResult::None,
        }
    }

    /// How many SSA outputs a well-formed adjustment of this kind declares.
    #[must_use]
    pub const fn output_arity(self) -> usize {
        match self.result() {
            FlagAdjustResult::Flags | FlagAdjustResult::Register => 1,
            FlagAdjustResult::None => 0,
        }
    }

    /// The effect class implied by this kind alone.
    ///
    /// An adjustment whose whole architectural write is its SSA result is a
    /// pure function of its operands. One that writes a control flag the IR
    /// cannot name is [`SsaEffectKind::Opaque`] — the same treatment
    /// `FpuControl` gets, and for the same reason: `Pure` plus no definition
    /// means dead-code elimination removes it outright, and `cld`, `std`,
    /// `clac` and `stac` change what the program does.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self.result() {
            FlagAdjustResult::Flags | FlagAdjustResult::Register => {
                SsaEffects::new(SsaEffectKind::Pure, false)
            }
            FlagAdjustResult::None => SsaEffects::new(SsaEffectKind::Opaque, false),
        }
    }

    /// The effect class for an operation of this kind carrying `output_count`
    /// outputs.
    ///
    /// A kind that declares a result but carries none has written architectural
    /// state and handed back nothing to represent it, so the write is
    /// unobservable and must not be treated as pure — the malformed shape gets
    /// the conservative answer rather than the deletable one. The verifier
    /// reports the arity separately; this makes the effect safe meanwhile.
    #[must_use]
    pub const fn effects_for_outputs(self, output_count: usize) -> SsaEffects {
        if output_count == 0 && self.output_arity() != 0 {
            return SsaEffects::new(SsaEffectKind::Opaque, false);
        }
        self.effects()
    }

    /// The stable `flags.*` fingerprint key for this adjustment.
    ///
    /// `KindedVecData` carries no mnemonic, so the kind is the operation's only
    /// identity: without this every flag adjustment renders and fingerprints
    /// identically and `std` cannot be told from `clc`.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::InvertCarry => "flags.cfinv",
            Self::RotateMaskInsert => "flags.rmif",
            Self::SetNzFrom8 => "flags.setf8",
            Self::SetNzFrom16 => "flags.setf16",
            Self::ConvertToFpFlags => "flags.axflag",
            Self::ConvertFromFpFlags => "flags.xaflag",
            Self::ClearCarry => "flags.clear.carry",
            Self::SetCarry => "flags.set.carry",
            Self::ClearDirection => "flags.clear.direction",
            Self::SetDirection => "flags.set.direction",
            Self::ClearAlignCheck => "flags.clear.aligncheck",
            Self::SetAlignCheck => "flags.set.aligncheck",
            Self::ClearInterrupt => "flags.clear.interrupt",
            Self::SetInterrupt => "flags.set.interrupt",
            Self::LoadStatusFromReg => "flags.load.reg",
            Self::StoreStatusToReg => "flags.store.reg",
            Self::SetRegFromCarry => "flags.carry.toreg",
        }
    }

    /// The native mnemonic this adjustment lifts from.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::InvertCarry => "cfinv",
            Self::RotateMaskInsert => "rmif",
            Self::SetNzFrom8 => "setf8",
            Self::SetNzFrom16 => "setf16",
            Self::ConvertToFpFlags => "axflag",
            Self::ConvertFromFpFlags => "xaflag",
            Self::ClearCarry => "clc",
            Self::SetCarry => "stc",
            Self::ClearDirection => "cld",
            Self::SetDirection => "std",
            Self::ClearAlignCheck => "clac",
            Self::SetAlignCheck => "stac",
            Self::ClearInterrupt => "cli",
            Self::SetInterrupt => "sti",
            Self::LoadStatusFromReg => "sahf",
            Self::StoreStatusToReg => "lahf",
            Self::SetRegFromCarry => "salc",
        }
    }
}

/// What SSA value captures a [`FlagAdjustKind`]'s architectural write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlagAdjustResult {
    /// A flags value, which is the whole of what the operation writes.
    Flags,
    /// A general-purpose register (`lahf`, `salc`).
    Register,
    /// Nothing the IR can name: the write lands in a control flag (DF, AC, IF).
    ///
    /// Not deletable. There is no SSA value to make the write observable, so a
    /// pure classification plus an empty definition list is exactly what
    /// dead-code elimination removes.
    None,
}

/// Boxed payload for [`SsaOp::VectorComplexAdd`] — a complex-number add with a
/// 90° or 270° rotation of one operand's imaginary lane (AArch64 `fcadd` and the
/// integer `cadd`/`sqcadd`). The `inputs` are the two complex source vectors.
/// Pure.
///
/// [`SsaOp::VectorComplexAdd`]: crate::ir::ops::def::SsaOp::VectorComplexAdd
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorComplexAddData {
    /// `true` for a 270° rotation, `false` for 90°.
    pub rotate_270: bool,
    /// `true` for the saturating integer form (`sqcadd`).
    pub saturate: bool,
    /// Explicit SSA outputs (the complex result).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the two complex source vectors.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorPredicateBreak`] — generates a predicate from
/// a source predicate by breaking the active region at the first true element
/// (AArch64 SVE `brka`/`brkb`/`brkn`/`brkpa`/`brkpb` and their flag-setting
/// `*s` peers). `after` keeps the element that triggered the break (`brka*`);
/// `pair` selects the two-source `brkp*` forms; `propagate` selects `brkn`.
/// The `inputs` are the source predicate(s). Pure.
///
/// [`SsaOp::VectorPredicateBreak`]: crate::ir::ops::def::SsaOp::VectorPredicateBreak
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPredicateBreakData {
    /// `true` to break *after* the first true element (`brka`/`brkpa`).
    pub after: bool,
    /// `true` for the two-source paired forms (`brkpa`/`brkpb`).
    pub pair: bool,
    /// `true` for the propagate form (`brkn`).
    pub propagate: bool,
    /// Explicit SSA outputs (the generated predicate).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source predicate(s).
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorPredicateWhile`] — generates a predicate by
/// comparing an incrementing index from a scalar base against a scalar limit
/// (AArch64 SVE `whilelt`/`whilele`/`whilege`/`whilegt` signed and `whilelo`/
/// `whilels`/`whilehs`/`whilehi` unsigned). The `inputs` are `[first, last]`. Pure.
///
/// [`SsaOp::VectorPredicateWhile`]: crate::ir::ops::def::SsaOp::VectorPredicateWhile
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorPredicateWhileData {
    /// The comparison relating each index to the limit.
    pub kind: VectorCompareKind,
    /// `true` for the unsigned forms (`whilelo`/`whilels`/`whilehs`/`whilehi`).
    pub unsigned: bool,
    /// Explicit SSA outputs (the generated predicate).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the scalar base and limit.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorNarrowSaturate`] — a single-source saturating
/// narrowing of a vector (AArch64 `sqxtn`/`uqxtn`/`sqxtun` and the shifting
/// `sqshrn`/`uqshrn`/`sqrshrn`/`sqshrun`/`sqrshrun`/… families). Each source lane
/// is optionally right-shifted by `shift` (rounded when `rounding`), then clamped
/// to the narrow destination range — signed when `unsigned_dst` is false, to the
/// unsigned range when true — interpreting the source as signed when `signed_src`.
/// The `inputs` are `[src]`. Pure.
///
/// [`SsaOp::VectorNarrowSaturate`]: crate::ir::ops::def::SsaOp::VectorNarrowSaturate
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorNarrowSaturateData {
    /// `true` when the source lanes are interpreted as signed.
    pub signed_src: bool,
    /// `true` to saturate to the unsigned narrow range (`sqxtun`/`sqshrun`/…).
    pub unsigned_dst: bool,
    /// `true` for the rounding shift forms (`sqrshrn`/`uqrshrn`/`sqrshrun`/…).
    pub rounding: bool,
    /// Right-shift amount applied before narrowing (`0` for the extract forms
    /// `sqxtn`/`uqxtn`/`sqxtun`).
    pub shift: u8,
    /// Explicit SSA outputs defined by the operation (the narrowed vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the single source vector.
    pub inputs: Vec<SsaVarId>,
}

impl VectorMaddKind {
    /// Returns the stable textual identity used in similarity / display.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::MultiplyAddS16 => "vector.madd.s16",
            Self::MultiplyAddU8S8Sat => "vector.madd.u8s8.sat",
            Self::DotProductU8S8 => "vector.dp.u8s8",
            Self::DotProductU8S8Sat => "vector.dp.u8s8.sat",
            Self::DotProductS16 => "vector.dp.s16",
            Self::DotProductS16Sat => "vector.dp.s16.sat",
            Self::MultiplyAdd52Lo => "vector.madd52.lo",
            Self::MultiplyAdd52Hi => "vector.madd52.hi",
            Self::DotProductBf16 => "vector.dp.bf16",
            Self::MultiplyAccumulate => "vector.mac",
            Self::MultiplyAccumulateSat => "vector.mac.sat",
            Self::MultiplyAccumulatePairs => "vector.macd",
            Self::MultiplyAccumulatePairsSat => "vector.macd.sat",
            Self::DotProductS8S8 => "vector.dp.s8s8",
            Self::DotProductS8S8Sat => "vector.dp.s8s8.sat",
            Self::DotProductS8U8 => "vector.dp.s8u8",
            Self::DotProductS8U8Sat => "vector.dp.s8u8.sat",
            Self::DotProductU8U8 => "vector.dp.u8u8",
            Self::DotProductU8U8Sat => "vector.dp.u8u8.sat",
            Self::DotProductS16U16 => "vector.dp.s16u16",
            Self::DotProductS16U16Sat => "vector.dp.s16u16.sat",
            Self::DotProductU16S16 => "vector.dp.u16s16",
            Self::DotProductU16S16Sat => "vector.dp.u16s16.sat",
            Self::DotProductU16U16 => "vector.dp.u16u16",
            Self::DotProductU16U16Sat => "vector.dp.u16u16.sat",
        }
    }
}

/// Lane element class of a vector operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorElementKind {
    /// Integer lanes.
    Integer,
    /// Floating-point lanes.
    Float,
    /// Element class not recovered by the decoder.
    #[default]
    Unknown,
}

/// Lane element descriptor of a vector operation.
///
/// Carried on the SIMD ops so a host projecting the IR back out (e.g. the Visus
/// cleaned-IR lower leg) can render `paddd` vs `paddb` vs `addps` distinctly. The
/// descriptor travels with the op through normalization, so it is not lost the
/// way a side-table keyed by value id would be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorElement {
    /// Lane element class (integer / floating-point / unknown).
    pub kind: VectorElementKind,
    /// Width of one lane in bits (`0` when not recovered).
    pub bits: u32,
    /// `true` when the op operates on a single (scalar) lane rather than a
    /// packed vector — e.g. `addss`/`addsd` vs `addps`/`addpd`.
    pub scalar: bool,
}

/// Vector unary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorUnaryKind {
    /// Lane-wise integer or floating negation.
    Neg,
    /// Lane-wise bitwise not.
    Not,
    /// Lane-wise population count.
    Popcount,
    /// Lane-wise absolute value.
    Abs,
    /// Horizontal lane sum.
    HorizontalAdd,
    /// Lane-wise floating-point square root.
    Sqrt,
    /// Lane-wise floating-point round to integral value (result stays
    /// floating-point): x87 `frndint`, ARM `vrint*`, AArch64 `frint*`.
    Round,
    /// Lane-wise floating-point reciprocal approximation (`1/x`): x86
    /// `vrcp14*`/`vrcp28*`/`vrcpps`, AArch64 `frecpe`.
    Reciprocal,
    /// Lane-wise floating-point reciprocal-square-root approximation
    /// (`1/sqrt(x)`): x86 `vrsqrt14*`/`vrsqrt28*`/`vrsqrtps`, AArch64 `frsqrte`.
    ReciprocalSqrt,
    /// Lane-wise floating-point exponent extraction (`vgetexp*`): returns the
    /// unbiased exponent as a floating-point value.
    GetExponent,
    /// Lane-wise floating-point mantissa extraction (`vgetmant*`): the
    /// immediate normalization/sign control rides as a parameter.
    GetMantissa,
    /// Lane-wise floating-point round-to-scale (`vrndscale*`): rounds to a
    /// number of fraction bits selected by the control immediate.
    RoundScale,
    /// Lane-wise floating-point reduction (`vreduce*`): `x - round(x)` scaled by
    /// the control immediate.
    Reduce,
    /// Lane-wise identity (pass-through). Used as the operation of a predicated
    /// merge / blend (`vpblendm*`/`vblendm*`), where each active lane copies the
    /// source and inactive lanes take the passthrough.
    Identity,
    /// Lane-wise count of leading zero bits (`vplzcntd`/`vplzcntq`).
    LeadingZeros,
    /// Lane-wise floating-point fractional part `x - trunc(x)` (AMD XOP
    /// `vfrczps`/`vfrczpd`/`vfrczss`/`vfrczsd`).
    Fraction,
    /// Lane-wise conflict detection (`vpconflictd`/`vpconflictq`): each lane
    /// receives a bitmask of the preceding lanes equal to it.
    Conflict,
    /// Lane-wise base-2 exponential `2^x` (`vexp2*`).
    Exp2,
    /// Lane-wise base-2 logarithm `log2(x)` (`vlog2*`).
    Log2,
    /// Lane-wise count of leading sign bits (`vcls`).
    LeadingSignBits,
    /// Lane-wise count of leading one bits (MSA `nloc`).
    LeadingOnes,
    /// Round toward zero to an integral value (truncate; VB6 `Fix`).
    Truncate,
    /// Round down to an integral value (floor, toward −∞; VB6 `Int`).
    Floor,
    /// Sign of the value (`-1` / `0` / `+1`; VB6 `Sgn`).
    Sign,
    /// Lane-wise bit reversal within each element (AArch64 NEON/SVE `rbit`).
    BitReverse,
}

/// Vector binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorBinaryKind {
    /// Lane-wise addition.
    Add,
    /// Lane-wise subtraction.
    Sub,
    /// Lane-wise multiplication.
    Mul,
    /// Lane-wise division.
    Div,
    /// Lane-wise minimum.
    Min,
    /// Lane-wise maximum.
    Max,
    /// Lane-wise bitwise and.
    And,
    /// Lane-wise bitwise or.
    Or,
    /// Lane-wise bitwise xor.
    Xor,
    /// Lane-wise shift left.
    Shl,
    /// Lane-wise shift right.
    Shr,
    /// Lane-wise bitwise and-not (`a & ~b`).
    AndNot,
    /// Lane-wise multiply keeping the high half of the product.
    MulHigh,
    /// Lane-wise signed multiply keeping the rounded high 16 bits of the
    /// scaled product (`pmulhrsw`: `((a*b >> 14) + 1) >> 1`).
    MulHighRound,
    /// Lane-wise rounding average `(a + b + 1) >> 1`.
    Avg,
    /// Lane-wise absolute difference `|a - b|`.
    AbsDiff,
    /// Lane-wise saturating addition (clamps on overflow).
    SatAdd,
    /// Lane-wise saturating subtraction (clamps on underflow).
    SatSub,
    /// Lane-wise floating-point scale: `a * 2^floor(b)` (x86 `vscalef*`).
    Scale,
    /// Lane-wise floating-point range restriction selecting a min/max-class
    /// result under an immediate control (x86 `vrange*`).
    Range,
    /// Lane-wise bitwise rotate left (x86 `vprol*`, XOP `vprot*`).
    Rol,
    /// Lane-wise bitwise rotate right (x86 `vpror*`).
    Ror,
    /// Lane-wise variable logical shift by a signed count, the sign selecting
    /// direction (AMD XOP `vpshl*`: positive shifts left, negative right).
    VariableShiftLogical,
    /// Lane-wise variable arithmetic shift by a signed count (AMD XOP `vpsha*`).
    VariableShiftArithmetic,
    /// Lane-wise apply-sign: negate / zero / keep the first operand according to
    /// the sign of the second (x86 `psign*`).
    ApplySign,
    /// 3DNow! reciprocal Newton-Raphson refinement step 1 (`pfrcpit1`).
    ReciprocalIter1,
    /// 3DNow! reciprocal Newton-Raphson refinement step 2 (`pfrcpit2`).
    ReciprocalIter2,
    /// 3DNow! reciprocal-square-root Newton-Raphson refinement step 1
    /// (`pfrsqit1`).
    ReciprocalSqrtIter1,
    /// Cyrix EMMI maximum by magnitude (`pmagw`): per lane, keep the operand
    /// with the greater absolute value.
    MaxMagnitude,
    /// Lane-wise negated multiply: `-(a * b)` (AArch64 scalar-FP `fnmul`).
    MulNegate,
    /// Bitwise OR with complemented second operand: `a | ~b` (NEON `vorn`).
    OrNot,
    /// Lane-wise add of absolute values: `|a| + |b|` (MSA `add_a`).
    AddAbs,
    /// Lane-wise integer remainder: `a % b` (MSA `mod_s`/`mod_u`). Sign-agnostic
    /// like [`Self::Div`]; the per-lane signedness comes from the element type.
    Rem,
    /// Lane-wise bitwise NOR: `~(a | b)` (MSA `nor.v`).
    Nor,
}

/// Vector ternary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorTernaryKind {
    /// Lane-wise fused multiply-add.
    Fma,
    /// Lane-wise select controlled by a vector mask.
    Select,
    /// Lane-wise floating-point fix-up under an immediate selector table (x86
    /// `vfixupimm*`): patches special inputs (NaN/inf/zero/…) of the first
    /// operand using the classification of the second and a control immediate.
    FixupImm,
    /// Lane-wise funnel shift left: concatenate the first two operands and shift
    /// left by the third (count), taking the high half (x86 `vpshldv*`).
    FunnelLeft,
    /// Lane-wise funnel shift right: concatenate the first two operands and
    /// shift right by the third (count), taking the low half (x86 `vpshrdv*`).
    FunnelRight,
}

/// Vector comparison operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorCompareKind {
    /// Lane-wise equality comparison.
    Eq,
    /// Lane-wise inequality comparison.
    Ne,
    /// Lane-wise less-than comparison.
    Lt,
    /// Lane-wise less-than-or-equal comparison.
    Le,
    /// Lane-wise greater-than comparison.
    Gt,
    /// Lane-wise greater-than-or-equal comparison.
    Ge,
    /// Lane-wise unordered test (either operand is NaN) — x86 `cmpps` predicate
    /// `UNORD`.
    Unordered,
    /// Lane-wise ordered test (neither operand is NaN) — x86 `cmpps` predicate
    /// `ORD`.
    Ordered,
    /// Lane-wise not-less-than (true when `!(a < b)`, including unordered) —
    /// x86 `cmpps` predicate `NLT`.
    NotLt,
    /// Lane-wise not-less-or-equal (true when `!(a <= b)`, including unordered)
    /// — x86 `cmpps` predicate `NLE`.
    NotLe,
    /// Lane-wise not-greater-or-equal (true when `!(a >= b)`, including
    /// unordered) — x86 `cmpps` predicate `NGE`.
    NotGe,
    /// Lane-wise not-greater-than (true when `!(a > b)`, including unordered) —
    /// x86 `cmpps` predicate `NGT`.
    NotGt,
    /// Lane-wise constant-true predicate — x86 `cmpps` `TRUE`.
    AlwaysTrue,
    /// Lane-wise constant-false predicate — x86 `cmpps` `FALSE`.
    AlwaysFalse,
}

/// Vector cast operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorCastKind {
    /// Converts lanes with signed integer interpretation.
    Signed,
    /// Converts lanes with unsigned integer interpretation.
    Unsigned,
    /// Converts lanes with floating-point interpretation.
    Float,
}

/// Vector mask unary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorMaskUnaryKind {
    /// Lane-wise predicate negation.
    Not,
}

/// Vector mask binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorMaskBinaryKind {
    /// Lane-wise predicate conjunction.
    And,
    /// Lane-wise predicate disjunction.
    Or,
    /// Lane-wise predicate exclusive-or.
    Xor,
    /// Lane-wise `left & !right`.
    AndNot,
}

/// Scalar summary operation over vector lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorReduceKind {
    /// Adds all lanes.
    Add,
    /// Multiplies all lanes.
    Mul,
    /// Bitwise-ands all lanes.
    And,
    /// Bitwise-ors all lanes.
    Or,
    /// Bitwise-xors all lanes.
    Xor,
    /// Computes the minimum lane.
    Min,
    /// Computes the maximum lane.
    Max,
    /// Tests whether any predicate lane is true.
    Any,
    /// Tests whether all predicate lanes are true.
    All,
}

/// Operation used when extracting a scalar bitmask from a vector or mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorBitmaskKind {
    /// Extracts each lane's most significant bit, as in x86 `pmovmskb`.
    LaneMostSignificantBits,
    /// Extracts predicate lane bits from a vector mask.
    PredicateBits,
}

/// Controls predicated vector memory behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorMaskMode {
    /// Inactive lanes preserve a passthrough vector value.
    Merge,
    /// Inactive lanes are zeroed or ignored.
    Zero,
}

/// Direction of an AVX-512-style vector lane packing operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorPackKind {
    /// Packs active source lanes contiguously into the low destination lanes.
    Compress,
    /// Expands contiguous low source lanes into active destination lanes.
    Expand,
}

/// Fault behavior for vector loads that may complete partially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorFaultMode {
    /// Ordinary load where any lane fault traps the instruction.
    Normal,
    /// First-fault load where lanes after the first fault are suppressed.
    FirstFault,
    /// Fault-only-first load where only the first active lane can fault.
    FaultOnlyFirst,
}

/// Memory layout for segmented vector memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorSegmentLayout {
    /// Interleaved structure layout, such as RVV `vlseg` or AArch64 `ld2`.
    Interleaved,
    /// Consecutive structure layout, with each segment stored contiguously.
    Consecutive,
}

impl_kinded_payload! {
    VectorPermuteData { outputs, inputs };
    VectorDotProductData { imm8, element_bits, outputs, inputs };
    VectorIntDotProductData { signed_a, signed_b, source_bits, dest_bits, outputs, inputs };
    VectorShuffleBitsData { outputs, inputs };
    VectorIntersectData { outputs, inputs };
    VectorBitfieldData { insert, index, length, outputs, inputs };
    VectorConditionalMoveData { condition, outputs, inputs };
    VectorStringCompareData { imm8, explicit_length, result_index, outputs, inputs };
    VectorHorizontalMinPosData { outputs, inputs };
    VectorHorizontalReduceData {
        subtract, unsigned, source_bits, dest_bits, outputs, inputs
    };
    VectorPackNarrowData { unsigned, outputs, inputs };
    VectorSmeOuterProductData { subtract, signed_a, signed_b, float, outputs, inputs };
    VectorMatrixMulAccData { signed_a, signed_b, float, outputs, inputs };
    VectorReverseChunksData { chunk_bits, outputs, inputs };
    VectorCountAdjustData {
        decrement, saturate, signed, by_predicate, element_bits, outputs, inputs
    };
    VectorExtendInLaneData { signed, source_bits, element_bits, outputs, inputs };
    outputs_only VectorElementCountData { element_bits, multiplier, outputs };
    VectorSveAddressGenData { signed_extend, shift, outputs, inputs };
    VectorSveComputeData { op, element_bits, rotation, outputs, inputs };
    VectorPredicateOpData { op, element_bits, outputs, inputs };
    VectorSmeMiscData { op, element_bits, outputs, inputs };
    VectorStructLoadReplicateData { count, element_bits, outputs, inputs };
    VectorComplexAddData { rotate_270, saturate, outputs, inputs };
    VectorPredicateBreakData { after, pair, propagate, outputs, inputs };
    VectorPredicateWhileData { kind, unsigned, outputs, inputs };
    VectorNarrowSaturateData { signed_src, unsigned_dst, rounding, shift, outputs, inputs };
}
