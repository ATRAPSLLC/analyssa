//! Vector cryptographic kinds ([`VectorCryptoKind`]): AES, SHA, SM3, SM4 and
//! GHASH steps, and what each computes.

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::{
    aes, ghash,
    lanes::VectorConst,
    ops::effects::{SsaEffectKind, SsaEffects},
    sha,
    sha::Word,
    sm3, sm4,
};

/// The cryptographic primitive a [`Kernel::Crypto`] computes: the AES, SHA,
/// SM3, SM4, GF(2^8) and GHASH round and message steps, each named by the
/// step it is. A lift never decomposes a round into primitive bit operations
/// nor carries it as an opaque blob. Every variant is a pure function of its
/// inputs.
///
/// [`Kernel::Crypto`]: crate::ir::ops::Kernel::Crypto
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
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
    /// AES key-locker encryption of one block using a wrapped key handle
    /// (`aesenc128kl`/`aesenc256kl`): the block and the handle in, the block
    /// and the flags (`ZF` set where the handle is illegal) out.
    AesEncryptKeyLocker,
    /// AES key-locker decryption of one block using a wrapped key handle
    /// (`aesdec128kl`/`aesdec256kl`), as [`Self::AesEncryptKeyLocker`].
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
    /// Expands a 128-bit AES key by one round.
    ///
    /// Inputs are the four current key words and a round number from one through ten. The low word is XORed with SubWord(RotWord(high word)) and the round constant; subsequent words XOR the previous result with the corresponding old word.
    AesExpand128,

    /// Expands a 256-bit AES key by one half-key round.
    ///
    /// Inputs are the previous four words, current four words and a round number from two through fourteen. The current high word is substituted, also rotated and XORed with Rcon(round/2-1) for even rounds, then XOR-chained with the previous four words. Words are low first.
    AesExpand256,

    /// Multiplies two GHASH field elements.
    ///
    /// Reverses the bits within each input byte, multiplies modulo x^128+x^7+x^2+x+1, and reverses the bits within every result byte. Both inputs and the result have 128 bits.
    GhashMultiply,

    /// XORs a GHASH accumulator with a message and multiplies by the hash key.
    ///
    /// Inputs are the accumulator, message block and key, in that order. Applies the same field representation and polynomial as GhashMultiply to (accumulator XOR message) and key.
    GhashMultiplyAdd,

    /// Expands four SHA-2 message schedule words.
    ///
    /// The group contains four 32-bit or four 64-bit words. Inputs, low word first, are W0..3, {W4,W9,W10,W11}, and {W12,unused,W14,W15}. Computes W16..19 using the standard SHA-256 or SHA-512 small sigma recurrence and returns them low word first.
    Sha2Schedule4,

    /// Performs two SHA-2 rounds with the low pair of message-plus-constant words.
    ///
    /// Inputs are {c,d,g,h}, {a,b,e,f}, and {WK3,WK2,WK1,WK0}, each written high word first. Uses WK0 and WK1 and returns {a,b,e,f} after two standard SHA-256 or SHA-512 rounds. The group has four 32-bit or four 64-bit words.
    Sha2Compress2Low,

    /// Performs two SHA-2 rounds with the high pair of message-plus-constant words.
    ///
    /// Uses the same packed state and result as Sha2Compress2Low, selecting WK2 and WK3 instead. The group has four 32-bit or four 64-bit words.
    Sha2Compress2High,

    /// Expands eight SM3 message schedule words.
    ///
    /// Inputs are W0..7 and W8..15, low word first. Byte-swaps every 32-bit word, applies the standard SM3 W16..23 recurrence, then byte-swaps each result word and returns W16..23 low word first. Each group has 256 bits.
    Sm3Expand8,

    /// Performs two SM3 compression rounds on a complete state.
    ///
    /// Inputs are {H,G,F,E,D,C,B,A}, {unused,unused,W5,W4,unused,unused,W1,W0}, written high word first, and round-pair index r from zero through 31. Byte-swaps every word, performs rounds 2r and 2r+1 with W0/W1 and W0 XOR W4/W1 XOR W5, and returns the byte-swapped complete state in the same order.
    Sm3Compress2,

    /// Expands four SM4 round-key words.
    ///
    /// Inputs are the previous four key words, low word first, and a round-group index r from zero through seven. Applies the SM4 key recurrence with CK[4r..4r+3], the standard substitution and rotations by 13 and 23, returning the next four words low word first.
    Sm4Expand4,
    /// Performs inverse ShiftRows, inverse SubBytes, AddRoundKey, then inverse MixColumns.
    ///
    /// Inputs are the 128-bit state and untransformed round key. This inverse
    /// cipher round mixes the key together with the substituted state.
    AesDecryptKeyMix,
    /// AES single encryption round with mix-columns, the round key added
    /// first: AddRoundKey, then ShiftRows, SubBytes and MixColumns (AArch64
    /// `aese` then `aesmc`, and SVE_AES2's `aesemc`).
    AesEncryptRoundMix,
    /// AES single decryption round with inverse mix-columns, the round key
    /// added first: AddRoundKey, then inverse ShiftRows, inverse SubBytes and
    /// inverse MixColumns (AArch64 `aesd` then `aesimc`, and SVE_AES2's
    /// `aesdimc`).
    AesDecryptRoundInvMix,
    /// SHA-512 hash update part 1 (AArch64 `sha512h`): the first halves of
    /// two rounds, each round's `T1`.
    Sha512Hash,
    /// SHA-512 hash update part 2 (AArch64 `sha512h2`): the new A and B of
    /// two rounds from their `T1`.
    Sha512Hash2,
    /// SHA-512 message schedule update step 0 (AArch64 `sha512su0`).
    Sha512ScheduleUpdate0,
    /// SHA-512 message schedule update step 1 (AArch64 `sha512su1`).
    Sha512ScheduleUpdate1,
    /// AES key-locker encryption of eight blocks using a wrapped key handle
    /// (`aesencwide128kl`/`aesencwide256kl`): the handle and the blocks of
    /// `xmm0` to `xmm7` in, the eight blocks and the flags out.
    AesEncryptKeyLockerWide,
    /// AES key-locker decryption of eight blocks using a wrapped key handle
    /// (`aesdecwide128kl`/`aesdecwide256kl`), as
    /// [`Self::AesEncryptKeyLockerWide`].
    AesDecryptKeyLockerWide,
}

impl VectorCryptoKind {
    /// Evaluates a supported AES round on packed column-major state bytes.
    ///
    /// Byte zero is the low byte. Returns `None` for non-round primitives.
    #[must_use]
    pub fn aes_round(self, state: u128, key: u128) -> Option<u128> {
        let (inverse, last, key_before_mix) = match self {
            Self::AesEncrypt => (false, false, false),
            Self::AesEncryptLast => (false, true, false),
            Self::AesDecrypt => (true, false, false),
            Self::AesDecryptLast => (true, true, false),
            Self::AesDecryptKeyMix => (true, false, true),
            // The round key added first is the full round of the keyed state
            // with no key after it.
            Self::AesEncryptRoundMix => {
                return aes::round(false, false, false, state ^ key, 0);
            }
            Self::AesDecryptRoundInvMix => {
                return aes::round(true, false, false, state ^ key, 0);
            }
            _ => return None,
        };
        aes::round(inverse, last, key_before_mix, state, key)
    }

    /// The results this primitive computes of vector constants of elements
    /// `element_bits` wide, one for each of its outputs: each step in every
    /// group of its register, a 128-bit block for the AES, GHASH and SM4
    /// steps (`vaesenc ymm` encrypts two blocks), SM3's 256 bits of RISC-V,
    /// and four elements for RISC-V's SHA-2 steps, 32 or 64 bits each. A
    /// source of one group (RISC-V's `.vs` forms) serves every group.
    ///
    /// - The AES rounds of `[state, key]` ([`Self::aes_round`]), AArch64's
    ///   `aese` and `aesd` the round of the keyed state without its column
    ///   mix, and the column mix (`aesmc`) and its inverse (`aesimc`) of one
    ///   block.
    /// - `aeskeygenassist` of `[block, rcon]`: the substituted words 1 and 3,
    ///   each also rotated and keyed by `rcon`.
    /// - `gf2p8mulb` of two byte vectors: each byte pair's product in the AES
    ///   field. `gf2p8affineqb` and `gf2p8affineinvqb` of `[x, matrix, b]`:
    ///   each byte of `x` (its inverse for the latter) multiplied by its
    ///   quadword's matrix, bit `i` of the result the parity of the matrix's
    ///   byte `7 - i` and the byte, plus bit `i` of `b`.
    /// - The SM4 steps of `[state, keys]` (`vsm4rnds4`, `sm4e`) and of
    ///   `[key, constants]` (`vsm4key4`, `sm4ekey`): four rounds of each
    ///   block, the key schedule's with its own linear transform.
    /// - The SHA-1, SHA-2 and SM3 steps, of the registers each reads.
    /// - RISC-V's key expansions: `vaeskf1` of `[key, round]`, `vaeskf2` of
    ///   `[previous, current, round]`, `vsm4k` of `[key, round group]`; its
    ///   GHASH steps, `vgmul` of `[multiplier, subkey]` and `vghsh` of
    ///   `[hash, block, subkey]`; its SHA-2 steps, `vsha2ms` of the three
    ///   schedule groups and `vsha2cl` and `vsha2ch` of `[cdgh, abef, wk]`;
    ///   and its SM3 steps, `vsm3me` of `[W0..7, W8..15]` and `vsm3c` of
    ///   `[state, words, round]`.
    ///
    /// # Returns
    ///
    /// The results, or `None` for inputs of another shape and for a step this
    /// does not compute: the Key Locker steps, whose key is wrapped by the
    /// processor's own key, compute nothing constant inputs determine.
    #[must_use]
    pub fn evaluate(self, inputs: &[VectorConst], element_bits: u32) -> Option<Vec<VectorConst>> {
        // The step of each 128-bit block.
        let lanewise = |sources: &[&VectorConst],
                        step: &dyn Fn(&[u128]) -> Option<u128>|
         -> Option<Vec<VectorConst>> {
            let result = Self::grouped::<u128, 1>(128, sources, |blocks| {
                let blocks: Vec<u128> = blocks.iter().map(|[block]| *block).collect();
                Some([step(&blocks)?])
            })?;
            Some(vec![result])
        };
        // An immediate operand, read as the integer it is.
        let immediate = |value: &VectorConst| u8::try_from(value.lane(0) & 0xff).ok();
        let bytes = |block: u128| block.to_le_bytes();
        match (self, inputs) {
            (
                Self::AesEncrypt
                | Self::AesEncryptLast
                | Self::AesDecrypt
                | Self::AesDecryptLast
                | Self::AesDecryptKeyMix
                | Self::AesEncryptRoundMix
                | Self::AesDecryptRoundInvMix,
                [state, key],
            ) => lanewise(&[state, key], &|block| match block {
                [state, key] => self.aes_round(*state, *key),
                _ => None,
            }),
            (Self::AesEncryptRound | Self::AesDecryptRound, [state, key]) => {
                let inverse = self == Self::AesDecryptRound;
                lanewise(&[state, key], &|block| match block {
                    [state, key] => aes::round(inverse, true, false, state ^ key, 0),
                    _ => None,
                })
            }
            (Self::AesMixColumns | Self::AesInvMixColumns, [block]) => {
                let inverse = self == Self::AesInvMixColumns;
                lanewise(&[block], &|block| {
                    aes::mix_columns(*block.first()?, inverse)
                })
            }
            (Self::AesKeygenAssist, [block, rcon]) => {
                let rcon = u32::from(immediate(rcon)?);
                lanewise(&[block], &|block| {
                    let word = |index: usize| -> Option<u32> {
                        let bytes = bytes(*block.first()?);
                        let word = bytes
                            .get(index.checked_mul(4)?..index.checked_mul(4)?.checked_add(4)?)?;
                        Some(u32::from_le_bytes(word.try_into().ok()?))
                    };
                    let (first, third) = (
                        aes::substitute_word(word(1)?),
                        aes::substitute_word(word(3)?),
                    );
                    let keyed = |word: u32| word.rotate_right(8) ^ rcon;
                    Some(
                        u128::from(first)
                            | u128::from(keyed(first)) << 32
                            | u128::from(third) << 64
                            | u128::from(keyed(third)) << 96,
                    )
                })
            }
            (Self::Gf2p8Mul, [left, right]) => lanewise(&[left, right], &|block| match block {
                [left, right] => {
                    let (left, right) = (bytes(*left), bytes(*right));
                    let mut product = [0u8; 16];
                    for (byte, (left, right)) in product.iter_mut().zip(left.into_iter().zip(right))
                    {
                        *byte = aes::multiply(left, right);
                    }
                    Some(u128::from_le_bytes(product))
                }
                _ => None,
            }),
            (Self::Gf2p8Affine | Self::Gf2p8AffineInv, [values, matrix, b]) => {
                let (inverse, b) = (self == Self::Gf2p8AffineInv, immediate(b)?);
                lanewise(&[values, matrix], &|block| match block {
                    [values, matrix] => {
                        let (values, matrix) = (bytes(*values), bytes(*matrix));
                        let mut result = [0u8; 16];
                        for (index, byte) in result.iter_mut().enumerate() {
                            let value = *values.get(index)?;
                            let value = if inverse {
                                aes::reciprocal(value)
                            } else {
                                value
                            };
                            // The quadword holding this byte holds its matrix.
                            let row = |bit: usize| -> Option<u8> {
                                matrix
                                    .get(
                                        (index / 8)
                                            .checked_mul(8)?
                                            .checked_add(7usize.checked_sub(bit)?)?,
                                    )
                                    .copied()
                            };
                            let mut affine = 0u8;
                            for bit in 0..8 {
                                let parity = (row(bit)? & value).count_ones() & 1;
                                affine |= u8::try_from(parity).ok()? << bit;
                            }
                            *byte = affine ^ b;
                        }
                        Some(u128::from_le_bytes(result))
                    }
                    _ => None,
                })
            }
            (Self::Sm4Rounds | Self::Sm4Key, [state, keys]) => {
                let transform = if self == Self::Sm4Rounds {
                    sm4::Transform::Cipher
                } else {
                    sm4::Transform::Key
                };
                let result = Self::grouped::<u32, 4>(32, &[state, keys], |groups| match groups {
                    [state, keys] => transform.rounds4(*state, *keys),
                    _ => None,
                })?;
                Some(vec![result])
            }
            (Self::Sm4Expand4, [key, group]) => {
                let group = immediate(group)?;
                let result = Self::grouped::<u32, 4>(32, &[key], |groups| match groups {
                    [key] => sm4::Transform::expand4(*key, group),
                    _ => None,
                })?;
                Some(vec![result])
            }
            (Self::AesExpand128, [key, round]) => {
                let round = immediate(round)?;
                let result = Self::grouped::<u32, 4>(32, &[key], |groups| match groups {
                    [key] => Some(aes::expand128(*key, round)),
                    _ => None,
                })?;
                Some(vec![result])
            }
            (Self::AesExpand256, [previous, current, round]) => {
                let round = immediate(round)?;
                let result =
                    Self::grouped::<u32, 4>(32, &[previous, current], |groups| match groups {
                        [previous, current] => Some(aes::expand256(*previous, *current, round)),
                        _ => None,
                    })?;
                Some(vec![result])
            }
            (Self::GhashMultiply, [multiplier, subkey]) => {
                lanewise(&[multiplier, subkey], &|blocks| match blocks {
                    [multiplier, subkey] => Some(ghash::multiply(*multiplier, *subkey)),
                    _ => None,
                })
            }
            (Self::GhashMultiplyAdd, [hash, block, subkey]) => {
                lanewise(&[hash, block, subkey], &|blocks| match blocks {
                    [hash, block, subkey] => Some(ghash::multiply_add(*hash, *block, *subkey)),
                    _ => None,
                })
            }
            (Self::Sha2Schedule4 | Self::Sha2Compress2Low | Self::Sha2Compress2High, [_, _, _]) => {
                let result = match element_bits {
                    32 => self.sha2_group::<u32>(inputs)?,
                    64 => self.sha2_group::<u64>(inputs)?,
                    _ => return None,
                };
                Some(vec![result])
            }
            (Self::Sm3Expand8, [first, second]) => {
                let result =
                    Self::grouped::<u32, 8>(32, &[first, second], |groups| match groups {
                        [first, second] => Some(sm3::riscv_sm3_expand8(*first, *second)),
                        _ => None,
                    })?;
                Some(vec![result])
            }
            (Self::Sm3Compress2, [state, words, round]) => {
                let round = immediate(round)?;
                let result = Self::grouped::<u32, 8>(32, &[state, words], |groups| match groups {
                    [state, words] => Some(sm3::riscv_sm3_compress2(*state, *words, round)),
                    _ => None,
                })?;
                Some(vec![result])
            }
            _ => self.evaluate_hash(inputs),
        }
    }

    /// The SHA-1, SHA-2 and SM3 steps of [`Self::evaluate`]
    /// ([`crate::ir::sha`], [`crate::ir::sm3`]), SHA-1's, SHA-256's and SM3's
    /// of four 32-bit words, SHA-512's of 64-bit words: the x86 steps of the
    /// registers they read, the one they accumulate into first (`sha1rnds4`
    /// with its round group after them, `sha256rnds2` its message words,
    /// `xmm0`; `vsha512rnds2` of `[ymm1, ymm2, xmm3]`, `vsm3rnds2` of
    /// `[xmm1, xmm2, xmm3]` and its round), and the Arm ones of their
    /// registers likewise (`sha1c` of `[abcd, e, w]`, `sha1h` of one word,
    /// `sha512h` of `[qd, qn, vm]`, `sm3tt1a` of `[vd, vn, vm]` and the
    /// element of `vm` it names).
    fn evaluate_hash(self, inputs: &[VectorConst]) -> Option<Vec<VectorConst>> {
        let words = u32::words::<4>;
        let pairs = u64::words::<2>;
        let quads = u64::words::<4>;
        let word = |value: &VectorConst| u32::try_from(value.lane(0) & 0xffff_ffff).ok();
        let packed = u32::packed::<4>;
        let result = match (self, inputs) {
            (Self::Sha1Rounds4, [abcd, ew, function]) => packed(sha::sha1_rounds4(
                words(abcd)?,
                words(ew)?,
                u8::try_from(function.lane(0) & 3).ok()?,
            ))?,
            (Self::Sha1NextE, [state, message]) => {
                packed(sha::sha1_next_e(words(state)?, words(message)?))?
            }
            (Self::Sha1Msg1, [first, second]) => {
                packed(sha::sha1_message1(words(first)?, words(second)?))?
            }
            (Self::Sha1Msg2, [partial, last]) => {
                packed(sha::sha1_message2(words(partial)?, words(last)?))?
            }
            (Self::Sha256Rounds2, [cdgh, abef, wk]) => {
                let [wk0, wk1, _, _] = words(wk)?;
                packed(sha::sha2_rounds2(words(cdgh)?, words(abef)?, [wk0, wk1]))?
            }
            (Self::Sha256Msg1, [first, second]) => {
                packed(sha::sha2_message1(words(first)?, words(second)?[0]))?
            }
            (Self::Sha256Msg2, [partial, last]) => {
                packed(sha::sha2_message2(words(partial)?, words(last)?))?
            }
            (Self::Sha512Rounds2, [cdgh, abef, wk]) => {
                u64::packed(sha::sha2_rounds2(quads(cdgh)?, quads(abef)?, pairs(wk)?))?
            }
            (Self::Sha512Msg1, [first, second]) => {
                u64::packed(sha::sha2_message1(quads(first)?, pairs(second)?[0]))?
            }
            (Self::Sha512Msg2, [partial, last]) => {
                u64::packed(sha::sha2_message2(quads(partial)?, quads(last)?))?
            }
            (Self::Sha512Hash, [qd, qn, vm]) => {
                u64::packed(sha::arm_sha512_hash(pairs(qd)?, pairs(qn)?, pairs(vm)?))?
            }
            (Self::Sha512Hash2, [qd, qn, vm]) => {
                u64::packed(sha::arm_sha512_hash2(pairs(qd)?, pairs(qn)?, pairs(vm)?))?
            }
            (Self::Sha512ScheduleUpdate0, [vd, vn]) => {
                u64::packed(sha::arm_sha512_schedule0(pairs(vd)?, pairs(vn)?))?
            }
            (Self::Sha512ScheduleUpdate1, [vd, vn, vm]) => u64::packed(sha::arm_sha512_schedule1(
                pairs(vd)?,
                pairs(vn)?,
                pairs(vm)?,
            ))?,
            (
                Self::Sha1HashChoose | Self::Sha1HashParity | Self::Sha1HashMajority,
                [abcd, e, w],
            ) => {
                let function = match self {
                    Self::Sha1HashChoose => 0,
                    Self::Sha1HashParity => 1,
                    _ => 2,
                };
                packed(sha::arm_sha1_hash(
                    words(abcd)?,
                    word(e)?,
                    words(w)?,
                    function,
                ))?
            }
            (Self::Sha1FixedRotate, [value]) => {
                VectorConst::from_lanes(32, [u128::from(sha::arm_sha1_fixed_rotate(word(value)?))])?
            }
            (Self::Sha1ScheduleUpdate0, [vd, vn, vm]) => {
                packed(sha::arm_sha1_schedule0(words(vd)?, words(vn)?, words(vm)?))?
            }
            (Self::Sha1ScheduleUpdate1, [vd, vn]) => {
                packed(sha::arm_sha1_schedule1(words(vd)?, words(vn)?))?
            }
            (Self::Sha256Hash | Self::Sha256Hash2, [first, second, w]) => {
                packed(sha::arm_sha256_hash(
                    words(first)?,
                    words(second)?,
                    words(w)?,
                    self == Self::Sha256Hash,
                ))?
            }
            (Self::Sha256ScheduleUpdate0, [vd, vn]) => {
                packed(sha::arm_sha256_schedule0(words(vd)?, words(vn)?))?
            }
            (Self::Sha256ScheduleUpdate1, [vd, vn, vm]) => packed(sha::arm_sha256_schedule1(
                words(vd)?,
                words(vn)?,
                words(vm)?,
            ))?,
            (Self::Sm3Msg1, [dst, src1, src2]) => {
                packed(sm3::sm3_message1(words(dst)?, words(src1)?, words(src2)?))?
            }
            (Self::Sm3Msg2, [dst, src1, src2]) => {
                packed(sm3::sm3_message2(words(dst)?, words(src1)?, words(src2)?))?
            }
            (Self::Sm3Rounds2, [dst, src1, src2, round]) => packed(sm3::sm3_rounds2(
                words(dst)?,
                words(src1)?,
                words(src2)?,
                u8::try_from(round.lane(0) & 0xff).ok()?,
            ))?,
            (Self::Sm3PartW1, [vd, vn, vm]) => {
                packed(sm3::arm_sm3_partw1(words(vd)?, words(vn)?, words(vm)?))?
            }
            (Self::Sm3PartW2, [vd, vn, vm]) => {
                packed(sm3::arm_sm3_partw2(words(vd)?, words(vn)?, words(vm)?))?
            }
            (Self::Sm3SS1, [vn, vm, va]) => {
                packed(sm3::arm_sm3_ss1(words(vn)?, words(vm)?, words(va)?))?
            }
            (Self::Sm3TT1A | Self::Sm3TT1B, [vd, vn, vm, lane]) => packed(sm3::arm_sm3_tt1(
                words(vd)?,
                words(vn)?,
                *words(vm)?.get(usize::try_from(lane.lane(0)).ok()?)?,
                self == Self::Sm3TT1B,
            ))?,
            (Self::Sm3TT2A | Self::Sm3TT2B, [vd, vn, vm, lane]) => packed(sm3::arm_sm3_tt2(
                words(vd)?,
                words(vn)?,
                *words(vm)?.get(usize::try_from(lane.lane(0)).ok()?)?,
                self == Self::Sm3TT2B,
            ))?,
            _ => return None,
        };
        Some(vec![result])
    }

    /// `step` of each group of `N` words `bits` wide of the sources, the
    /// result laid out as the first source is. A source of one group serves
    /// every group; a source of another count, or not a whole number of
    /// groups, computes nothing.
    fn grouped<W, const N: usize>(
        bits: u32,
        sources: &[&VectorConst],
        step: impl Fn(&[[W; N]]) -> Option<[W; N]>,
    ) -> Option<VectorConst>
    where
        W: Copy + Into<u128> + TryFrom<u128>,
    {
        let lane_bits = sources.first()?.lane_bits();
        let groups = sources
            .iter()
            .map(|source| {
                let words = source
                    .relaned(bits)?
                    .lane_values()
                    .map(|lane| W::try_from(lane).ok())
                    .collect::<Option<Vec<W>>>()?;
                let (groups, rest) = words.as_chunks::<N>();
                rest.is_empty().then(|| groups.to_vec())
            })
            .collect::<Option<Vec<Vec<[W; N]>>>>()?;
        let count = groups.first()?.len();
        if groups
            .iter()
            .any(|source| source.len() != count && source.len() != 1)
        {
            return None;
        }
        let mut words = Vec::new();
        for index in 0..count {
            let group = groups
                .iter()
                .map(|source| source.get(index).or_else(|| source.first()).copied())
                .collect::<Option<Vec<[W; N]>>>()?;
            words.extend(step(&group)?);
        }
        VectorConst::from_lanes(bits, words.into_iter().map(Into::into))?.relaned(lane_bits)
    }

    /// RISC-V's SHA-2 steps of [`Self::evaluate`] for words of `W`: each
    /// element group of four words.
    fn sha2_group<W: Word>(self, inputs: &[VectorConst]) -> Option<VectorConst> {
        let [first, second, third] = inputs else {
            return None;
        };
        Self::grouped::<W, 4>(W::BITS, &[first, second, third], |groups| {
            let [first, second, third] = groups else {
                return None;
            };
            Some(match self {
                Self::Sha2Schedule4 => sha::sha2_schedule4(*first, *second, *third),
                Self::Sha2Compress2Low => sha::sha2_rounds2(*first, *second, [third[0], third[1]]),
                Self::Sha2Compress2High => sha::sha2_rounds2(*first, *second, [third[2], third[3]]),
                _ => return None,
            })
        })
    }

    /// Returns the data-input count of an element-group cryptographic primitive.
    ///
    /// Other crypto operations use their own instruction interfaces and return `None`.
    #[must_use]
    pub const fn group_input_count(self) -> Option<usize> {
        match self {
            Self::AesEncrypt
            | Self::AesEncryptLast
            | Self::AesDecrypt
            | Self::AesDecryptLast
            | Self::AesDecryptKeyMix
            | Self::AesEncryptRoundMix
            | Self::AesDecryptRoundInvMix
            | Self::Sm4Rounds
            | Self::AesExpand128
            | Self::GhashMultiply
            | Self::Sm3Expand8
            | Self::Sm4Expand4 => Some(2),
            Self::AesExpand256
            | Self::GhashMultiplyAdd
            | Self::Sha2Schedule4
            | Self::Sha2Compress2Low
            | Self::Sha2Compress2High
            | Self::Sm3Compress2 => Some(3),
            _ => None,
        }
    }

    /// Returns the inputs of a step over whole registers, outside a lane
    /// group: the sources each instruction reads in order, an accumulating
    /// destination first, an immediate or a lane index as a constant in its
    /// place, and a register read without being named last (`sha256rnds2`'s
    /// `xmm0`). A kind computed over lane groups reads as many whole values
    /// as it reads groups ([`Self::group_input_count`]).
    #[must_use]
    pub const fn whole_input_count(self) -> usize {
        match self {
            Self::AesInvMixColumns | Self::AesMixColumns | Self::Sha1FixedRotate => 1,
            Self::AesEncryptRound
            | Self::AesDecryptRound
            | Self::AesKeygenAssist
            | Self::Gf2p8Mul
            | Self::Sha1Msg1
            | Self::Sha1Msg2
            | Self::Sha1NextE
            | Self::Sha1ScheduleUpdate1
            | Self::Sha256Msg1
            | Self::Sha256Msg2
            | Self::Sha256ScheduleUpdate0
            | Self::Sha512Msg1
            | Self::Sha512Msg2
            | Self::Sha512ScheduleUpdate0
            | Self::Sm4Key
            | Self::AesEncryptKeyLocker
            | Self::AesDecryptKeyLocker
            | Self::AesEncrypt
            | Self::AesEncryptLast
            | Self::AesDecrypt
            | Self::AesDecryptLast
            | Self::AesDecryptKeyMix
            | Self::AesEncryptRoundMix
            | Self::AesDecryptRoundInvMix
            | Self::Sm4Rounds
            | Self::AesExpand128
            | Self::GhashMultiply
            | Self::Sm3Expand8
            | Self::Sm4Expand4 => 2,
            Self::Gf2p8Affine
            | Self::Gf2p8AffineInv
            | Self::Sha1HashChoose
            | Self::Sha1HashMajority
            | Self::Sha1HashParity
            | Self::Sha1Rounds4
            | Self::Sha1ScheduleUpdate0
            | Self::Sha256Hash
            | Self::Sha256Hash2
            | Self::Sha256Rounds2
            | Self::Sha256ScheduleUpdate1
            | Self::Sha512Hash
            | Self::Sha512Hash2
            | Self::Sha512Rounds2
            | Self::Sha512ScheduleUpdate1
            | Self::Sm3Msg1
            | Self::Sm3Msg2
            | Self::Sm3PartW1
            | Self::Sm3PartW2
            | Self::Sm3SS1
            | Self::AesExpand256
            | Self::GhashMultiplyAdd
            | Self::Sha2Schedule4
            | Self::Sha2Compress2Low
            | Self::Sha2Compress2High
            | Self::Sm3Compress2 => 3,
            Self::Sm3Rounds2 | Self::Sm3TT1A | Self::Sm3TT1B | Self::Sm3TT2A | Self::Sm3TT2B => 4,
            Self::AesEncryptKeyLockerWide | Self::AesDecryptKeyLockerWide => 9,
        }
    }

    /// Returns the blocks a step writes: eight for a wide key-locker step,
    /// one for any other.
    #[must_use]
    pub const fn data_outputs(self) -> usize {
        match self {
            Self::AesEncryptKeyLockerWide | Self::AesDecryptKeyLockerWide => 8,
            _ => 1,
        }
    }

    /// Returns whether the step also writes the flags: a key-locker step
    /// sets `ZF` where its handle is illegal.
    #[must_use]
    pub const fn writes_flags(self) -> bool {
        matches!(
            self,
            Self::AesEncryptKeyLocker
                | Self::AesDecryptKeyLocker
                | Self::AesEncryptKeyLockerWide
                | Self::AesDecryptKeyLockerWide
        )
    }

    /// Returns whether `bits` describes a supported complete cryptographic group.
    #[must_use]
    pub const fn accepts_group_bits(self, bits: u32) -> bool {
        match self {
            Self::Sha2Schedule4 | Self::Sha2Compress2Low | Self::Sha2Compress2High => {
                matches!(bits, 128 | 256)
            }
            Self::Sm3Expand8 | Self::Sm3Compress2 => bits == 256,
            _ => self.group_input_count().is_some() && bits == 128,
        }
    }

    /// Returns the IR name of this vector crypto operation, such as `aes.enc`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AesEncrypt => "aes.enc",
            Self::AesEncryptLast => "aes.enc.last",
            Self::AesDecrypt => "aes.dec",
            Self::AesDecryptLast => "aes.dec.last",
            Self::AesInvMixColumns => "aes.mix.inv",
            Self::AesKeygenAssist => "aes.keygen",
            Self::AesEncryptRound => "aes.enc.round",
            Self::AesDecryptRound => "aes.dec.round",
            Self::AesMixColumns => "aes.mix",
            Self::Sha1Rounds4 => "sha1.rounds4",
            Self::Sha1NextE => "sha1.next_e",
            Self::Sha1Msg1 => "sha1.msg1",
            Self::Sha1Msg2 => "sha1.msg2",
            Self::Sha256Rounds2 => "sha256.rounds2",
            Self::Sha256Msg1 => "sha256.msg1",
            Self::Sha256Msg2 => "sha256.msg2",
            Self::Sha512Rounds2 => "sha512.rounds2",
            Self::Sha512Msg1 => "sha512.msg1",
            Self::Sha512Msg2 => "sha512.msg2",
            Self::Sm3Msg1 => "sm3.msg1",
            Self::Sm3Msg2 => "sm3.msg2",
            Self::Sm3Rounds2 => "sm3.rounds2",
            Self::Sm4Key => "sm4.key",
            Self::Sm4Rounds => "sm4.rounds",
            Self::Gf2p8Affine => "gf2p8.affine",
            Self::Gf2p8AffineInv => "gf2p8.affine_inv",
            Self::Gf2p8Mul => "gf2p8.mul",
            Self::AesEncryptKeyLocker => "aes.enc.locked",
            Self::AesDecryptKeyLocker => "aes.dec.locked",
            Self::AesEncryptKeyLockerWide => "aes.enc.locked.wide",
            Self::AesDecryptKeyLockerWide => "aes.dec.locked.wide",
            Self::Sha1HashChoose => "sha1.choose",
            Self::Sha1HashMajority => "sha1.majority",
            Self::Sha1HashParity => "sha1.parity",
            Self::Sha1FixedRotate => "sha1.rotate",
            Self::Sha1ScheduleUpdate0 => "sha1.sched0",
            Self::Sha1ScheduleUpdate1 => "sha1.sched1",
            Self::Sha256Hash => "sha256.hash",
            Self::Sha256Hash2 => "sha256.hash2",
            Self::Sha256ScheduleUpdate0 => "sha256.sched0",
            Self::Sha256ScheduleUpdate1 => "sha256.sched1",
            Self::Sm3PartW1 => "sm3.partw1",
            Self::Sm3PartW2 => "sm3.partw2",
            Self::Sm3SS1 => "sm3.ss1",
            Self::Sm3TT1A => "sm3.tt1a",
            Self::Sm3TT1B => "sm3.tt1b",
            Self::Sm3TT2A => "sm3.tt2a",
            Self::Sm3TT2B => "sm3.tt2b",
            Self::AesExpand128 => "aes.expand128",
            Self::AesExpand256 => "aes.expand256",
            Self::GhashMultiply => "ghash.mul",
            Self::GhashMultiplyAdd => "ghash.muladd",
            Self::Sha2Schedule4 => "sha2.schedule4",
            Self::Sha2Compress2Low => "sha2.compress2.low",
            Self::Sha2Compress2High => "sha2.compress2.high",
            Self::Sm3Expand8 => "sm3.expand8",
            Self::Sm3Compress2 => "sm3.compress2",
            Self::Sm4Expand4 => "sm4.expand4",
            Self::AesDecryptKeyMix => "aes.dec.key.mix",
            Self::AesEncryptRoundMix => "aes.enc.round.mix",
            Self::AesDecryptRoundInvMix => "aes.dec.round.mix.inv",
            Self::Sha512Hash => "sha512.hash",
            Self::Sha512Hash2 => "sha512.hash2",
            Self::Sha512ScheduleUpdate0 => "sha512.sched0",
            Self::Sha512ScheduleUpdate1 => "sha512.sched1",
        }
    }

    /// Effect summary - every vector-crypto primitive is a pure function of its
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
            Self::AesEncryptKeyLocker => "crypto.aesenckl",
            Self::AesDecryptKeyLocker => "crypto.aesdeckl",
            Self::AesEncryptKeyLockerWide => "crypto.aesencwidekl",
            Self::AesDecryptKeyLockerWide => "crypto.aesdecwidekl",
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
            Self::AesExpand128 => "crypto.aes.expand128",
            Self::AesExpand256 => "crypto.aes.expand256",
            Self::GhashMultiply => "crypto.ghash.mul",
            Self::GhashMultiplyAdd => "crypto.ghash.muladd",
            Self::Sha2Schedule4 => "crypto.sha2.schedule4",
            Self::Sha2Compress2Low => "crypto.sha2.compress2.low",
            Self::Sha2Compress2High => "crypto.sha2.compress2.high",
            Self::Sm3Expand8 => "crypto.sm3.expand8",
            Self::Sm3Compress2 => "crypto.sm3.compress2",
            Self::Sm4Expand4 => "crypto.sm4.expand4",
            Self::AesDecryptKeyMix => "crypto.aes.dec.key.mix",
            Self::AesEncryptRoundMix => "crypto.aesemc",
            Self::AesDecryptRoundInvMix => "crypto.aesdimc",
            Self::Sha512Hash => "crypto.sha512h",
            Self::Sha512Hash2 => "crypto.sha512h2",
            Self::Sha512ScheduleUpdate0 => "crypto.sha512su0",
            Self::Sha512ScheduleUpdate1 => "crypto.sha512su1",
        }
    }
}
