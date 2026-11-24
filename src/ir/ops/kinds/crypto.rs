//! Cryptographic and checksum kinds: pointer authentication, scalar ciphers,
//! carry-less products, memory tags and CRC polynomials.

use std::fmt;

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::{aes, integer::BitWidth, sm4};

/// PAC sub-operation carried by [`Kernel::PointerAuth`](crate::ir::ops::Kernel::PointerAuth).
///
/// ARMv8.3 pointer authentication signs a pointer with a cryptographic MAC in
/// its unused high bits, authenticates (validates) it before use, strips the
/// MAC back to the raw pointer, or computes a generic MAC into a register.
/// Distinguishing these is required to faithfully reconstruct the native
/// instruction role; without it every PAC op collapses to "sign".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum PacKind {
    /// Sign a pointer, inserting the authentication code (`pacia`/`pacib`/
    /// `pacda`/`pacdb`/`paciasp`/`pacibsp`/`paciaz`/…).
    Sign,
    /// Authenticate (validate) a signed pointer, faulting/poisoning on tamper
    /// (`autia`/`autib`/`autda`/`autdb`/`autiasp`/`autibsp`/…).
    Authenticate,
    /// Strip the authentication code, yielding the raw pointer (`xpaci`/
    /// `xpacd`/`xpaclri`).
    Strip,
    /// Compute a generic authentication code into a register (`pacga`).
    GenericMac,
    /// Sign a pointer with a second modifier beside the first: the address of
    /// the instruction (`paciasppc`, `pacibsppc`, `pacnbiasppc`, ...).
    SignDiversified,
    /// Authenticate a pointer signed with a second modifier: the address of
    /// the signing instruction a label or a register names, or `x15`
    /// (`autiasppc`, `autiasppcr`, `autia171615`, ...).
    AuthenticateDiversified,
}

impl PacKind {
    /// Returns the neutral display mnemonic for this sub-operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Sign => "pac.sign",
            Self::Authenticate => "pac.auth",
            Self::Strip => "pac.strip",
            Self::GenericMac => "pac.genmac",
            Self::SignDiversified => "pac.sign.diversified",
            Self::AuthenticateDiversified => "pac.auth.diversified",
        }
    }

    /// The values the step reads, in order: the pointer (the value for
    /// `pacga`), the modifier (zero where the instruction names none:
    /// `paciza`, `paciaz`), a diversified step's second modifier, and the
    /// key's low and high halves. A strip reads the pointer alone.
    #[must_use]
    pub const fn input_count(self) -> usize {
        match self {
            Self::Sign | Self::Authenticate | Self::GenericMac => 4,
            Self::SignDiversified | Self::AuthenticateDiversified => 5,
            Self::Strip => 1,
        }
    }

    /// Returns the stable display / fingerprint key for this sub-operation.
    ///
    /// [`Kernel::PointerAuth`](crate::ir::ops::Kernel::PointerAuth) reads it, so the four PAC operations keep
    /// four fingerprints. One spelling for the whole family would collapse them
    /// onto a single identity - including the sign/authenticate pair, whose
    /// *difference* is the security property.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Sign => "compute.pac.sign",
            Self::Authenticate => "compute.pac.auth",
            Self::Strip => "compute.pac.strip",
            Self::GenericMac => "compute.pac.genmac",
            Self::SignDiversified => "compute.pac.sign.diversified",
            Self::AuthenticateDiversified => "compute.pac.auth.diversified",
        }
    }
}

impl fmt::Display for PacKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sign => "sign",
            Self::Authenticate => "auth",
            Self::Strip => "strip",
            Self::GenericMac => "genmac",
            Self::SignDiversified => "sign.diversified",
            Self::AuthenticateDiversified => "auth.diversified",
        };
        f.write_str(name)
    }
}

/// The scalar block-cipher step a [`Kernel::ScalarCipher`](crate::ir::ops::Kernel::ScalarCipher) computes:
/// RISC-V Zkne and Zknd's AES steps, on a 32-bit column (`aes32*`) or a
/// 64-bit half of the state (`aes64*`), and Zksed's SM4 steps. Each is a pure
/// function of its sources and the byte select or round number its
/// instruction encodes, which the operation reads as its last input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum ScalarCipherKind {
    /// One byte of the second source, the byte select's, through the AES forward S-box and xor'd into the first source at that byte's place (`aes32esi`).
    Aes32EncryptByte,
    /// As [`Self::Aes32EncryptByte`], the substituted byte taken through its MixColumns column first (`aes32esmi`).
    Aes32EncryptByteMix,
    /// One byte of the second source through the AES inverse S-box, xor'd into the first source at its place (`aes32dsi`).
    Aes32DecryptByte,
    /// As [`Self::Aes32DecryptByte`], through the inverse MixColumns column first (`aes32dsmi`).
    Aes32DecryptByteMix,
    /// One half of the AES state the two sources hold after ShiftRows and SubBytes (`aes64es`).
    Aes64EncryptHalf,
    /// As [`Self::Aes64EncryptHalf`], then MixColumns on its two columns (`aes64esm`).
    Aes64EncryptHalfMix,
    /// One half of the AES state after the inverse ShiftRows and the inverse SubBytes (`aes64ds`).
    Aes64DecryptHalf,
    /// As [`Self::Aes64DecryptHalf`], then the inverse MixColumns (`aes64dsm`).
    Aes64DecryptHalfMix,
    /// The inverse MixColumns of the two columns the source holds, for the equivalent inverse cipher's key schedule (`aes64im`).
    Aes64InverseMix,
    /// The first step of an AES key-schedule round: the source's high word rotated (except for round 10), substituted and xor'd with the round constant the round number selects, in both words (`aes64ks1i`).
    Aes64KeyStep1,
    /// The second step of an AES key-schedule round: the first source's high word xor'd into each word of the second, cumulatively (`aes64ks2`).
    Aes64KeyStep2,
    /// One byte of the second source, the byte select's, through the SM4 S-box and the cipher's linear transform, rotated to that byte's place and xor'd into the first source (`sm4ed`).
    Sm4EncryptByte,
    /// As [`Self::Sm4EncryptByte`] with the key schedule's linear transform (`sm4ks`).
    Sm4KeyByte,
}

impl ScalarCipherKind {
    /// Returns the inputs: the two sources and the byte select of a
    /// byte-wise step (`aes32esi`, `sm4ed`), the one source of `aes64im`, and
    /// the two of the other 64-bit steps, `aes64ks1i`'s round number a
    /// constant in its second place.
    #[must_use]
    pub const fn input_count(self) -> usize {
        match self {
            Self::Aes32EncryptByte
            | Self::Aes32EncryptByteMix
            | Self::Aes32DecryptByte
            | Self::Aes32DecryptByteMix
            | Self::Sm4EncryptByte
            | Self::Sm4KeyByte => 3,
            Self::Aes64InverseMix => 1,
            Self::Aes64EncryptHalf
            | Self::Aes64EncryptHalfMix
            | Self::Aes64DecryptHalf
            | Self::Aes64DecryptHalfMix
            | Self::Aes64KeyStep1
            | Self::Aes64KeyStep2 => 2,
        }
    }

    /// The value this step writes for the sources it reads.
    ///
    /// `inputs` are the step's sources in the order its instruction names
    /// them, the byte select or round number its encoding holds last: the
    /// first and second source and the byte select for an `aes32` or SM4
    /// step, the two halves of the state, low first, for an `aes64` round
    /// step, the source for `aes64im`, the source and the round number for
    /// `aes64ks1i`, and the two sources for `aes64ks2`. A step on a 32-bit
    /// column writes the column sign-extended to 64 bits, as RV64 writes it;
    /// the low 32 bits are what RV32 writes.
    ///
    /// # Returns
    ///
    /// The value, or `None` for a different number of inputs, a byte select
    /// past 3, or a round number past 10, which is reserved.
    #[must_use]
    // A word or doubleword of a wider value is its low bits: truncation is
    // the point.
    #[allow(clippy::cast_possible_truncation)]
    pub fn evaluate(self, inputs: &[u64]) -> Option<u64> {
        match self {
            Self::Aes32EncryptByte
            | Self::Aes32EncryptByteMix
            | Self::Aes32DecryptByte
            | Self::Aes32DecryptByteMix
            | Self::Sm4EncryptByte
            | Self::Sm4KeyByte => {
                let [first, second, select] = *inputs else {
                    return None;
                };
                if select > 3 {
                    return None;
                }
                let shift = u32::try_from(select).ok()?.checked_mul(8)?;
                let byte = second.checked_shr(shift)?.to_le_bytes()[0];
                let column = match self {
                    Self::Aes32EncryptByte => u32::from(aes::substitute(byte)),
                    Self::Aes32DecryptByte => u32::from(aes::inverse_substitute(byte)),
                    // The substituted byte alone in its column, mixed: the
                    // byte's share of the mixed column, as row zero's.
                    Self::Aes32EncryptByteMix => {
                        aes::mix_columns(u128::from(aes::substitute(byte)), false)? as u32
                    }
                    Self::Aes32DecryptByteMix => {
                        aes::mix_columns(u128::from(aes::inverse_substitute(byte)), true)? as u32
                    }
                    Self::Sm4EncryptByte => sm4::Transform::Cipher.byte(byte)?,
                    _ => sm4::Transform::Key.byte(byte)?,
                };
                let word = first as u32 ^ column.rotate_left(shift);
                Some(i64::from(word.cast_signed()).cast_unsigned())
            }
            Self::Aes64EncryptHalf
            | Self::Aes64EncryptHalfMix
            | Self::Aes64DecryptHalf
            | Self::Aes64DecryptHalfMix => {
                let [low, high] = *inputs else {
                    return None;
                };
                let (inverse, last) = match self {
                    Self::Aes64EncryptHalf => (false, true),
                    Self::Aes64EncryptHalfMix => (false, false),
                    Self::Aes64DecryptHalf => (true, true),
                    _ => (true, false),
                };
                let state = (u128::from(high) << 64) | u128::from(low);
                Some(aes::round(inverse, last, false, state, 0)? as u64)
            }
            Self::Aes64InverseMix => {
                let [columns] = *inputs else {
                    return None;
                };
                Some(aes::mix_columns(u128::from(columns), true)? as u64)
            }
            Self::Aes64KeyStep1 => {
                let [source, round] = *inputs else {
                    return None;
                };
                // Round ten substitutes the word unrotated and adds no
                // constant: the odd steps of the AES-256 schedule.
                let constant = match round {
                    0..=9 => [
                        0x01u32, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
                    ]
                    .get(usize::try_from(round).ok()?)
                    .copied()?,
                    10 => 0,
                    _ => return None,
                };
                let high = (source >> 32) as u32;
                let word = if round == 10 {
                    high
                } else {
                    high.rotate_right(8)
                };
                let substituted = u32::from_le_bytes(word.to_le_bytes().map(aes::substitute));
                let word = u64::from(substituted ^ constant);
                Some((word << 32) | word)
            }
            Self::Aes64KeyStep2 => {
                let [first, second] = *inputs else {
                    return None;
                };
                let high = (first >> 32) as u32;
                let low = high ^ second as u32;
                let upper = low ^ (second >> 32) as u32;
                Some((u64::from(upper) << 32) | u64::from(low))
            }
        }
    }

    /// Returns the neutral display mnemonic for this step.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Aes32EncryptByte => "aes32esi",
            Self::Aes32EncryptByteMix => "aes32esmi",
            Self::Aes32DecryptByte => "aes32dsi",
            Self::Aes32DecryptByteMix => "aes32dsmi",
            Self::Aes64EncryptHalf => "aes64es",
            Self::Aes64EncryptHalfMix => "aes64esm",
            Self::Aes64DecryptHalf => "aes64ds",
            Self::Aes64DecryptHalfMix => "aes64dsm",
            Self::Aes64InverseMix => "aes64im",
            Self::Aes64KeyStep1 => "aes64ks1i",
            Self::Aes64KeyStep2 => "aes64ks2",
            Self::Sm4EncryptByte => "sm4ed",
            Self::Sm4KeyByte => "sm4ks",
        }
    }

    /// Returns the stable display / fingerprint key for this step.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Aes32EncryptByte => "compute.aes32esi",
            Self::Aes32EncryptByteMix => "compute.aes32esmi",
            Self::Aes32DecryptByte => "compute.aes32dsi",
            Self::Aes32DecryptByteMix => "compute.aes32dsmi",
            Self::Aes64EncryptHalf => "compute.aes64es",
            Self::Aes64EncryptHalfMix => "compute.aes64esm",
            Self::Aes64DecryptHalf => "compute.aes64ds",
            Self::Aes64DecryptHalfMix => "compute.aes64dsm",
            Self::Aes64InverseMix => "compute.aes64im",
            Self::Aes64KeyStep1 => "compute.aes64ks1i",
            Self::Aes64KeyStep2 => "compute.aes64ks2",
            Self::Sm4EncryptByte => "compute.sm4ed",
            Self::Sm4KeyByte => "compute.sm4ks",
        }
    }
}

/// The part of a carry-less product [`Kernel::CarrylessMultiply`] yields.
///
/// The carry-less (polynomial, GF(2)) product of two `n`-bit values has
/// `2n - 1` significant bits. An instruction keeps all of it, `2n` bits wide
/// (x86 `pclmulqdq`, Arm `pmull`), or, writing one `n`-bit register, one
/// slice of it: the low `n` bits (RISC-V `clmul`), bits `2n - 1` down to `n`
/// (`clmulh`), or bits `2n - 2` down to `n - 1` (`clmulr`), which is the
/// product of the bit-reversed operands, reversed. `n` is the width of the
/// operands.
///
/// [`Kernel::CarrylessMultiply`]: crate::ir::ops::Kernel::CarrylessMultiply
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum CarrylessPart {
    /// The low `n` bits of the product (`clmul`).
    Low,
    /// The high `n` bits of the `2n`-bit product (`clmulh`).
    High,
    /// Bits `2n - 2` down to `n - 1` of the product (`clmulr`).
    Reversed,
    /// The whole product, `2n` bits wide (`pclmulqdq`, `pmull`).
    Full,
}

impl CarrylessPart {
    /// This part of the carry-less product of `left` and `right`, each read
    /// as its low `bits` bits.
    ///
    /// # Returns
    ///
    /// The part, `bits` wide, or `2 * bits` wide for [`Self::Full`]; `None`
    /// for a width other than 8, 16, 32, 64 or 128, and for the whole
    /// product of 128-bit factors, which no `u128` holds.
    #[must_use]
    pub fn product(self, left: u128, right: u128, bits: u32) -> Option<u128> {
        if !matches!(bits, 8 | 16 | 32 | 64 | 128) || self == Self::Full && bits == 128 {
            return None;
        }
        let mask = BitWidth::new(bits).checked_mask_u128()?;
        let (left, right) = (left & mask, right & mask);
        // The `2 * bits`-bit product, its low and high 128 bits.
        let (mut low, mut high) = (0u128, 0u128);
        for bit in (0..bits).filter(|bit| right.checked_shr(*bit).is_some_and(|b| b & 1 != 0)) {
            low ^= left.checked_shl(bit).unwrap_or(0);
            high ^= 128u32
                .checked_sub(bit)
                .and_then(|shift| left.checked_shr(shift))
                .unwrap_or(0);
        }
        // Bits `shift` up of the product.
        let from = |shift: u32| -> u128 {
            let below = low.checked_shr(shift).unwrap_or(0);
            let above = 128u32
                .checked_sub(shift)
                .and_then(|back| high.checked_shl(back))
                .unwrap_or(0);
            match shift {
                0 => low,
                128.. => high.checked_shr(shift.saturating_sub(128)).unwrap_or(0),
                _ => below | above,
            }
        };
        Some(match self {
            Self::Low => low & mask,
            Self::High => from(bits) & mask,
            Self::Reversed => from(bits.checked_sub(1)?) & mask,
            Self::Full => low & BitWidth::new(bits.checked_mul(2)?).checked_mask_u128()?,
        })
    }

    /// Returns the IR name of this part's product, such as
    /// `mul.carryless.high`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Low => "mul.carryless",
            Self::High => "mul.carryless.high",
            Self::Reversed => "mul.carryless.reversed",
            Self::Full => "mul.carryless.long",
        }
    }

    /// Returns the neutral display mnemonic for this part.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Low => "clmul",
            Self::High => "clmulh",
            Self::Reversed => "clmulr",
            Self::Full => "pclmul",
        }
    }

    /// Returns the stable display / fingerprint key for this part, which
    /// [`Kernel::CarrylessMultiply`] reads, so the four parts keep four
    /// fingerprints.
    ///
    /// [`Kernel::CarrylessMultiply`]: crate::ir::ops::Kernel::CarrylessMultiply
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Low => "compute.clmul",
            Self::High => "compute.clmulh",
            Self::Reversed => "compute.clmulr",
            Self::Full => "compute.clmul.full",
        }
    }
}

/// The allocation-tag computation a [`Kernel::MemoryTag`](crate::ir::ops::Kernel::MemoryTag) performs on an
/// address, the Arm Memory Tagging Extension's tag in bits 59 to 56.
///
/// Each reads the exclusion mask GCR_EL1 holds, a register the row names
/// beside its operands, and chooses a tag the mask does not exclude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum MemoryTagKind {
    /// A random tag inserted into the address, excluding the tags of the mask
    /// operand as well (`irg`). The tag comes from a generator whose state
    /// RGSR_EL1 holds, or from an implementation's own source.
    Random,
    /// An offset added to the address and a tag offset to its tag, the tag
    /// stepping past the excluded ones (`addg`, `subg`).
    Add,
}

impl MemoryTagKind {
    /// Returns the neutral display mnemonic for this tag computation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Random => "tag.random",
            Self::Add => "tag.add",
        }
    }

    /// Returns the stable display / fingerprint key for this tag computation.
    ///
    /// [`Kernel::MemoryTag`](crate::ir::ops::Kernel::MemoryTag) reads it, so the two computations keep two
    /// fingerprints.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Random => "compute.tag.random",
            Self::Add => "compute.tag.add",
        }
    }
}

/// The polynomial a [`Kernel::Checksum`](crate::ir::ops::Kernel::Checksum) accumulates a CRC-32 by.
///
/// Both are reflected 32-bit CRCs, and a step of either is the same shape:
/// the accumulator and the data in, the accumulator out. What they compute
/// differs in the polynomial alone, so a program checksumming with one is not
/// checksumming with the other: zlib's and Ethernet's `crc32` is the IEEE one,
/// iSCSI's, ext4's and SSE4.2's `crc32` the Castagnoli one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum CrcPolynomial {
    /// The IEEE 802.3 polynomial `0x04C11DB7` (reflected `0xEDB88320`): ARM
    /// and AArch64 `crc32b`/`crc32h`/`crc32w`/`crc32x`.
    Ieee,
    /// The Castagnoli polynomial `0x1EDC6F41` (reflected `0x82F63B78`): x86
    /// `crc32`, ARM and AArch64 `crc32c*`.
    Castagnoli,
}

impl CrcPolynomial {
    /// The polynomial reflected, as a step of the checksum reads it: least
    /// significant bit first.
    #[must_use]
    pub const fn reflected(self) -> u32 {
        match self {
            Self::Ieee => 0xedb8_8320,
            Self::Castagnoli => 0x82f6_3b78,
        }
    }

    /// One step of the checksum: `crc` updated by the low `bits` bits of
    /// `data`, least significant first, with neither inverted, as every
    /// instruction computing one steps it (the inversions before the first
    /// step and after the last are the program's).
    ///
    /// # Returns
    ///
    /// The updated checksum, or `None` for a data width other than 8, 16, 32
    /// or 64.
    #[must_use]
    pub fn step(self, crc: u32, data: u64, bits: u32) -> Option<u32> {
        if !matches!(bits, 8 | 16 | 32 | 64) {
            return None;
        }
        let polynomial = self.reflected();
        let mut crc = crc;
        for bit in 0..bits {
            let carry = (u64::from(crc) ^ data.checked_shr(bit)?) & 1;
            crc >>= 1;
            if carry != 0 {
                crc ^= polynomial;
            }
        }
        Some(crc)
    }

    /// Returns the neutral display mnemonic for this polynomial's step.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Ieee => "crc32",
            Self::Castagnoli => "crc32c",
        }
    }

    /// Returns the stable display / fingerprint key for this polynomial's
    /// step.
    ///
    /// [`Kernel::Checksum`](crate::ir::ops::Kernel::Checksum) reads it, so the two polynomials keep two
    /// fingerprints.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Ieee => "compute.crc32",
            Self::Castagnoli => "compute.crc32c",
        }
    }
}
