//! The SM4 block cipher's substitution and linear transforms (GB/T 32907).
//!
//! A round of SM4 takes each byte of a 32-bit word through the S-box, then
//! the word through a linear transform of rotations: the cipher's `L` for the
//! data, the key schedule's `L'` for the round keys. Both transforms commute
//! with rotation, so a byte substituted and transformed alone and rotated to
//! its place is that byte's share of the transformed word, which is what the
//! scalar steps compute one byte at a time. The vector steps compute four
//! whole rounds of a 128-bit block.

/// The SM4 S-box.
///
/// It is the affine map of the circulant matrix with first row `0xa7` and the
/// constant `0xd3`, applied before and after the inverse in the field of the
/// polynomial `x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1`.
const SBOX: [u8; 256] = [
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
];

/// The linear transform a round applies after the S-box: the cipher's `L`,
/// `B ^ (B <<< 2) ^ (B <<< 10) ^ (B <<< 18) ^ (B <<< 24)`, or the key
/// schedule's `L'`, `B ^ (B <<< 13) ^ (B <<< 23)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// The cipher's `L`, of `vsm4rnds4`, `sm4e` and `sm4ed`.
    Cipher,
    /// The key schedule's `L'`, of `vsm4key4`, `sm4ekey` and `sm4ks`.
    Key,
}

impl Transform {
    /// `byte` through the S-box and the transform, as the low byte of a word.
    #[must_use]
    pub fn byte(self, byte: u8) -> Option<u32> {
        let substituted = SBOX.get(usize::from(byte)).copied().map(u32::from)?;
        Some(self.linear(substituted))
    }

    /// `word`'s four bytes each through the S-box, then the transform: the
    /// round function `T` (or `T'`) of GB/T 32907.
    #[must_use]
    pub fn word(self, word: u32) -> Option<u32> {
        let substitute = |byte: u8| SBOX.get(usize::from(byte)).copied();
        let [b0, b1, b2, b3] = word.to_le_bytes();
        let substituted = [
            substitute(b0)?,
            substitute(b1)?,
            substitute(b2)?,
            substitute(b3)?,
        ];
        Some(self.linear(u32::from_le_bytes(substituted)))
    }

    /// Four rounds of the state `[X0, X1, X2, X3]` (word 0 the oldest) by the
    /// round keys, or for the key schedule the constants, `keys`, word 0 the
    /// first round's: each round's `X0 ^ T(X1 ^ X2 ^ X3 ^ K)` is the next
    /// state's last word. The result is the four new words, the first
    /// round's in word 0: `vsm4rnds4`'s and `sm4e`'s of a 128-bit block, and
    /// with the key schedule's transform `vsm4key4`'s and `sm4ekey`'s.
    #[must_use]
    pub fn rounds4(self, state: [u32; 4], keys: [u32; 4]) -> Option<[u32; 4]> {
        let [mut x0, mut x1, mut x2, mut x3] = state;
        for key in keys {
            let next = x0 ^ self.word(x1 ^ x2 ^ x3 ^ key)?;
            (x0, x1, x2, x3) = (x1, x2, x3, next);
        }
        Some([x0, x1, x2, x3])
    }

    /// RISC-V's `vsm4k` of one element group: the next four round keys of
    /// the four `key` (word 0 the oldest) by the key schedule's constants of
    /// round group `group` (its low three bits).
    #[must_use]
    pub fn expand4(key: [u32; 4], group: u8) -> Option<[u32; 4]> {
        Self::Key.rounds4(key, Self::constants(group))
    }

    /// The key schedule's constants `CK[4 * group]` to `CK[4 * group + 3]`
    /// of round group `group` (its low three bits): byte `j` of `CK[i]`,
    /// from the most significant, is `(4i + j) * 7` modulo 256.
    #[must_use]
    pub fn constants(group: u8) -> [u32; 4] {
        let first = (group & 7).wrapping_mul(16);
        std::array::from_fn(|word| {
            let index = first.wrapping_add(u8::try_from(word).unwrap_or(0).wrapping_mul(4));
            u32::from_be_bytes(std::array::from_fn(|byte| {
                index
                    .wrapping_add(u8::try_from(byte).unwrap_or(0))
                    .wrapping_mul(7)
            }))
        })
    }

    /// The transform of a substituted word.
    const fn linear(self, word: u32) -> u32 {
        match self {
            Self::Cipher => {
                word ^ word.rotate_left(2)
                    ^ word.rotate_left(10)
                    ^ word.rotate_left(18)
                    ^ word.rotate_left(24)
            }
            Self::Key => word ^ word.rotate_left(13) ^ word.rotate_left(23),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is the S-box its affine-inverse-affine construction gives,
    /// and a bijection.
    #[test]
    fn the_sbox_is_its_construction() {
        let multiply = |mut left: u16, mut right: u16| {
            let mut product = 0u16;
            while right != 0 {
                if right & 1 != 0 {
                    product ^= left;
                }
                right >>= 1;
                left <<= 1;
                if left & 0x100 != 0 {
                    left ^= 0x1f5;
                }
            }
            product
        };
        let inverse = |value: u16| {
            let (mut result, mut power, mut exponent) = (1u16, value, 254u32);
            while exponent != 0 {
                if exponent & 1 != 0 {
                    result = multiply(result, power);
                }
                power = multiply(power, power);
                exponent >>= 1;
            }
            if value == 0 { 0 } else { result }
        };
        let affine = |value: u16| {
            let mut result = 0u16;
            for bit in 0..8 {
                let row = 0xa7u8.rotate_left(bit);
                let parity = (u16::from(row) & value).count_ones() & 1;
                result |= u16::try_from(parity).unwrap() << bit;
            }
            result ^ 0xd3
        };
        let mut seen = [false; 256];
        for byte in 0..=255u8 {
            let expected = affine(inverse(affine(u16::from(byte))));
            assert_eq!(u16::from(SBOX[usize::from(byte)]), expected, "{byte:#04x}");
            seen[usize::from(SBOX[usize::from(byte)])] = true;
        }
        assert!(seen.iter().all(|seen| *seen));
    }

    /// Four rounds at a time compose the cipher: GB/T 32907's example, the
    /// key `0123456789abcdeffedcba9876543210` enciphering itself to
    /// `681edf34d206965e86b3e94f536e4246`. The key schedule is eight steps of
    /// the key with `FK` added (`vsm4key4` by the constants `CK`, RISC-V's
    /// `vsm4k` by its round group), the cipher eight `vsm4rnds4` steps by
    /// the round keys, its last four words in reverse; the round keys in
    /// reverse decipher it back. The constants are those the RISC-V
    /// specification lists, and only a round group's low three bits count.
    #[test]
    fn four_rounds_at_a_time_compose_the_cipher() {
        const FK: [u32; 4] = [0xa3b1_bac6, 0x56aa_3350, 0x677d_9197, 0xb270_22dc];
        let key = [0x0123_4567, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210];
        // The first and the last of the constants as the RISC-V
        // specification lists them.
        assert_eq!(
            Transform::constants(0),
            [0x0007_0e15, 0x1c23_2a31, 0x383f_464d, 0x545b_6269]
        );
        assert_eq!(
            Transform::constants(7),
            [0x1017_1e25, 0x2c33_3a41, 0x484f_565d, 0x646b_7279]
        );
        let mut words = [
            key[0] ^ FK[0],
            key[1] ^ FK[1],
            key[2] ^ FK[2],
            key[3] ^ FK[3],
        ];
        let mut round_keys = Vec::new();
        for group in 0..8 {
            words = Transform::expand4(words, group).unwrap();
            round_keys.extend(words);
        }
        assert_eq!(Transform::expand4(key, 9), Transform::expand4(key, 1));
        let crypt = |block: [u32; 4], keys: &[u32]| {
            let mut state = block;
            for group in keys.chunks(4) {
                state = Transform::Cipher
                    .rounds4(state, group.try_into().unwrap())
                    .unwrap();
            }
            let [x0, x1, x2, x3] = state;
            [x3, x2, x1, x0]
        };
        let cipher = crypt(key, &round_keys);
        assert_eq!(cipher, [0x681e_df34, 0xd206_965e, 0x86b3_e94f, 0x536e_4246]);
        round_keys.reverse();
        assert_eq!(crypt(cipher, &round_keys), key);
    }
}
