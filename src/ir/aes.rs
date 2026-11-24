//! Exact AES round transformations over little-endian packed column-major bytes.

/// Evaluates an AES round with explicit inverse, final-round and key-order policies.
///
/// Input byte zero is the low byte, and each consecutive four bytes form one
/// AES state column. A key before inverse mixing implements the standard
/// inverse cipher; a key after inverse mixing implements its equivalent form.
#[must_use]
pub fn round(
    inverse: bool,
    last: bool,
    key_before_mix: bool,
    state: u128,
    key: u128,
) -> Option<u128> {
    let source = state.to_le_bytes();
    let mut result = [0u8; 16];
    for (column, bytes) in result.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        for (row, byte) in bytes.iter_mut().enumerate() {
            let delta = if inverse {
                4usize.checked_sub(row)?
            } else {
                row
            };
            let index = column
                .checked_add(delta)?
                .checked_rem(4)?
                .checked_mul(4)?
                .checked_add(row)?;
            let value = *source.get(index)?;
            *byte = if inverse {
                inverse_substitute(value)
            } else {
                substitute(value)
            };
        }
    }
    if key_before_mix {
        result = (u128::from_le_bytes(result) ^ key).to_le_bytes();
    }
    if !last {
        mix(&mut result, inverse)?;
    }
    let result = u128::from_le_bytes(result);
    Some(if key_before_mix { result } else { result ^ key })
}

/// Multiplies byte polynomials modulo x^8+x^4+x^3+x+1, the field AES and
/// the GF(2^8) instructions (`gf2p8mulb`) compute in.
#[must_use]
pub fn multiply(mut left: u8, mut right: u8) -> u8 {
    let mut result = 0;
    for _ in 0..8 {
        if right & 1 != 0 {
            result ^= left;
        }
        let high = left & 0x80 != 0;
        left = left.wrapping_shl(1);
        if high {
            left ^= 0x1b;
        }
        right = right.wrapping_shr(1);
    }
    result
}

/// Returns the multiplicative inverse in that field, with zero mapped to
/// zero.
#[must_use]
pub fn reciprocal(value: u8) -> u8 {
    let mut result = 1;
    let mut power = value;
    let mut exponent = 254u16;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = multiply(result, power);
        }
        power = multiply(power, power);
        exponent = exponent.wrapping_shr(1);
    }
    result
}

/// Applies the AES forward affine substitution.
#[must_use]
pub fn substitute(value: u8) -> u8 {
    let value = reciprocal(value);
    value
        ^ value.rotate_left(1)
        ^ value.rotate_left(2)
        ^ value.rotate_left(3)
        ^ value.rotate_left(4)
        ^ 0x63
}

/// Applies the AES inverse affine substitution.
#[must_use]
pub fn inverse_substitute(value: u8) -> u8 {
    reciprocal(value.rotate_left(1) ^ value.rotate_left(3) ^ value.rotate_left(6) ^ 0x05)
}

/// The key schedule's SubWord: each byte of `word` through the S-box.
#[must_use]
pub fn substitute_word(word: u32) -> u32 {
    u32::from_le_bytes(word.to_le_bytes().map(substitute))
}

/// The key schedule's round constant `Rcon[index]`, `x^index` in the AES
/// field: `0x01`, `0x02` and on to `0x1b` and `0x36`.
#[must_use]
fn round_constant(index: u8) -> u8 {
    (0..index).fold(1, |constant, _| multiply(constant, 2))
}

/// RISC-V's `vaeskf1` of one round key: AES-128's next round key, of round
/// `round` (1 to 10; another value of its low four bits is the one with
/// bit 3 inverted). Words are word 0 first, each its key bytes in order from
/// its low byte, so RotWord rotates a word right by a byte.
#[must_use]
pub fn expand128(key: [u32; 4], round: u8) -> [u32; 4] {
    let mut round = round & 0xf;
    if round == 0 || round > 10 {
        round ^= 8;
    }
    let constant = u32::from(round_constant(round.saturating_sub(1)));
    let first = substitute_word(key[3].rotate_right(8)) ^ constant ^ key[0];
    let second = first ^ key[1];
    let third = second ^ key[2];
    [first, second, third, third ^ key[3]]
}

/// RISC-V's `vaeskf2` of the previous and the current round key: AES-256's
/// next round key, of round `round` (2 to 14; another value of its low four
/// bits is the one with bit 3 inverted). An odd round substitutes the
/// current key's last word, an even one also rotates it and adds the round
/// constant of half the round, less one.
#[must_use]
pub fn expand256(previous: [u32; 4], current: [u32; 4], round: u8) -> [u32; 4] {
    let mut round = round & 0xf;
    if !(2..=14).contains(&round) {
        round ^= 8;
    }
    let last = current[3];
    let mixed = if round & 1 == 1 {
        substitute_word(last)
    } else {
        substitute_word(last.rotate_right(8))
            ^ u32::from(round_constant((round >> 1).saturating_sub(1)))
    };
    let first = mixed ^ previous[0];
    let second = first ^ previous[1];
    let third = second ^ previous[2];
    [first, second, third, third ^ previous[3]]
}

/// The AES MixColumns, or its inverse, of every column of `state`, packed
/// as [`round`] packs it.
#[must_use]
pub fn mix_columns(state: u128, inverse: bool) -> Option<u128> {
    let mut bytes = state.to_le_bytes();
    mix(&mut bytes, inverse)?;
    Some(u128::from_le_bytes(bytes))
}

/// Applies the forward or inverse column matrix in the AES byte field.
fn mix(bytes: &mut [u8; 16], inverse: bool) -> Option<()> {
    let coefficients = if inverse {
        [14u8, 11, 13, 9]
    } else {
        [2u8, 3, 1, 1]
    };
    for column in bytes.as_chunks_mut::<4>().0 {
        let source = [
            *column.first()?,
            *column.get(1)?,
            *column.get(2)?,
            *column.get(3)?,
        ];
        for (row, byte) in column.iter_mut().enumerate() {
            let mut result = 0;
            for (offset, coefficient) in coefficients.iter().enumerate() {
                let index = row.checked_add(offset)?.checked_rem(4)?;
                result ^= multiply(*source.get(index)?, *coefficient);
            }
            *byte = result;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ops::VectorCryptoKind;

    #[test]
    fn aes_rounds_match_fips_197_forward_and_inverse_intermediate_states() {
        let state = 0x00102030405060708090a0b0c0d0e0f0u128.swap_bytes();
        let key = 0xd6aa74fdd2af72fadaa678f1d6ab76feu128.swap_bytes();
        assert_eq!(
            round(false, false, false, state, key),
            Some(0x89d810e8855ace682d1843d8cb128fe4u128.swap_bytes())
        );
        let state = 0x7ad5fda789ef4e272bca100b3d9ff59fu128.swap_bytes();
        let key = 0x549932d1f08557681093ed9cbe2c974eu128.swap_bytes();
        assert_eq!(
            round(true, false, true, state, key),
            Some(0x54d990a16ba09ab596bbf40ea111702fu128.swap_bytes())
        );
        assert_ne!(
            round(true, false, false, state, key),
            round(true, false, true, state, key)
        );
        let mut transformed = key.to_le_bytes();
        mix(&mut transformed, true).unwrap();
        assert_eq!(
            round(true, false, false, state, u128::from_le_bytes(transformed)),
            round(true, false, true, state, key)
        );
    }

    /// The key expansion steps' round keys encipher FIPS-197's Appendix C.1
    /// and C.3 examples: AES-128's ten `vaeskf1` rounds from the key, and
    /// AES-256's rounds 2 to 14 of `vaeskf2` from the key's two halves, the
    /// rounds [`round`] computes by them. A round number out of range is the
    /// one with bit 3 inverted, and bit 4 is ignored.
    #[test]
    fn the_key_expansion_steps_encipher_the_fips_197_examples() {
        let block = |hex: &str| {
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).unwrap())
                .collect();
            bytes
        };
        let words = |bytes: &[u8]| -> [u32; 4] {
            std::array::from_fn(|index| {
                u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
            })
        };
        let packed = |words: [u32; 4]| {
            words
                .iter()
                .rev()
                .fold(0u128, |block, word| block << 32 | u128::from(*word))
        };
        let encipher = |keys: &[[u32; 4]], plain: &str| {
            let rounds = keys.len() - 1;
            let mut state = u128::from_le_bytes(block(plain).try_into().unwrap()) ^ packed(keys[0]);
            for (index, key) in keys.iter().enumerate().skip(1) {
                state = round(false, index == rounds, false, state, packed(*key)).unwrap();
            }
            state.to_le_bytes().to_vec()
        };
        let plain = "00112233445566778899aabbccddeeff";
        let key = block("000102030405060708090a0b0c0d0e0f");
        let mut keys = vec![words(&key)];
        for number in 1..=10 {
            keys.push(expand128(*keys.last().unwrap(), number));
        }
        assert_eq!(
            encipher(&keys, plain),
            block("69c4e0d86a7b0430d8cdb78070b4c55a")
        );
        let key = block("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        let mut keys = vec![words(&key[..16]), words(&key[16..])];
        for number in 2..=14 {
            let (previous, current) = (keys[keys.len() - 2], keys[keys.len() - 1]);
            keys.push(expand256(previous, current, number));
        }
        assert_eq!(
            encipher(&keys, plain),
            block("8ea2b7ca516745bfeafc49904b496089")
        );
        let (previous, current) = (keys[0], keys[1]);
        for (outside, inside) in [(0, 8), (11, 3), (15, 7), (0x13, 3)] {
            assert_eq!(
                expand128(current, outside),
                expand128(current, inside),
                "{outside}"
            );
        }
        for (outside, inside) in [(0, 8), (1, 9), (15, 7), (0x12, 2)] {
            assert_eq!(
                expand256(previous, current, outside),
                expand256(previous, current, inside),
                "{outside}"
            );
        }
    }

    /// Arm's inverse round with inverse mix-columns adds the round key first,
    /// then inverse-shifts, inverse-substitutes and inverse-mixes: from
    /// FIPS-197's Appendix C.1 round 1 start state, its inverse-shifted and
    /// inverse-substituted state mixed back, the key a no-op at zero and the
    /// same keyed state reached from its two halves.
    #[test]
    fn an_arm_inverse_aes_round_with_mix_columns_adds_the_key_first() {
        let start = 0x7ad5fda789ef4e272bca100b3d9ff59fu128.swap_bytes();
        let substituted = 0xbd6e7c3df2b5779e0b61216e8b10b689u128.swap_bytes();
        let mut expected = substituted.to_le_bytes();
        mix(&mut expected, true).unwrap();
        let expected = u128::from_le_bytes(expected);
        let round = VectorCryptoKind::AesDecryptRoundInvMix;
        assert_eq!(round.aes_round(start, 0), Some(expected));
        let key = 0x549932d1f08557681093ed9cbe2c974eu128.swap_bytes();
        assert_eq!(round.aes_round(start ^ key, key), Some(expected));
    }

    #[test]
    fn aes_substitution_matches_literal_sbox_and_bijection_over_all_bytes() {
        let first = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7,
            0xab, 0x76,
        ];
        for (index, expected) in first.into_iter().enumerate() {
            assert_eq!(substitute(index as u8), expected);
        }
        for byte in 0..=255u8 {
            assert_eq!(inverse_substitute(substitute(byte)), byte);
            assert_eq!(substitute(inverse_substitute(byte)), byte);
        }
    }
    #[test]
    fn aes_final_rounds_match_fips_197_without_mixing() {
        assert_eq!(
            round(
                false,
                true,
                false,
                0xbd6e7c3df2b5779e0b61216e8b10b689u128.swap_bytes(),
                0x13111d7fe3944a17f307a78b4d2b30c5u128.swap_bytes()
            ),
            Some(0x69c4e0d86a7b0430d8cdb78070b4c55au128.swap_bytes())
        );
        assert_eq!(
            round(
                true,
                true,
                false,
                0x6353e08c0960e104cd70b751bacad0e7u128.swap_bytes(),
                0x000102030405060708090a0b0c0d0e0fu128.swap_bytes()
            ),
            Some(0x00112233445566778899aabbccddeeffu128.swap_bytes())
        );
    }
}
