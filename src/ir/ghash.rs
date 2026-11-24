//! The GHASH multiply of GCM (NIST SP 800-38D) as RISC-V's `vgmul` and
//! `vghsh` compute it, of 128-bit blocks holding their bytes in memory
//! order, byte 0 the low one.
//!
//! GCM numbers a block's bits from the most significant bit of its first
//! byte. Reversing the bits of every byte (`brev8`) turns that order into
//! the integer's own, where the multiply is a shift-and-add reduced by
//! `x^128 + x^7 + x^2 + x + 1`.

/// `block` with the bits of each byte reversed.
fn reflected(block: u128) -> u128 {
    u128::from_le_bytes(block.to_le_bytes().map(u8::reverse_bits))
}

/// `vgmul` of one element group: the product of `multiplier` and the hash
/// subkey `subkey` in GCM's field.
#[must_use]
pub fn multiply(multiplier: u128, subkey: u128) -> u128 {
    let multiplier = reflected(multiplier);
    let mut power = reflected(subkey);
    let mut product = 0u128;
    for bit in 0..128 {
        if multiplier >> bit & 1 == 1 {
            product ^= power;
        }
        let reduce = power >> 127 == 1;
        power <<= 1;
        if reduce {
            power ^= 0x87;
        }
    }
    reflected(product)
}

/// `vghsh` of one element group: the next partial hash of the partial hash
/// `hash` and the block `block` by the hash subkey `subkey`,
/// `(hash ^ block) * subkey`.
#[must_use]
pub fn multiply_add(hash: u128, block: u128, subkey: u128) -> u128 {
    multiply(hash ^ block, subkey)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block of its bytes in order.
    fn block(hex: &str) -> u128 {
        u128::from_str_radix(hex, 16).unwrap().swap_bytes()
    }

    /// GCM's test case 2 (the zero key, IV and plaintext block) hashes its
    /// ciphertext to the value its tag holds: the hash subkey
    /// `66e94bd4ef8a2c3b884cfa59ca342b2e`, the ciphertext
    /// `0388dace60b6a392f328c2b971b2fe78` added and multiplied, then the
    /// lengths block. The values are SP 800-38D's own algorithm's, with
    /// openssl's AES, which give the test case's tag.
    #[test]
    fn the_ghash_steps_hash_gcm_test_case_2() {
        let subkey = block("66e94bd4ef8a2c3b884cfa59ca342b2e");
        let cipher = block("0388dace60b6a392f328c2b971b2fe78");
        let first = block("5e2ec746917062882c85b0685353deb7");
        assert_eq!(multiply_add(0, cipher, subkey), first);
        assert_eq!(multiply(cipher, subkey), first);
        let lengths = block("00000000000000000000000000000080");
        assert_eq!(
            multiply_add(first, lengths, subkey),
            block("f38cbb1ad69223dcc3457ae5b6b0f885")
        );
    }
}
