//! The SM3 steps the x86 and Arm instructions compute, over four 32-bit
//! words, word 0 the low one of the register (`xmm1[31:0]`, `Vd.S[0]`).
//!
//! GB/T 32905 states the compression and the message expansion; the x86
//! steps (`vsm3msg1`, `vsm3msg2`, `vsm3rnds2`) pack them as the SDM states
//! them, the Arm ones (`sm3partw1`, `sm3ss1`, `sm3tt1a`) as the Arm
//! Architecture Reference Manual does.

/// The permutation P0 of the compression.
const fn p0(x: u32) -> u32 {
    x ^ x.rotate_left(9) ^ x.rotate_left(17)
}

/// The permutation P1 of the message expansion.
const fn p1(x: u32) -> u32 {
    x ^ x.rotate_left(15) ^ x.rotate_left(23)
}

/// The round constant `T_j` of round `j`, before its rotation by `j`.
const fn constant(round: u32) -> u32 {
    if round < 16 { 0x79cc_4519 } else { 0x7a87_9d8a }
}

/// The boolean function FF of round `j`: parity, then majority.
const fn ff(round: u32, x: u32, y: u32, z: u32) -> u32 {
    if round < 16 {
        x ^ y ^ z
    } else {
        (x & y) | (x & z) | (y & z)
    }
}

/// The boolean function GG of round `j`: parity, then choose.
const fn gg(round: u32, x: u32, y: u32, z: u32) -> u32 {
    if round < 16 {
        x ^ y ^ z
    } else {
        (x & y) | (!x & z)
    }
}

/// `vsm3msg1 xmm1, xmm2, xmm3`: the first part of the next four message
/// words, `P1(W[j-16] ^ W[j-9] ^ (W[j-3] <<< 15))`, of `xmm1` and `xmm3`
/// (`W[j-16]` and `W[j-9]`, either way about) and `xmm2`'s words 0 to 2
/// (`W[j-3]`). The fourth word's `W[j-3]` is the first word computed, which
/// `vsm3msg2` adds.
#[must_use]
pub fn sm3_message1(dst: [u32; 4], src1: [u32; 4], src2: [u32; 4]) -> [u32; 4] {
    [
        p1(dst[0] ^ src2[0] ^ src1[0].rotate_left(15)),
        p1(dst[1] ^ src2[1] ^ src1[1].rotate_left(15)),
        p1(dst[2] ^ src2[2] ^ src1[2].rotate_left(15)),
        p1(dst[3] ^ src2[3]),
    ]
}

/// `vsm3msg2 xmm1, xmm2, xmm3`: the next four message words, the first part
/// in `xmm1` with `W[j-13] <<< 7` from `xmm2` and `W[j-6]` from `xmm3`, the
/// fourth word also with the first's `W[j-3]` term.
#[must_use]
pub fn sm3_message2(dst: [u32; 4], src1: [u32; 4], src2: [u32; 4]) -> [u32; 4] {
    let first = src1[0].rotate_left(7) ^ src2[0] ^ dst[0];
    [
        first,
        src1[1].rotate_left(7) ^ src2[1] ^ dst[1],
        src1[2].rotate_left(7) ^ src2[2] ^ dst[2],
        src1[3].rotate_left(7)
            ^ src2[3]
            ^ dst[3]
            ^ first.rotate_left(6)
            ^ first.rotate_left(15)
            ^ first.rotate_left(30),
    ]
}

/// `vsm3rnds2 xmm1, xmm2, xmm3, round`: two SM3 rounds from round `round`
/// (even, 0 to 62). `xmm2` holds A, B, E and F from word 3 down, `xmm1`
/// C, D, G and H as they were two rounds before (C and D rotated right by
/// 9, G and H by 19, as A, B, E and F were then), and `xmm3` the message
/// words `W[j]`, `W[j+1]`, `W[j+4]` and `W[j+5]` from word 0 up. The result
/// is A, B, E and F after the two rounds, packed as `xmm2`.
#[must_use]
pub fn sm3_rounds2(dst: [u32; 4], src1: [u32; 4], src2: [u32; 4], round: u8) -> [u32; 4] {
    let [f, e, b, a] = src1;
    let [h, g, d, c] = dst;
    let state = [
        a,
        b,
        c.rotate_left(9),
        d.rotate_left(9),
        e,
        f,
        g.rotate_left(19),
        h.rotate_left(19),
    ];
    let [w0, w1, w4, w5] = src2;
    // The first of the two rounds is even.
    let round = u32::from(round & 0x3e);
    let state = round_of(round_of(state, round, w0, w4), round | 1, w1, w5);
    let [a, b, _, _, e, f, _, _] = state;
    [f, e, b, a]
}

/// One SM3 compression round `j` of the state A to H by the message word
/// `w` (`W[j]`) and the one four after it (`W[j+4]`, so `W'[j]` is
/// `w ^ far`).
fn round_of(state: [u32; 8], j: u32, w: u32, far: u32) -> [u32; 8] {
    let [a, b, c, d, e, f, g, h] = state;
    let ss1 = a
        .rotate_left(12)
        .wrapping_add(e)
        .wrapping_add(constant(j).rotate_left(j))
        .rotate_left(7);
    let ss2 = ss1 ^ a.rotate_left(12);
    let tt1 = ff(j, a, b, c)
        .wrapping_add(d)
        .wrapping_add(ss2)
        .wrapping_add(w ^ far);
    let tt2 = gg(j, e, f, g)
        .wrapping_add(h)
        .wrapping_add(ss1)
        .wrapping_add(w);
    [
        tt1,
        a,
        b.rotate_left(9),
        c,
        p0(tt2),
        e,
        f.rotate_left(19),
        g,
    ]
}

/// GB/T 32905's message expansion step: `W[j]` of the sixteen words before.
const fn expanded(w16: u32, w9: u32, w3: u32, w13: u32, w6: u32) -> u32 {
    p1(w16 ^ w9 ^ w3.rotate_left(15)) ^ w13.rotate_left(7) ^ w6
}

/// RISC-V's `vsm3me` of one element group: `W[16]` to `W[23]` of `W[0]` to
/// `W[7]` in `first` and `W[8]` to `W[15]` in `second`, word 0 first. Every
/// word is byte-swapped in and out: the registers hold the message's words
/// as their bytes lie in memory.
#[must_use]
pub fn riscv_sm3_expand8(first: [u32; 8], second: [u32; 8]) -> [u32; 8] {
    let mut w: Vec<u32> = first
        .into_iter()
        .chain(second)
        .map(u32::swap_bytes)
        .collect();
    for j in 16..24usize {
        let word = |back: usize| {
            j.checked_sub(back)
                .and_then(|index| w.get(index))
                .copied()
                .unwrap_or(0)
        };
        let next = expanded(word(16), word(9), word(3), word(13), word(6));
        w.push(next);
    }
    let mut words = w.into_iter().skip(16).map(u32::swap_bytes);
    std::array::from_fn(|_| words.next().unwrap_or(0))
}

/// RISC-V's `vsm3c` of one element group: rounds `2 * round` and the one
/// after (`round` 0 to 31) of the state A to H (A in word 0) by `W[j]` and
/// `W[j+1]` in `words`' words 0 and 1 and `W[j+4]` and `W[j+5]` in its words
/// 4 and 5. The result is the state after the two rounds, as the state is
/// laid out. Every word is byte-swapped in and out.
#[must_use]
pub fn riscv_sm3_compress2(state: [u32; 8], words: [u32; 8], round: u8) -> [u32; 8] {
    let state = state.map(u32::swap_bytes);
    let [w0, w1, _, _, w4, w5, _, _] = words.map(u32::swap_bytes);
    let j = u32::from(round & 0x1f).saturating_mul(2);
    round_of(round_of(state, j, w0, w4), j | 1, w1, w5).map(u32::swap_bytes)
}

/// Arm's `sm3partw1 vd, vn, vm`: the first part of the next four message
/// words, `P1(W[j-16] ^ W[j-9] ^ (W[j-3] <<< 15))` of `vd` and `vn` (either
/// way about) and `vm`'s words 1 to 3 (`W[j-3]` of the first three words);
/// the fourth word's `W[j-3]` is the first word's result.
#[must_use]
pub fn arm_sm3_partw1(vd: [u32; 4], vn: [u32; 4], vm: [u32; 4]) -> [u32; 4] {
    let [x0, x1, x2, x3] = [vd[0] ^ vn[0], vd[1] ^ vn[1], vd[2] ^ vn[2], vd[3] ^ vn[3]];
    let first = p1(x0 ^ vm[1].rotate_left(15));
    [
        first,
        p1(x1 ^ vm[2].rotate_left(15)),
        p1(x2 ^ vm[3].rotate_left(15)),
        p1(x3 ^ first.rotate_left(15)),
    ]
}

/// Arm's `sm3partw2 vd, vn, vm`: the next four message words, the first
/// part in `vd` with `W[j-6]` from `vn` and `W[j-13] <<< 7` from `vm`, the
/// fourth word also with the first's correction: `sm3partw1` read the first
/// word's first part as its `W[j-3]`.
#[must_use]
pub fn arm_sm3_partw2(vd: [u32; 4], vn: [u32; 4], vm: [u32; 4]) -> [u32; 4] {
    let tmp = [
        vn[0] ^ vm[0].rotate_left(7),
        vn[1] ^ vm[1].rotate_left(7),
        vn[2] ^ vm[2].rotate_left(7),
        vn[3] ^ vm[3].rotate_left(7),
    ];
    let correction = p1(tmp[0].rotate_left(15));
    [
        vd[0] ^ tmp[0],
        vd[1] ^ tmp[1],
        vd[2] ^ tmp[2],
        vd[3] ^ tmp[3] ^ correction,
    ]
}

/// Arm's `sm3ss1 vd, vn, vm, va`: SS1 of A (`vn`'s word 3), E (`vm`'s) and
/// the rotated round constant (`va`'s), in word 3, the others zero.
#[must_use]
pub fn arm_sm3_ss1(vn: [u32; 4], vm: [u32; 4], va: [u32; 4]) -> [u32; 4] {
    let ss1 = vn[3]
        .rotate_left(12)
        .wrapping_add(vm[3])
        .wrapping_add(va[3])
        .rotate_left(7);
    [0, 0, 0, ss1]
}

/// Arm's `sm3tt1a` and `sm3tt1b vd, vn, vm.s[i]`: the round's new A to D of
/// `vd` (D, C, B and A from word 0 up), SS1 in `vn`'s word 3 and `W'[j]`
/// `wj`, FF the parity (`sm3tt1a`, rounds 0 to 15) or the majority.
#[must_use]
pub fn arm_sm3_tt1(vd: [u32; 4], vn: [u32; 4], wj: u32, majority: bool) -> [u32; 4] {
    let [d, c, b, a] = vd;
    let ss2 = vn[3] ^ a.rotate_left(12);
    let ff = if majority {
        (a & b) | (a & c) | (b & c)
    } else {
        a ^ b ^ c
    };
    let tt1 = ff.wrapping_add(d).wrapping_add(ss2).wrapping_add(wj);
    [c, b.rotate_left(9), a, tt1]
}

/// Arm's `sm3tt2a` and `sm3tt2b vd, vn, vm.s[i]`: the round's new E to H of
/// `vd` (H, G, F and E from word 0 up), SS1 in `vn`'s word 3 and `W[j]`
/// `wj`, GG the parity (`sm3tt2a`, rounds 0 to 15) or the choose.
#[must_use]
pub fn arm_sm3_tt2(vd: [u32; 4], vn: [u32; 4], wj: u32, choose: bool) -> [u32; 4] {
    let [h, g, f, e] = vd;
    let gg = if choose {
        (e & f) | (!e & g)
    } else {
        e ^ f ^ g
    };
    let tt2 = gg.wrapping_add(h).wrapping_add(vn[3]).wrapping_add(wj);
    [g, f.rotate_left(19), e, p0(tt2)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GB/T 32905's initial value.
    const IV: [u32; 8] = [
        0x7380_166f,
        0x4914_b2b9,
        0x1724_42d7,
        0xda8a_0600,
        0xa96f_30bc,
        0x1631_38aa,
        0xe38d_ee4d,
        0xb0fb_0e4e,
    ];

    /// GB/T 32905's digest of "abc".
    const ABC: [u32; 8] = [
        0x66c7_f0f4,
        0x62ee_edd9,
        0xd1f2_d46b,
        0xdc10_e4e2,
        0x4167_c487,
        0x5cf2_f7a2,
        0x297d_a02b,
        0x8f4b_a8e0,
    ];

    /// GB/T 32905's expansion of the one padded block of "abc", `W[0]` to
    /// `W[67]`, by its recurrence.
    fn expansion() -> Vec<u32> {
        let mut w = vec![0u32; 16];
        w[0] = 0x6162_6380;
        w[15] = 0x18;
        for j in 16..68 {
            let next = p1(w[j - 16] ^ w[j - 9] ^ w[j - 3].rotate_left(15))
                ^ w[j - 13].rotate_left(7)
                ^ w[j - 6];
            w.push(next);
        }
        w
    }

    /// Four words of the expansion from `j`, word 0 the first.
    fn group(w: &[u32], j: usize) -> [u32; 4] {
        [w[j], w[j + 1], w[j + 2], w[j + 3]]
    }

    /// The x86 SM3 steps hash "abc" to GB/T 32905's digest, composed from
    /// the SDM's statements: the expansion four words at a time by
    /// `vsm3msg1` and `vsm3msg2`, the compression two rounds at a time by
    /// `vsm3rnds2` of the state as it is and as it was two rounds before.
    #[test]
    fn the_x86_sm3_steps_hash_abc_to_its_digest() {
        let w = expansion();
        for j in (16..68).step_by(4) {
            let previous = [w[j - 3], w[j - 2], w[j - 1], 0];
            let partial = sm3_message1(group(&w, j - 16), previous, group(&w, j - 9));
            let words = sm3_message2(partial, group(&w, j - 13), group(&w, j - 6));
            assert_eq!(words, group(&w, j), "x86 expansion from {j}");
        }
        let [a, b, c, d, e, f, g, h] = IV;
        let mut abef = [f, e, b, a];
        let mut before = [
            h.rotate_right(19),
            g.rotate_right(19),
            d.rotate_right(9),
            c.rotate_right(9),
        ];
        for j in (0..64).step_by(2) {
            let words = [w[j], w[j + 1], w[j + 4], w[j + 5]];
            let next = sm3_rounds2(before, abef, words, u8::try_from(j).unwrap());
            (before, abef) = (abef, next);
        }
        let [f, e, b, a] = abef;
        let [h, g, d, c] = before;
        let state = [
            a,
            b,
            c.rotate_left(9),
            d.rotate_left(9),
            e,
            f,
            g.rotate_left(19),
            h.rotate_left(19),
        ];
        let digest: Vec<u32> = state.iter().zip(IV).map(|(word, iv)| word ^ iv).collect();
        assert_eq!(digest, ABC);
    }

    /// The RISC-V SM3 steps hash "abc" to GB/T 32905's digest, from registers
    /// holding the words as their bytes lie in memory: the expansion eight
    /// words at a time by `vsm3me` of the sixteen before, the compression
    /// two rounds at a time by `vsm3c` of the whole state.
    #[test]
    fn the_riscv_sm3_steps_hash_abc_to_its_digest() {
        let w = expansion();
        let memory: Vec<u32> = w.iter().take(16).map(|word| word.swap_bytes()).collect();
        let mut words = memory;
        for j in (16..72).step_by(8) {
            let eight = |from: usize| -> [u32; 8] { words[from..from + 8].try_into().unwrap() };
            let next = riscv_sm3_expand8(eight(j - 16), eight(j - 8));
            words.extend(next);
        }
        for (j, word) in w.iter().enumerate() {
            assert_eq!(words[j].swap_bytes(), *word, "RISC-V expansion word {j}");
        }
        let mut state = IV.map(u32::swap_bytes);
        for round in 0..32u8 {
            let j = usize::from(round) * 2;
            let message = [
                words[j],
                words[j + 1],
                0,
                0,
                words[j + 4],
                words[j + 5],
                0,
                0,
            ];
            state = riscv_sm3_compress2(state, message, round);
        }
        let digest: Vec<u32> = state
            .iter()
            .zip(IV)
            .map(|(word, iv)| word.swap_bytes() ^ iv)
            .collect();
        assert_eq!(digest, ABC);
    }

    /// The Arm SM3 steps hash "abc" to GB/T 32905's digest, composed as
    /// Linux's `sm3-ce` composes them: the expansion by `sm3partw1` and
    /// `sm3partw2`; each round `sm3ss1` of A, E and the rotated constant,
    /// then `sm3tt1` of A to D (D in word 0) by `W[j] ^ W[j+4]` and `sm3tt2`
    /// of E to H by `W[j]`, the `a` forms for rounds 0 to 15.
    #[test]
    fn the_arm_sm3_steps_hash_abc_to_its_digest() {
        let w = expansion();
        for j in (16..68).step_by(4) {
            let partial = arm_sm3_partw1(group(&w, j - 16), group(&w, j - 9), group(&w, j - 4));
            let words = arm_sm3_partw2(partial, group(&w, j - 6), group(&w, j - 13));
            assert_eq!(words, group(&w, j), "Arm expansion from {j}");
        }
        let [a, b, c, d, e, f, g, h] = IV;
        let (mut dcba, mut hgfe) = ([d, c, b, a], [h, g, f, e]);
        for j in 0..64 {
            let round = u32::try_from(j).unwrap();
            let rotated = constant(round).rotate_left(round);
            let ss1 = arm_sm3_ss1(dcba, hgfe, [0, 0, 0, rotated]);
            let later = j >= 16;
            dcba = arm_sm3_tt1(dcba, ss1, w[j] ^ w[j + 4], later);
            hgfe = arm_sm3_tt2(hgfe, ss1, w[j], later);
        }
        let ([d, c, b, a], [h, g, f, e]) = (dcba, hgfe);
        let digest: Vec<u32> = [a, b, c, d, e, f, g, h]
            .iter()
            .zip(IV)
            .map(|(word, iv)| word ^ iv)
            .collect();
        assert_eq!(digest, ABC);
    }
}
