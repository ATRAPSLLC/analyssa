//! The SHA-1 and SHA-2 steps the x86 and Arm instructions compute, over the
//! words of their registers, word 0 the low one (`xmm1[31:0]`, `Vd.S[0]`,
//! `Vd.D[0]`).
//!
//! FIPS 180-4 states the rounds and the message schedule; the x86 steps
//! (`sha1rnds4`, `sha256rnds2`, `vsha512rnds2`) pack the working variables
//! as the SDM states them, the Arm ones (`sha1c`, `sha256h`, `sha512h`) as
//! the Arm Architecture Reference Manual does. SHA-256 and SHA-512 are one
//! algorithm over 32-bit and 64-bit [`Word`]s.

use std::ops::{BitAnd, BitXor, Not};

use crate::ir::lanes::VectorConst;

/// `f` of each pair of words of `a` and `b` at one index.
fn each(a: [u32; 4], b: [u32; 4], f: impl Fn(u32, u32) -> u32) -> [u32; 4] {
    let mut result = a;
    for (word, other) in result.iter_mut().zip(b) {
        *word = f(*word, other);
    }
    result
}

/// SHA-1's round constant and function of round group `function` (0 to 3:
/// choose, parity, majority, parity).
fn sha1_function(function: u8, b: u32, c: u32, d: u32) -> (u32, u32) {
    match function & 3 {
        0 => ((b & c) | (!b & d), 0x5a82_7999),
        1 => (b ^ c ^ d, 0x6ed9_eba1),
        2 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
        _ => (b ^ c ^ d, 0xca62_c1d6),
    }
}

/// `sha1rnds4 abcd, ew, function`: four SHA-1 rounds of group `function`.
/// `abcd` holds A in word 3 down to D in word 0; `ew` holds the first round's
/// E plus W in word 3 and the other three rounds' W below it. The result is
/// A to D after the four rounds, as `abcd` holds them.
#[must_use]
pub fn sha1_rounds4(abcd: [u32; 4], ew: [u32; 4], function: u8) -> [u32; 4] {
    let [mut d, mut c, mut b, mut a] = abcd;
    let mut e = 0u32;
    // The first round's word is the high one.
    for word in ew.into_iter().rev() {
        let (f, k) = sha1_function(function, b, c, d);
        let next = f
            .wrapping_add(a.rotate_left(5))
            .wrapping_add(word)
            .wrapping_add(e)
            .wrapping_add(k);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = next;
    }
    [d, c, b, a]
}

/// `sha1nexte state, message`: E of the next four rounds, A of `state`
/// rotated by 30 added to word 3 of `message`, its other words kept.
#[must_use]
pub fn sha1_next_e(state: [u32; 4], message: [u32; 4]) -> [u32; 4] {
    let [w0, w1, w2, w3] = message;
    [w0, w1, w2, w3.wrapping_add(state[3].rotate_left(30))]
}

/// `sha1msg1 w0to3, w4to7`: the exclusive-ors of message words four apart
/// a schedule step begins with, `W[t-16] ^ W[t-14]` for the next four words.
/// Each register holds its first word in word 3.
#[must_use]
pub fn sha1_message1(first: [u32; 4], second: [u32; 4]) -> [u32; 4] {
    let [w3, w2, w1, w0] = first;
    let [_, _, w5, w4] = second;
    [w5 ^ w3, w4 ^ w2, w3 ^ w1, w2 ^ w0]
}

/// `sha1msg2 partial, w12to15`: the next four message words, each the
/// partial sum (`sha1msg1` and the exclusive-or with `W[t-8]`) with `W[t-3]`
/// added, rotated by one.
#[must_use]
pub fn sha1_message2(partial: [u32; 4], last: [u32; 4]) -> [u32; 4] {
    let [w15, w14, w13, _] = last;
    let w16 = (partial[3] ^ w13).rotate_left(1);
    let w17 = (partial[2] ^ w14).rotate_left(1);
    let w18 = (partial[1] ^ w15).rotate_left(1);
    let w19 = (partial[0] ^ w16).rotate_left(1);
    [w19, w18, w17, w16]
}

/// A SHA-2 word: SHA-256's 32 bits or SHA-512's 64, with the rotations and
/// shifts of its Σ and σ functions, and its layout in a register.
pub trait Word:
    Copy
    + BitAnd<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Into<u128>
    + TryFrom<u128>
{
    /// The word's width.
    const BITS: u32;
    /// Σ0's three rotations.
    const BIG_SIGMA0: [u32; 3];
    /// Σ1's three rotations.
    const BIG_SIGMA1: [u32; 3];
    /// σ0's two rotations and its shift.
    const SMALL_SIGMA0: [u32; 3];
    /// σ1's two rotations and its shift.
    const SMALL_SIGMA1: [u32; 3];

    /// The word rotated right by `by`.
    #[must_use]
    fn rotate(self, by: u32) -> Self;

    /// The word shifted right by `by`, zero for a shift past its width.
    #[must_use]
    fn shift(self, by: u32) -> Self;

    /// The sum of the words, modulo the word's width.
    #[must_use]
    fn plus(self, other: Self) -> Self;

    /// Σ0 of a working variable.
    #[must_use]
    fn big_sigma0(self) -> Self {
        let [first, second, third] = Self::BIG_SIGMA0;
        self.rotate(first) ^ self.rotate(second) ^ self.rotate(third)
    }

    /// Σ1 of a working variable.
    #[must_use]
    fn big_sigma1(self) -> Self {
        let [first, second, third] = Self::BIG_SIGMA1;
        self.rotate(first) ^ self.rotate(second) ^ self.rotate(third)
    }

    /// σ0 of a message word.
    #[must_use]
    fn small_sigma0(self) -> Self {
        let [first, second, shift] = Self::SMALL_SIGMA0;
        self.rotate(first) ^ self.rotate(second) ^ self.shift(shift)
    }

    /// σ1 of a message word.
    #[must_use]
    fn small_sigma1(self) -> Self {
        let [first, second, shift] = Self::SMALL_SIGMA1;
        self.rotate(first) ^ self.rotate(second) ^ self.shift(shift)
    }

    /// The choose of the word, `y` and `z`: `y`'s bits where the word's are
    /// set, `z`'s elsewhere.
    #[must_use]
    fn choose(self, y: Self, z: Self) -> Self {
        (self & y) ^ (!self & z)
    }

    /// The bitwise majority of the word, `y` and `z`.
    #[must_use]
    fn majority(self, y: Self, z: Self) -> Self {
        (self & y) ^ (self & z) ^ (y & z)
    }

    /// The `N` words of a register, word 0 the low one, or `None` for a
    /// register of another width.
    fn words<const N: usize>(value: &VectorConst) -> Option<[Self; N]> {
        let lanes = value.relaned(Self::BITS)?;
        let words = lanes
            .lane_values()
            .map(|lane| Self::try_from(lane).ok())
            .collect::<Option<Vec<Self>>>()?;
        words.try_into().ok()
    }

    /// The register of `words`, word 0 the low one.
    fn packed<const N: usize>(words: [Self; N]) -> Option<VectorConst> {
        VectorConst::from_lanes(Self::BITS, words.map(Into::into))
    }
}

impl Word for u32 {
    const BITS: u32 = 32;
    const BIG_SIGMA0: [u32; 3] = [2, 13, 22];
    const BIG_SIGMA1: [u32; 3] = [6, 11, 25];
    const SMALL_SIGMA0: [u32; 3] = [7, 18, 3];
    const SMALL_SIGMA1: [u32; 3] = [17, 19, 10];

    fn rotate(self, by: u32) -> Self {
        self.rotate_right(by)
    }

    fn shift(self, by: u32) -> Self {
        self.checked_shr(by).unwrap_or(0)
    }

    fn plus(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Word for u64 {
    const BITS: u32 = 64;
    const BIG_SIGMA0: [u32; 3] = [28, 34, 39];
    const BIG_SIGMA1: [u32; 3] = [14, 18, 41];
    const SMALL_SIGMA0: [u32; 3] = [1, 8, 7];
    const SMALL_SIGMA1: [u32; 3] = [19, 61, 6];

    fn rotate(self, by: u32) -> Self {
        self.rotate_right(by)
    }

    fn shift(self, by: u32) -> Self {
        self.checked_shr(by).unwrap_or(0)
    }

    fn plus(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

/// One SHA-2 round of the working variables `[a, b, c, d, e, f, g, h]` by
/// the message word plus round constant `wk`.
fn sha2_round<W: Word>(state: [W; 8], wk: W) -> [W; 8] {
    let [a, b, c, d, e, f, g, h] = state;
    let t1 = h.plus(e.big_sigma1()).plus(e.choose(f, g)).plus(wk);
    let t2 = a.big_sigma0().plus(a.majority(b, c));
    [t1.plus(t2), a, b, c, d.plus(t1), e, f, g]
}

/// `sha256rnds2 cdgh, abef, wk` and `vsha512rnds2`: two SHA-2 rounds by the
/// message words plus round constants `wk`, the first round's word 0.
/// `cdgh` holds C, D, G and H from word 3 down, `abef` A, B, E and F; the
/// result is A, B, E and F after the two rounds, packed as `abef`.
#[must_use]
pub fn sha2_rounds2<W: Word>(cdgh: [W; 4], abef: [W; 4], wk: [W; 2]) -> [W; 4] {
    let [h, g, d, c] = cdgh;
    let [f, e, b, a] = abef;
    let state = sha2_round(sha2_round([a, b, c, d, e, f, g, h], wk[0]), wk[1]);
    let [a, b, _, _, e, f, _, _] = state;
    [f, e, b, a]
}

/// `sha256msg1 w0to3, w4to7` and `vsha512msg1`: `W[t-16] + σ0(W[t-15])`
/// for the next four message words, word 0 of `first` its first word and
/// `next` the word after `first`'s.
#[must_use]
pub fn sha2_message1<W: Word>(first: [W; 4], next: W) -> [W; 4] {
    let [w0, w1, w2, w3] = first;
    [
        w0.plus(w1.small_sigma0()),
        w1.plus(w2.small_sigma0()),
        w2.plus(w3.small_sigma0()),
        w3.plus(next.small_sigma0()),
    ]
}

/// `sha256msg2 partial, w12to15` and `vsha512msg2`: the next four message
/// words, each the partial sum (the first step's and `W[t-7]` added) plus
/// `σ1(W[t-2])`.
#[must_use]
pub fn sha2_message2<W: Word>(partial: [W; 4], last: [W; 4]) -> [W; 4] {
    let (w14, w15) = (last[2], last[3]);
    let w16 = partial[0].plus(w14.small_sigma1());
    let w17 = partial[1].plus(w15.small_sigma1());
    let w18 = partial[2].plus(w16.small_sigma1());
    let w19 = partial[3].plus(w17.small_sigma1());
    [w16, w17, w18, w19]
}

/// RISC-V's `vsha2ms` of one element group: the next four message words of
/// `W[0]` to `W[3]` in `first`, `W[4]`, `W[9]`, `W[10]` and `W[11]` in
/// `second` and `W[12]` to `W[15]` in `third`, each word 0 first. It is
/// x86's two steps with the words seven back added between them.
#[must_use]
pub fn sha2_schedule4<W: Word>(first: [W; 4], second: [W; 4], third: [W; 4]) -> [W; 4] {
    let [w4, w9, w10, w11] = second;
    let [p0, p1, p2, p3] = sha2_message1(first, w4);
    let partial = [p0.plus(w9), p1.plus(w10), p2.plus(w11), p3.plus(third[0])];
    sha2_message2(partial, third)
}

/// Arm's SHA-1 hash update (`sha1c`, `sha1p`, `sha1m` by `function` 0, 1
/// and 2): four rounds of the hash `abcd` (A in word 0) and E `e` by the
/// message words `w`, word 0 first. The result is the new `abcd`.
#[must_use]
pub fn arm_sha1_hash(abcd: [u32; 4], e: u32, w: [u32; 4], function: u8) -> [u32; 4] {
    // Choose, parity and majority, with SHA-1's group constants left to the
    // message words, which Arm adds before the instruction.
    let function = |b: u32, c: u32, d: u32| match function {
        0 => (b & c) | (!b & d),
        2 => (b & c) | (b & d) | (c & d),
        _ => b ^ c ^ d,
    };
    let [mut a, mut b, mut c, mut d] = abcd;
    let mut y = e;
    for word in w {
        y = y
            .wrapping_add(a.rotate_left(5))
            .wrapping_add(function(b, c, d))
            .wrapping_add(word);
        // The rotate of `Y : X` by a word: X's words move up, Y becomes the
        // new A and the old D the new Y.
        b = b.rotate_left(30);
        let old_d = d;
        d = c;
        c = b;
        b = a;
        a = y;
        y = old_d;
    }
    [a, b, c, d]
}

/// Arm's `sha1h`: the fixed rotate by 30.
#[must_use]
pub const fn arm_sha1_fixed_rotate(word: u32) -> u32 {
    word.rotate_left(30)
}

/// Arm's `sha1su0 vd, vn, vm`: the high half of `vd` and the low half of
/// `vn`, as the next message words' `W[t-14]`, with `vd` and `vm` added by
/// exclusive-or.
#[must_use]
pub fn arm_sha1_schedule0(vd: [u32; 4], vn: [u32; 4], vm: [u32; 4]) -> [u32; 4] {
    let joined = [vd[2], vd[3], vn[0], vn[1]];
    each(each(joined, vd, |x, y| x ^ y), vm, |x, y| x ^ y)
}

/// Arm's `sha1su1 vd, vn`: the next four message words of the partial sum
/// `vd` and `W[t-3]` in `vn`'s words 1 to 3, each rotated by one, the last
/// also by the first's.
#[must_use]
pub fn arm_sha1_schedule1(vd: [u32; 4], vn: [u32; 4]) -> [u32; 4] {
    let shifted = [vn[1], vn[2], vn[3], 0];
    let t = each(vd, shifted, |x, y| x ^ y);
    [
        t[0].rotate_left(1),
        t[1].rotate_left(1),
        t[2].rotate_left(1),
        t[3].rotate_left(1) ^ t[0].rotate_left(2),
    ]
}

/// Arm's SHA-256 hash update (`sha256h` for `part1`, `sha256h2` otherwise):
/// four rounds of the halves `x` (A to D, A in word 0) and `y` (E to H) by
/// the message words plus constants `w`. `sha256h` reads `[abcd, efgh, w]`
/// and returns the new A to D, `sha256h2` reads `[efgh, abcd, w]`, `abcd`
/// before the rounds, and returns the new E to H.
#[must_use]
pub fn arm_sha256_hash(first: [u32; 4], second: [u32; 4], w: [u32; 4], part1: bool) -> [u32; 4] {
    let (abcd, efgh) = if part1 {
        (first, second)
    } else {
        (second, first)
    };
    let [a, b, c, d] = abcd;
    let [e, f, g, h] = efgh;
    let mut state = [a, b, c, d, e, f, g, h];
    for word in w {
        state = sha2_round(state, word);
    }
    let [a, b, c, d, e, f, g, h] = state;
    if part1 { [a, b, c, d] } else { [e, f, g, h] }
}

/// Arm's `sha256su0 vd, vn`: `W[t-16] + σ0(W[t-15])` for the next four
/// message words, `vd` holding `W[t-16]` and `vn` the word after it.
#[must_use]
pub fn arm_sha256_schedule0(vd: [u32; 4], vn: [u32; 4]) -> [u32; 4] {
    let next = [vd[1], vd[2], vd[3], vn[0]];
    each(vd, next, |x, y| x.plus(y.small_sigma0()))
}

/// Arm's `sha256su1 vd, vn, vm`: the next four message words of the partial
/// sum `vd`, `W[t-7]` from `vn` and `vm`, and `σ1(W[t-2])` from `vm`'s high
/// half, then from the first two words computed.
#[must_use]
pub fn arm_sha256_schedule1(vd: [u32; 4], vn: [u32; 4], vm: [u32; 4]) -> [u32; 4] {
    let seven = [vn[1], vn[2], vn[3], vm[0]];
    let w16 = vd[0].plus(seven[0]).plus(vm[2].small_sigma1());
    let w17 = vd[1].plus(seven[1]).plus(vm[3].small_sigma1());
    let w18 = vd[2].plus(seven[2]).plus(w16.small_sigma1());
    let w19 = vd[3].plus(seven[3]).plus(w17.small_sigma1());
    [w16, w17, w18, w19]
}

/// Arm's `sha512h qd, qn, vm.2d`: the first halves of two SHA-512 rounds,
/// each round's `T1` without D. `vm` holds D and E (D in word 0), `qn` F and
/// G, and `qd` each round's H plus its message word and constant, the first
/// round's in word 1. The result is the first round's `T1` in word 1 and
/// the second's in word 0.
#[must_use]
pub fn arm_sha512_hash(qd: [u64; 2], qn: [u64; 2], vm: [u64; 2]) -> [u64; 2] {
    let ([wk1, wk0], [f, g], [d, e]) = (qd, qn, vm);
    let first = e.choose(f, g).plus(e.big_sigma1()).plus(wk0);
    // The first round's E, which the second round's T1 reads.
    let next_e = first.plus(d);
    let second = next_e.choose(e, f).plus(next_e.big_sigma1()).plus(wk1);
    [second, first]
}

/// Arm's `sha512h2 qd, qn, vm.2d`: the second halves of two SHA-512 rounds.
/// `vm` holds A and B (A in word 0), `qn` C (in word 0), and `qd` the rounds'
/// `T1` as [`arm_sha512_hash`] returns them. The result is the new A in word
/// 0 and B in word 1.
#[must_use]
pub fn arm_sha512_hash2(qd: [u64; 2], qn: [u64; 2], vm: [u64; 2]) -> [u64; 2] {
    let ([second, first], c, [a, b]) = (qd, qn[0], vm);
    let next_a = c.majority(b, a).plus(a.big_sigma0()).plus(first);
    let last_a = next_a.majority(a, b).plus(next_a.big_sigma0()).plus(second);
    [last_a, next_a]
}

/// Arm's `sha512su0 vd.2d, vn.2d`: `W[t-16] + σ0(W[t-15])` for the next two
/// message words, `vd` holding `W[t-16]` and `vn` the word after `vd`'s.
#[must_use]
pub fn arm_sha512_schedule0(vd: [u64; 2], vn: [u64; 2]) -> [u64; 2] {
    let [w0, w1] = vd;
    [w0.plus(w1.small_sigma0()), w1.plus(vn[0].small_sigma0())]
}

/// Arm's `sha512su1 vd.2d, vn.2d, vm.2d`: the next two message words of the
/// partial sum `vd`, `σ1(W[t-2])` of `vn`'s words and `W[t-7]` in `vm`.
#[must_use]
pub fn arm_sha512_schedule1(vd: [u64; 2], vn: [u64; 2], vm: [u64; 2]) -> [u64; 2] {
    [
        vd[0].plus(vn[0].small_sigma1()).plus(vm[0]),
        vd[1].plus(vn[1].small_sigma1()).plus(vm[1]),
    ]
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use super::*;

    /// FIPS 180-4's SHA-512 round constants. SHA-256's are the high 32 bits
    /// of the first 64.
    const K512: [u64; 80] = [
        0x428a_2f98_d728_ae22,
        0x7137_4491_23ef_65cd,
        0xb5c0_fbcf_ec4d_3b2f,
        0xe9b5_dba5_8189_dbbc,
        0x3956_c25b_f348_b538,
        0x59f1_11f1_b605_d019,
        0x923f_82a4_af19_4f9b,
        0xab1c_5ed5_da6d_8118,
        0xd807_aa98_a303_0242,
        0x1283_5b01_4570_6fbe,
        0x2431_85be_4ee4_b28c,
        0x550c_7dc3_d5ff_b4e2,
        0x72be_5d74_f27b_896f,
        0x80de_b1fe_3b16_96b1,
        0x9bdc_06a7_25c7_1235,
        0xc19b_f174_cf69_2694,
        0xe49b_69c1_9ef1_4ad2,
        0xefbe_4786_384f_25e3,
        0x0fc1_9dc6_8b8c_d5b5,
        0x240c_a1cc_77ac_9c65,
        0x2de9_2c6f_592b_0275,
        0x4a74_84aa_6ea6_e483,
        0x5cb0_a9dc_bd41_fbd4,
        0x76f9_88da_8311_53b5,
        0x983e_5152_ee66_dfab,
        0xa831_c66d_2db4_3210,
        0xb003_27c8_98fb_213f,
        0xbf59_7fc7_beef_0ee4,
        0xc6e0_0bf3_3da8_8fc2,
        0xd5a7_9147_930a_a725,
        0x06ca_6351_e003_826f,
        0x1429_2967_0a0e_6e70,
        0x27b7_0a85_46d2_2ffc,
        0x2e1b_2138_5c26_c926,
        0x4d2c_6dfc_5ac4_2aed,
        0x5338_0d13_9d95_b3df,
        0x650a_7354_8baf_63de,
        0x766a_0abb_3c77_b2a8,
        0x81c2_c92e_47ed_aee6,
        0x9272_2c85_1482_353b,
        0xa2bf_e8a1_4cf1_0364,
        0xa81a_664b_bc42_3001,
        0xc24b_8b70_d0f8_9791,
        0xc76c_51a3_0654_be30,
        0xd192_e819_d6ef_5218,
        0xd699_0624_5565_a910,
        0xf40e_3585_5771_202a,
        0x106a_a070_32bb_d1b8,
        0x19a4_c116_b8d2_d0c8,
        0x1e37_6c08_5141_ab53,
        0x2748_774c_df8e_eb99,
        0x34b0_bcb5_e19b_48a8,
        0x391c_0cb3_c5c9_5a63,
        0x4ed8_aa4a_e341_8acb,
        0x5b9c_ca4f_7763_e373,
        0x682e_6ff3_d6b2_b8a3,
        0x748f_82ee_5def_b2fc,
        0x78a5_636f_4317_2f60,
        0x84c8_7814_a1f0_ab72,
        0x8cc7_0208_1a64_39ec,
        0x90be_fffa_2363_1e28,
        0xa450_6ceb_de82_bde9,
        0xbef9_a3f7_b2c6_7915,
        0xc671_78f2_e372_532b,
        0xca27_3ece_ea26_619c,
        0xd186_b8c7_21c0_c207,
        0xeada_7dd6_cde0_eb1e,
        0xf57d_4f7f_ee6e_d178,
        0x06f0_67aa_7217_6fba,
        0x0a63_7dc5_a2c8_98a6,
        0x113f_9804_bef9_0dae,
        0x1b71_0b35_131c_471b,
        0x28db_77f5_2304_7d84,
        0x32ca_ab7b_40c7_2493,
        0x3c9e_be0a_15c9_bebc,
        0x431d_67c4_9c10_0d4c,
        0x4cc5_d4be_cb3e_42b6,
        0x597f_299c_fc65_7e2a,
        0x5fcb_6fab_3ad6_faec,
        0x6c44_198c_4a47_5817,
    ];

    /// FIPS 180-4's SHA-512 initial hash. SHA-256's is its high 32 bits.
    const H512: [u64; 8] = [
        0x6a09_e667_f3bc_c908,
        0xbb67_ae85_84ca_a73b,
        0x3c6e_f372_fe94_f82b,
        0xa54f_f53a_5f1d_36f1,
        0x510e_527f_ade6_82d1,
        0x9b05_688c_2b3e_6c1f,
        0x1f83_d9ab_fb41_bd6b,
        0x5be0_cd19_137e_2179,
    ];

    /// FIPS 180-4's SHA-1 initial hash and round constants.
    const H1: [u32; 5] = [
        0x6745_2301,
        0xefcd_ab89,
        0x98ba_dcfe,
        0x1032_5476,
        0xc3d2_e1f0,
    ];
    const K1: [u32; 4] = [0x5a82_7999, 0x6ed9_eba1, 0x8f1b_bcdc, 0xca62_c1d6];

    /// The digests of "abc".
    const SHA1_ABC: [u32; 5] = [
        0xa999_3e36,
        0x4706_816a,
        0xba3e_2571,
        0x7850_c26c,
        0x9cd0_d89d,
    ];
    const SHA256_ABC: [u32; 8] = [
        0xba78_16bf,
        0x8f01_cfea,
        0x4141_40de,
        0x5dae_2223,
        0xb003_61a3,
        0x9617_7a9c,
        0xb410_ff61,
        0xf200_15ad,
    ];
    const SHA512_ABC: [u64; 8] = [
        0xddaf_35a1_9361_7aba,
        0xcc41_7349_ae20_4131,
        0x12e6_fa4e_89a9_7ea2,
        0x0a9e_eee6_4b55_d39a,
        0x2192_992a_274f_c1a8,
        0x36ba_3c23_a3fe_ebbd,
        0x454d_4423_643c_e80e,
        0x2a9a_c94f_a54c_a49f,
    ];

    /// The words of "abc" and its end bit, as SHA-256 and SHA-512 read them.
    const ABC256: u32 = 0x6162_6380;
    const ABC512: u64 = 0x6162_6380_0000_0000;

    /// SHA-256's round constants.
    fn k256() -> Vec<u32> {
        K512.iter()
            .take(64)
            .map(|k| u32::try_from(k >> 32).unwrap())
            .collect()
    }

    /// SHA-256's initial hash.
    fn h256() -> [u32; 8] {
        H512.map(|h| u32::try_from(h >> 32).unwrap())
    }

    /// The one padded block of "abc": `first` the word of its bytes and end
    /// bit, and its length of 24 bits.
    fn abc_block<W: Word + From<u8>>(first: W) -> Vec<W> {
        let mut block = vec![W::from(0); 16];
        block[0] = first;
        block[15] = W::from(0x18);
        block
    }

    /// FIPS 180-4's SHA-1 message schedule of "abc", by its recurrence.
    fn sha1_schedule() -> Vec<u32> {
        let mut w = abc_block(ABC256);
        for t in 16..80 {
            w.push((w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]).rotate_left(1));
        }
        w
    }

    /// FIPS 180-4's SHA-2 message schedule of `block` for `rounds` rounds,
    /// by its recurrence.
    fn sha2_schedule<W: Word>(block: Vec<W>, rounds: usize) -> Vec<W> {
        let mut w = block;
        for t in 16..rounds {
            let next = w[t - 2]
                .small_sigma1()
                .plus(w[t - 7])
                .plus(w[t - 15].small_sigma0())
                .plus(w[t - 16]);
            w.push(next);
        }
        w
    }

    /// Four words of the schedule from `t`, word 0 the first.
    fn group<W: Copy>(w: &[W], t: usize) -> [W; 4] {
        [w[t], w[t + 1], w[t + 2], w[t + 3]]
    }

    fn add<W: Word>(a: [W; 4], b: [W; 4]) -> [W; 4] {
        std::array::from_fn(|index| a[index].plus(b[index]))
    }

    /// The x86 SHA-2 steps' digest of `block` by the round constants `k` and
    /// the initial hash `h`, composed as Intel's reference code composes
    /// them: the state packed `ABEF` and `CDGH`, two rounds at a time by the
    /// low and then the high message words, and the schedule by the first
    /// message step, the words seven back and the second step.
    fn x86_sha2_digest<W: Word + PartialEq + Debug>(block: Vec<W>, k: &[W], h: [W; 8]) -> [W; 8] {
        let w = sha2_schedule(block, k.len());
        let mut messages: Vec<[W; 4]> = (0..4).map(|g| group(&w, 4 * g)).collect();
        for g in 4..k.len() / 4 {
            let (m0, m1, m2, m3) = (
                messages[g - 4],
                messages[g - 3],
                messages[g - 2],
                messages[g - 1],
            );
            let seven_back = [m2[1], m2[2], m2[3], m3[0]];
            messages.push(sha2_message2(add(sha2_message1(m0, m1[0]), seven_back), m3));
            assert_eq!(messages[g], group(&w, 4 * g), "x86 schedule group {g}");
        }
        let [a, b, c, d, e, f, g, hh] = h;
        let (mut abef, mut cdgh) = ([f, e, b, a], [hh, g, d, c]);
        for (index, message) in messages.iter().enumerate() {
            let wk = add(*message, group(k, 4 * index));
            let low = sha2_rounds2(cdgh, abef, [wk[0], wk[1]]);
            let high = sha2_rounds2(abef, low, [wk[2], wk[3]]);
            (cdgh, abef) = (low, high);
        }
        let [f, e, b, a] = add(abef, [h[5], h[4], h[1], h[0]]);
        let [hh, g, d, c] = add(cdgh, [h[7], h[6], h[3], h[2]]);
        [a, b, c, d, e, f, g, hh]
    }

    /// RISC-V's message schedule step of each four words before, the words
    /// `W[t-12]`, `W[t-7]` to `W[t-5]` and the last four, gives FIPS 180-4's
    /// schedule of "abc" for SHA-256 and SHA-512. Its compression steps are
    /// x86's two-round step by the low or the high pair of words.
    fn riscv_schedule<W: Word + PartialEq + Debug>(block: Vec<W>, rounds: usize) {
        let w = sha2_schedule(block, rounds);
        let mut messages: Vec<[W; 4]> = (0..4).map(|g| group(&w, 4 * g)).collect();
        for g in 4..rounds / 4 {
            let (m0, m1, m2, m3) = (
                messages[g - 4],
                messages[g - 3],
                messages[g - 2],
                messages[g - 1],
            );
            messages.push(sha2_schedule4(m0, [m1[0], m2[1], m2[2], m2[3]], m3));
            assert_eq!(messages[g], group(&w, 4 * g), "RISC-V schedule group {g}");
        }
    }

    #[test]
    fn the_riscv_schedule_steps_expand_abc() {
        riscv_schedule(abc_block(ABC256), 64);
        riscv_schedule(abc_block(ABC512), 80);
    }

    /// The x86 SHA-256 steps hash "abc" to FIPS 180-4's digest.
    #[test]
    fn the_x86_sha256_steps_hash_abc_to_its_digest() {
        assert_eq!(
            x86_sha2_digest(abc_block(ABC256), &k256(), h256()),
            SHA256_ABC
        );
    }

    /// The x86 SHA-512 steps hash "abc" to FIPS 180-4's digest.
    #[test]
    fn the_x86_sha512_steps_hash_abc_to_its_digest() {
        assert_eq!(x86_sha2_digest(abc_block(ABC512), &K512, H512), SHA512_ABC);
    }

    /// The Arm SHA-256 steps hash "abc" to FIPS 180-4's digest, composed as
    /// the Arm reference code composes them: `sha256h` and `sha256h2` of the
    /// halves before the rounds, and the schedule by `sha256su0` and
    /// `sha256su1`.
    #[test]
    fn the_arm_sha256_steps_hash_abc_to_its_digest() {
        let (k, h) = (k256(), h256());
        let w = sha2_schedule(abc_block(ABC256), 64);
        let mut messages: Vec<[u32; 4]> = (0..4).map(|g| group(&w, 4 * g)).collect();
        for g in 4..16 {
            let (m0, m1, m2, m3) = (
                messages[g - 4],
                messages[g - 3],
                messages[g - 2],
                messages[g - 1],
            );
            messages.push(arm_sha256_schedule1(arm_sha256_schedule0(m0, m1), m2, m3));
            assert_eq!(messages[g], group(&w, 4 * g), "Arm schedule group {g}");
        }
        let (mut abcd, mut efgh) = (group(&h, 0), group(&h, 4));
        for (index, message) in messages.iter().enumerate() {
            let wk = add(*message, group(&k, 4 * index));
            let before = abcd;
            abcd = arm_sha256_hash(abcd, efgh, wk, true);
            efgh = arm_sha256_hash(efgh, before, wk, false);
        }
        let [a, b, c, d] = add(abcd, group(&h, 0));
        let [e, f, g, hh] = add(efgh, group(&h, 4));
        assert_eq!([a, b, c, d, e, f, g, hh], SHA256_ABC);
    }

    /// The Arm SHA-512 steps hash "abc" to FIPS 180-4's digest, composed as
    /// Linux's `sha512-ce` composes them: the state in pairs, the earlier
    /// variable in word 0; each round's message word and constant added to
    /// its H, the first round's in word 1; `sha512h` of F and G and of D and
    /// E, D plus its results the new E and F, and `sha512h2` of its results,
    /// C and D, and A and B; the schedule by `sha512su0` and `sha512su1` of
    /// the words seven back.
    #[test]
    fn the_arm_sha512_steps_hash_abc_to_its_digest() {
        let w = sha2_schedule(abc_block(ABC512), 80);
        let pair = |t: usize| [w[t], w[t + 1]];
        let mut messages: Vec<[u64; 2]> = (0..8).map(|g| pair(2 * g)).collect();
        for g in 8..40 {
            let seven_back = [messages[g - 4][1], messages[g - 3][0]];
            let partial = arm_sha512_schedule0(messages[g - 8], messages[g - 7]);
            messages.push(arm_sha512_schedule1(partial, messages[g - 1], seven_back));
            assert_eq!(messages[g], pair(2 * g), "Arm schedule pair {g}");
        }
        let pair = |t: usize| [H512[t], H512[t + 1]];
        let (mut ab, mut cd, mut ef, mut gh) = (pair(0), pair(2), pair(4), pair(6));
        for (index, message) in messages.iter().enumerate() {
            let (k0, k1) = (K512[2 * index], K512[2 * index + 1]);
            let wk = [
                gh[0].plus(message[1]).plus(k1),
                gh[1].plus(message[0]).plus(k0),
            ];
            let t1 = arm_sha512_hash(wk, [ef[1], gh[0]], [cd[1], ef[0]]);
            let next_ef = [cd[0].plus(t1[0]), cd[1].plus(t1[1])];
            let next_ab = arm_sha512_hash2(t1, cd, ab);
            (ab, cd, ef, gh) = (next_ab, ab, next_ef, ef);
        }
        let state = [ab, cd, ef, gh].concat();
        let digest: Vec<u64> = state
            .iter()
            .zip(H512)
            .map(|(word, h)| word.plus(h))
            .collect();
        assert_eq!(digest, SHA512_ABC);
    }

    /// The x86 SHA-1 steps hash "abc" to FIPS 180-4's digest, composed as
    /// Intel's reference code composes them: `sha1nexte` of the previous
    /// state to each group's message, `sha1rnds4` by the group's function,
    /// and the schedule by `sha1msg1`, the exclusive-or with the words eight
    /// back and `sha1msg2`. Each register holds its first word in word 3.
    #[test]
    fn the_x86_sha1_steps_hash_abc_to_its_digest() {
        let w = sha1_schedule();
        let reversed = |t: usize| {
            let [w0, w1, w2, w3] = group(&w, t);
            [w3, w2, w1, w0]
        };
        let mut messages: Vec<[u32; 4]> = (0..4).map(|g| reversed(4 * g)).collect();
        for g in 4..20 {
            let (m0, m1, m2, m3) = (
                messages[g - 4],
                messages[g - 3],
                messages[g - 2],
                messages[g - 1],
            );
            let partial = each(sha1_message1(m0, m1), m2, |x, y| x ^ y);
            messages.push(sha1_message2(partial, m3));
            assert_eq!(messages[g], reversed(4 * g), "x86 schedule group {g}");
        }
        let [a, b, c, d, e] = H1;
        let mut abcd = [d, c, b, a];
        let mut previous = abcd;
        let mut first = messages[0];
        first[3] = first[3].wrapping_add(e);
        abcd = sha1_rounds4(abcd, first, 0);
        for (index, message) in messages.iter().enumerate().skip(1) {
            let ew = sha1_next_e(previous, *message);
            previous = abcd;
            abcd = sha1_rounds4(abcd, ew, u8::try_from(index / 5).unwrap());
        }
        let e = sha1_next_e(previous, [0, 0, 0, e])[3];
        let [d, c, b, a] = add(abcd, [H1[3], H1[2], H1[1], H1[0]]);
        assert_eq!([a, b, c, d, e], SHA1_ABC);
    }

    /// The Arm SHA-1 steps hash "abc" to FIPS 180-4's digest, composed as the
    /// Arm reference code composes them: each group's constant added to its
    /// message words, `sha1h` of A for the next group's E, `sha1c`, `sha1p`
    /// and `sha1m` by the group, and the schedule by `sha1su0` and `sha1su1`.
    #[test]
    fn the_arm_sha1_steps_hash_abc_to_its_digest() {
        let w = sha1_schedule();
        let mut messages: Vec<[u32; 4]> = (0..4).map(|g| group(&w, 4 * g)).collect();
        for g in 4..20 {
            let (m0, m1, m2, m3) = (
                messages[g - 4],
                messages[g - 3],
                messages[g - 2],
                messages[g - 1],
            );
            messages.push(arm_sha1_schedule1(arm_sha1_schedule0(m0, m1, m2), m3));
            assert_eq!(messages[g], group(&w, 4 * g), "Arm schedule group {g}");
        }
        let mut abcd = [H1[0], H1[1], H1[2], H1[3]];
        let mut e = H1[4];
        for (index, message) in messages.iter().enumerate() {
            let wk = message.map(|word| word.wrapping_add(K1[index / 5]));
            let next = arm_sha1_fixed_rotate(abcd[0]);
            let function = match index / 5 {
                0 => 0,
                2 => 2,
                _ => 1,
            };
            abcd = arm_sha1_hash(abcd, e, wk, function);
            e = next;
        }
        let [a, b, c, d] = add(abcd, [H1[0], H1[1], H1[2], H1[3]]);
        assert_eq!([a, b, c, d, e.wrapping_add(H1[4])], SHA1_ABC);
    }
}
