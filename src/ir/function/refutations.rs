//! Searches a pass ran over a function to a refutation, kept with the
//! function so a later run of the pass answers them without searching again.
//!
//! A search whose answer is a function of what it read alone gives the same
//! answer for as long as what it read stands. The pipeline runs a pass again
//! after any other pass changes the function, and a host normalizes one
//! function more than once, so most of those runs ask the same questions of
//! code that has not changed where the search looks. A refutation recorded
//! here names the blocks its search read and a fingerprint of them as it read
//! them ([`Refutation`]); the pass asks its question again only where those
//! blocks now read differently.
//!
//! A search spends steps, from a budget its pass shares among its searches,
//! and what a refutation stands for includes them: reused, it spends the
//! steps its search took, so the searches after it have what they would have
//! had, and the pass's results are the same with refutations or without. A
//! search that ran out of its steps settled nothing either, and is kept with
//! the steps it had: read the same, it runs out again with as many or fewer.
//!
//! Only what settles nothing is kept. A search that proves something rewrites
//! the function, so its question is not asked of the same code again, and a
//! refutation reused where it should not be costs only a rewrite the pass
//! does not make: a fingerprint that matched by chance leaves the function as
//! it is, never rewrites it wrongly.

use std::{fmt, hash::Hasher};

use rustc_hash::{FxHashMap, FxHasher};

/// How a search that settled nothing ended, with the steps it spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsettled {
    /// It refuted its question, spending this many steps.
    Refuted(usize),
    /// It ran out of the steps it had, this many.
    RanOut(usize),
}

/// How a search asked again of what it read before ends, with the steps left
/// now ([`Unsettled::again`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Again {
    /// It refutes again, leaving this many steps.
    Refuted(usize),
    /// It runs out again, leaving none.
    RanOut,
    /// It had fewer steps than are left now, and may settle: search it.
    Search,
}

impl Unsettled {
    /// How this search ends asked again of the same blocks with `steps` left:
    /// a refutation spends what it spent, or runs out where fewer are left;
    /// a search that ran out runs out again with as many or fewer.
    #[must_use]
    pub const fn again(self, steps: usize) -> Again {
        match self {
            Self::Refuted(spent) => match steps.checked_sub(spent) {
                Some(left) => Again::Refuted(left),
                None => Again::RanOut,
            },
            Self::RanOut(had) if steps > had => Again::Search,
            Self::RanOut(_) => Again::RanOut,
        }
    }
}

/// What one search that settled nothing read: the blocks, a fingerprint of
/// their content as the search read it, and how it ended ([`Unsettled`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refutation {
    /// The blocks the search read, ascending.
    read: Box<[usize]>,
    /// The fingerprint of those blocks as the search read them.
    content: u128,
    /// How it ended.
    ended: Unsettled,
}

impl Refutation {
    /// The record of a search that read `read`, found `content` there, and
    /// ended as `ended`.
    ///
    /// # Arguments
    ///
    /// * `read` - The blocks the search read, in any order.
    /// * `content` - The pass's fingerprint of those blocks.
    /// * `ended` - How it ended, with the steps it spent.
    #[must_use]
    pub fn new(read: impl IntoIterator<Item = usize>, content: u128, ended: Unsettled) -> Self {
        let mut read: Vec<usize> = read.into_iter().collect();
        read.sort_unstable();
        read.dedup();
        Self {
            read: read.into_boxed_slice(),
            content,
            ended,
        }
    }

    /// The blocks the search read, ascending.
    #[must_use]
    pub fn read(&self) -> &[usize] {
        &self.read
    }

    /// The fingerprint of the blocks as the search read them.
    #[must_use]
    pub const fn content(&self) -> u128 {
        self.content
    }

    /// How the search ended.
    #[must_use]
    pub const fn ended(&self) -> Unsettled {
        self.ended
    }
}

/// The refutations of a function's searches, by the question each answered.
///
/// A question is the pass's own fingerprint of what it asks and of whatever
/// else its answer depends on beyond the blocks it reads: the shape of the
/// control flow, the test asked about. One refutation is kept per question,
/// the latest.
#[derive(Debug, Clone, Default)]
pub struct Refutations {
    /// Each question's refutation.
    by_question: FxHashMap<u128, Refutation>,
}

impl Refutations {
    /// The refutation recorded for `question`, if any.
    #[must_use]
    pub fn get(&self, question: u128) -> Option<&Refutation> {
        self.by_question.get(&question)
    }

    /// Records `refutation` as the answer to `question`, replacing any
    /// recorded before.
    pub fn record(&mut self, question: u128, refutation: Refutation) {
        self.by_question.insert(question, refutation);
    }

    /// How many questions have a refutation.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_question.len()
    }

    /// Whether no question has a refutation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_question.is_empty()
    }
}

/// A 128-bit fingerprint of whatever is hashed into it: two differently
/// seeded `FxHash` streams over the same words.
///
/// The seeds are constants, so the fingerprint of the same content is the same
/// in every run and every process. `FxHash` is no cryptographic hash, and needs
/// not be: what a fingerprint here decides is whether work is done again, and
/// two contents that fingerprint alike only skip a rewrite (see the module
/// documentation). It is fast, and the fingerprint of a search's blocks is
/// taken every time the search's question is asked.
#[derive(Clone)]
pub struct Fingerprint {
    /// The low half's stream.
    low: FxHasher,
    /// The high half's stream.
    high: FxHasher,
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fingerprint({:032x})", self.finish128())
    }
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self::new()
    }
}

impl Fingerprint {
    /// An empty fingerprint.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            low: FxHasher::with_seed(0x243f_6a88_85a3_08d3),
            high: FxHasher::with_seed(0x1319_8a2e_0370_7344),
        }
    }

    /// The fingerprint of everything written so far.
    #[must_use]
    pub fn finish128(&self) -> u128 {
        (u128::from(self.high.finish()) << 64) | u128::from(self.low.finish())
    }
}

impl Hasher for Fingerprint {
    fn finish(&self) -> u64 {
        self.low.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.low.write(bytes);
        self.high.write(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        self.low.write_u8(value);
        self.high.write_u8(value);
    }

    fn write_u32(&mut self, value: u32) {
        self.low.write_u32(value);
        self.high.write_u32(value);
    }

    fn write_u64(&mut self, value: u64) {
        self.low.write_u64(value);
        self.high.write_u64(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.low.write_usize(value);
        self.high.write_usize(value);
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hash;

    use super::{Again, Fingerprint, Refutation, Refutations, Unsettled};
    use crate::{ir::function::SsaFunction, testing::MockTarget};

    /// The fingerprint of `values`, hashed in order.
    fn fingerprint(values: &[u64]) -> u128 {
        let mut hasher = Fingerprint::new();
        values.hash(&mut hasher);
        hasher.finish128()
    }

    #[test]
    fn the_same_content_has_the_same_fingerprint_and_other_content_another() {
        assert_eq!(fingerprint(&[1, 2, 3]), fingerprint(&[1, 2, 3]));
        assert_ne!(fingerprint(&[1, 2, 3]), fingerprint(&[1, 3, 2]));
        let both = fingerprint(&[7]);
        assert_ne!(both >> 64, both & u128::from(u64::MAX));
    }

    #[test]
    fn a_refutation_reads_its_blocks_once_each_in_order() {
        let refutation = Refutation::new([5, 2, 5, 9, 2], 11, Unsettled::Refuted(3));
        assert_eq!(refutation.read(), &[2, 5, 9]);
        assert_eq!(refutation.content(), 11);
    }

    /// **A function reads the same whatever it remembers**: its rendering,
    /// which a host compares to tell whether the function changed, leaves its
    /// refutations out.
    #[test]
    fn a_refutation_leaves_the_function_reading_as_it_did() {
        let mut function: SsaFunction<MockTarget> = SsaFunction::new(1, 2);
        let before = format!("{function:?}");
        function
            .refutations_mut()
            .record(7, Refutation::new([0, 3], 42, Unsettled::Refuted(5)));
        assert_eq!(function.refutations().len(), 1);
        assert_eq!(format!("{function:?}"), before);
    }

    /// **A reused search ends as it would have**: a refutation spends what it
    /// spent, or runs out where fewer steps are left; a search that ran out
    /// runs out again with as many or fewer, and is searched again with more.
    #[test]
    fn an_unsettled_search_ends_with_the_steps_it_would_have_left() {
        assert_eq!(Unsettled::Refuted(5).again(12), Again::Refuted(7));
        assert_eq!(Unsettled::Refuted(5).again(5), Again::Refuted(0));
        assert_eq!(Unsettled::Refuted(5).again(4), Again::RanOut);
        assert_eq!(Unsettled::RanOut(8).again(8), Again::RanOut);
        assert_eq!(Unsettled::RanOut(8).again(3), Again::RanOut);
        assert_eq!(Unsettled::RanOut(8).again(9), Again::Search);
    }

    #[test]
    fn a_question_keeps_its_latest_refutation() {
        let mut refutations = Refutations::default();
        assert!(refutations.is_empty());
        refutations.record(1, Refutation::new([0], 10, Unsettled::Refuted(1)));
        refutations.record(1, Refutation::new([1], 20, Unsettled::Refuted(1)));
        refutations.record(2, Refutation::new([2], 30, Unsettled::RanOut(9)));
        assert_eq!(refutations.len(), 2);
        assert_eq!(refutations.get(1).map(Refutation::content), Some(20));
        assert_eq!(refutations.get(3), None);
    }
}
