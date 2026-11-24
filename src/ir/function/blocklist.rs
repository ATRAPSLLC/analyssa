//! The blocks of a function, and the one question about them worth remembering.
//!
//! [`SsaFunction::get_definition`] answers from the definition site a variable
//! records, and falls back to scanning every instruction when the record is
//! stale. A variable no instruction defines - an incoming argument, a value the
//! function is entered with - has nothing to find, so each question about one
//! scanned the whole function to return nothing, and an analysis that decodes
//! an address back to such a root asks once per memory access.
//!
//! [`BlockList`] answers "does any instruction define this variable" from a set
//! built by one pass over the instructions and kept until the blocks are next
//! borrowed mutably. Every write to a block goes through that borrow, so the set
//! cannot be stale when it is read.
//!
//! [`SsaFunction::get_definition`]: crate::ir::function::SsaFunction::get_definition

use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::OnceLock,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    ir::{block::SsaBlock, variable::SsaVarId, varstore::VarSet},
    target::Target,
};

/// The blocks of an [`SsaFunction`](crate::ir::function::SsaFunction), indexed by block id.
///
/// Reads like a `Vec<SsaBlock<T>>`. Any mutable borrow of it goes through
/// [`DerefMut`] and forgets what [`Self::defines_variable`] has learned, so the
/// answer always describes the blocks as they stand.
pub struct BlockList<T: Target> {
    /// The blocks.
    blocks: Vec<SsaBlock<T>>,
    /// The variables some instruction defines, once a question has needed it.
    instruction_defined: OnceLock<VarSet>,
}

impl<T: Target> BlockList<T> {
    /// Creates a list with no blocks.
    pub const fn new() -> Self {
        Self::from_blocks(Vec::new())
    }

    /// Creates an empty list with room for `capacity` blocks.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_blocks(Vec::with_capacity(capacity))
    }

    /// Wraps `blocks`, with nothing yet learned about them.
    const fn from_blocks(blocks: Vec<SsaBlock<T>>) -> Self {
        Self {
            blocks,
            instruction_defined: OnceLock::new(),
        }
    }

    /// Releases the spare capacity of the list and of every block in it.
    ///
    /// The blocks are unchanged, so what [`Self::defines_variable`] has learned
    /// still holds and is kept.
    pub fn shrink_to_fit(&mut self) {
        self.blocks.shrink_to_fit();
        for block in &mut self.blocks {
            block.shrink_to_fit();
        }
    }

    /// Whether some instruction defines `var`.
    ///
    /// Phi results are not instruction definitions, and neither is
    /// [`SsaVarId::PLACEHOLDER`], which is not a variable. The first question
    /// after the blocks last changed costs one pass over every instruction; the
    /// rest cost a bit test.
    ///
    /// `bound` is the function's
    /// [`var_id_bound`](crate::ir::function::SsaFunction::var_id_bound): the set holds the ids
    /// below it as bits and any other id as one entry, so a definition of a
    /// far id costs no more than a definition of any other. It changes no
    /// answer, only what the set costs.
    pub fn defines_variable(&self, var: SsaVarId, bound: usize) -> bool {
        self.instruction_defined
            .get_or_init(|| {
                let mut defined = VarSet::new(bound);
                for block in &self.blocks {
                    for instruction in block.instructions() {
                        for def in instruction.op().defs() {
                            defined.insert(def);
                        }
                    }
                }
                defined
            })
            .contains(var)
    }
}

impl<T: Target> Default for BlockList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Target> Deref for BlockList<T> {
    type Target = Vec<SsaBlock<T>>;

    fn deref(&self) -> &Self::Target {
        &self.blocks
    }
}

impl<T: Target> DerefMut for BlockList<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.instruction_defined = OnceLock::new();
        &mut self.blocks
    }
}

impl<T: Target> Clone for BlockList<T> {
    fn clone(&self) -> Self {
        Self::from_blocks(self.blocks.clone())
    }

    /// Reuses the existing allocations, as [`Vec::clone_from`] does.
    fn clone_from(&mut self, source: &Self) {
        self.blocks.clone_from(&source.blocks);
        self.instruction_defined = OnceLock::new();
    }
}

impl<T: Target> fmt::Debug for BlockList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.blocks.fmt(f)
    }
}

#[cfg(feature = "serde")]
impl<T: Target> Serialize for BlockList<T>
where
    Vec<SsaBlock<T>>: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.blocks.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Target> Deserialize<'de> for BlockList<T>
where
    Vec<SsaBlock<T>>: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::deserialize(deserializer).map(Self::from_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{instruction::SsaInstruction, ops::SsaOp, value::ConstValue},
        testing::MockTarget,
    };

    /// The variable bound the tests ask with.
    const BOUND: usize = 8;

    /// A block holding one `Const` that defines `dest`.
    fn defining(dest: SsaVarId) -> SsaBlock<MockTarget> {
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::new(
            (),
            SsaOp::Const {
                dest,
                value: ConstValue::I32(1),
            },
        ));
        block
    }

    #[test]
    fn only_a_variable_an_instruction_defines_is_defined() {
        let defined = SsaVarId::from_index(3);
        let mut blocks = BlockList::<MockTarget>::new();
        blocks.push(defining(defined));

        assert!(blocks.defines_variable(defined, BOUND));
        assert!(!blocks.defines_variable(SsaVarId::from_index(2), BOUND));
        assert!(
            !blocks.defines_variable(SsaVarId::from_index(1_000), BOUND),
            "an id past every definition is not defined"
        );
    }

    #[test]
    fn a_mutable_borrow_forgets_what_was_learned() {
        let first = SsaVarId::from_index(1);
        let second = SsaVarId::from_index(2);
        let mut blocks = BlockList::<MockTarget>::new();
        blocks.push(defining(first));
        assert!(!blocks.defines_variable(second, BOUND));

        blocks.push(defining(second));
        assert!(blocks.defines_variable(second, BOUND), "pushed block");

        blocks.clear();
        assert!(!blocks.defines_variable(first, BOUND), "cleared");

        blocks.push(defining(first));
        for block in blocks.iter_mut() {
            block.instructions_mut().clear();
        }
        assert!(
            !blocks.defines_variable(first, BOUND),
            "emptied through iter_mut"
        );
    }

    #[test]
    fn a_clone_answers_for_its_own_blocks() {
        let defined = SsaVarId::from_index(1);
        let mut blocks = BlockList::<MockTarget>::new();
        blocks.push(defining(defined));
        assert!(blocks.defines_variable(defined, BOUND));

        let mut copy = blocks.clone();
        copy.clear();
        assert!(!copy.defines_variable(defined, BOUND));
        assert!(blocks.defines_variable(defined, BOUND));

        copy.clone_from(&blocks);
        assert!(copy.defines_variable(defined, BOUND));
    }

    #[test]
    fn a_far_id_is_defined_and_the_placeholder_is_not() {
        let far = SsaVarId::from_index(3_000_000_000);
        let mut blocks = BlockList::<MockTarget>::new();
        blocks.push(defining(far));
        blocks.push(defining(SsaVarId::PLACEHOLDER));

        assert!(blocks.defines_variable(far, BOUND), "a far id is defined");
        assert!(
            !blocks.defines_variable(SsaVarId::PLACEHOLDER, BOUND),
            "the placeholder is not a variable"
        );
        assert!(!blocks.defines_variable(SsaVarId::from_index(3_000_000_001), BOUND));
    }
}
