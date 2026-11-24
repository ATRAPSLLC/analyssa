//! One instruction the rules are asked about, and what is known around it.

use std::collections::BTreeMap;

use crate::{
    analysis::consts::Machine,
    ir::{function::SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    passes::algebraic::{
        Simplification, SimplificationCandidate, addresses, constants, conversions, identities,
        idioms, operations,
    },
    target::Target,
};

/// A rule: the simplification it finds of the instruction at a site, if any.
type Rule<T> = for<'s> fn(&Site<'s, T>) -> Option<SimplificationCandidate<T>>;

/// One instruction the rules are asked about, and what is known around it.
pub struct Site<'a, T: Target> {
    /// The function the instruction is in.
    pub ssa: &'a SsaFunction<T>,
    /// The instruction's operation.
    pub op: &'a SsaOp<T>,
    /// Its block.
    pub block_idx: usize,
    /// Its index in the block.
    pub instr_idx: usize,
    /// The values known to be constants.
    pub constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
    /// How many uses each value has.
    pub uses: &'a BTreeMap<SsaVarId, usize>,
    /// What the rules know of the target.
    pub machine: Machine,
    /// The value the operation defines, where it defines one.
    pub dest: Option<SsaVarId>,
}

impl<T: Target> Clone for Site<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Target> Copy for Site<'_, T> {}

impl<'a, T: Target> Site<'a, T> {
    /// The rules for any operation, tried in order before an operation of
    /// more than one live value is turned away.
    const OPERATIONS: [Rule<T>; 8] = [
        operations::results,
        operations::idle_lanes,
        operations::clear_direction,
        operations::known_mode,
        operations::known_shuffle,
        operations::store_pair,
        operations::narrowed_comparison,
        operations::never_empty,
    ];

    /// The rules for an operation of one live value, tried in order.
    const VALUES: [Rule<T>; 39] = [
        addresses::difference,
        addresses::round_trip,
        addresses::comparison,
        constants::const_fold,
        constants::partly_known,
        conversions::flag_read,
        conversions::convert_chain,
        conversions::reshape_chain,
        conversions::discarded_disjunct,
        identities::zero_test,
        identities::complement_xor,
        identities::negation_sub,
        identities::negation_fold,
        identities::double_negation,
        identities::double_complement,
        identities::authenticated_signature,
        conversions::narrow_truncated,
        addresses::constant_index,
        addresses::scaled_index,
        addresses::zero_offset,
        constants::every_bit_known,
        identities::kept_mask,
        identities::field_read_back,
        identities::bits_moved_back,
        idioms::byte_reversal,
        idioms::rotation,
        idioms::doubleword_sum,
        idioms::doubleword_regroup,
        idioms::doubleword_comparison,
        idioms::wide_load,
        idioms::complement_join,
        idioms::absorption,
        idioms::shifted_part,
        idioms::tripled_half,
        idioms::factored,
        idioms::sign_extension,
        identities::select_choice,
        idioms::mixed_boolean,
        constants::algebraic,
    ];

    /// The instruction `op` at `instr_idx` of block `block_idx` of `ssa`, with
    /// the constants and use counts the rules read.
    pub fn new(
        ssa: &'a SsaFunction<T>,
        op: &'a SsaOp<T>,
        block_idx: usize,
        instr_idx: usize,
        constants: &'a BTreeMap<SsaVarId, ConstValue<T>>,
        uses: &'a BTreeMap<SsaVarId, usize>,
        machine: Machine,
    ) -> Self {
        Self {
            ssa,
            op,
            block_idx,
            instr_idx,
            constants,
            uses,
            machine,
            dest: op.dest(),
        }
    }

    /// The first simplification a rule finds of the instruction, in the
    /// catalogue's order.
    pub fn simplification(&self) -> Option<SimplificationCandidate<T>> {
        if let Some(found) = Self::OPERATIONS.iter().find_map(|rule| rule(self)) {
            return Some(found);
        }
        // A flags output that nothing reads is not a second definition in any way
        // that matters. `Constant` and `Copy` replace the instruction with one that
        // has no flags output, which strands a reader only if a reader exists; the
        // rotate and byte-swap rewrites have no flags output either; and the MBA
        // rewrites set flags differently from the tangle they replace (`a + ~b + 1`
        // and `a - b` disagree on the carry). So every rule needs the flags dead,
        // and an op whose flags are read is left alone. x86 sets flags on the
        // arithmetic that spells these identities (`xor v6, v8 flags=v10`), so
        // checking rather than refusing every flags output is what lets the
        // catalogue apply there at all.
        //
        // Exactly two definitions, one of them an unread flags output, is the only
        // shape that qualifies: an op with two *value* definitions (wide
        // arithmetic) would still lose a live one.
        let only_live_def_is_the_value = self.op.defs().count() == 2 && self.flags_dead();
        if self.op.defs().count() != 1 && !only_live_def_is_the_value {
            return None;
        }
        self.dest?;
        Self::VALUES.iter().find_map(|rule| rule(self))
    }

    /// The candidate that rewrites the instruction as `simplification`, the
    /// value `dest` standing for what it rewrites.
    pub fn candidate(
        &self,
        dest: SsaVarId,
        simplification: Simplification<T>,
        description: &'static str,
    ) -> SimplificationCandidate<T> {
        SimplificationCandidate {
            block_idx: self.block_idx,
            instr_idx: self.instr_idx,
            dest,
            simplification,
            also: Vec::new(),
            description,
        }
    }

    /// Whether nothing reads the flags the operation defines, where it
    /// defines any.
    pub fn flags_dead(&self) -> bool {
        self.op
            .flags_dest()
            .is_none_or(|flags| self.uses.get(&flags).copied().unwrap_or(0) == 0)
    }

    /// What `var` negates, where a negation defines it.
    pub fn negated(&self, var: &SsaVarId) -> Option<SsaVarId> {
        match self.ssa.get_definition(*var) {
            Some(SsaOp::Neg { operand, .. }) => Some(*operand),
            _ => None,
        }
    }
}
