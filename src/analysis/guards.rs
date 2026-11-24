//! What a function's branches prove where they lead.
//!
//! A block entered along one edge, and no way in other than that edge, is
//! entered knowing that edge's answer to its terminator's question, and so is
//! every block it dominates ([`GuardFacts`]). The questions are read in one
//! form across the ways the IR spells them ([`Question`]), and an answer that
//! proves a value equal to a number ([`Equality`]) carries to what the value
//! was computed from and to what it is passed on to.
//!
//! The opaque-predicate pass decides terminators, selects and reads from the
//! facts; a type analysis reads what a phi joins on each edge.

use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
};

use rustc_hash::FxHashSet;

use crate::{
    analysis::{
        SsaCfg,
        defuse::DefUseIndex,
        memory::{MemoryOp, MemorySsa},
    },
    graph::{
        NodeId, RootedGraph,
        algorithms::{DominatorTree, compute_dominators, reverse_postorder},
    },
    ir::{
        block::SsaBlock,
        exception::ExceptionBlocks,
        function::SsaFunction,
        integer::BitWidth,
        ops::{SsaOp, kinds::CmpKind, native::FlagCondition},
        phi::{PhiEdge, PhiNode, PhiOperand},
        value::ConstValue,
        variable::SsaVarId,
        varstore::{VarMap, VarSet},
    },
    pointer::PointerSize,
    target::Target,
};

/// A select at a join whose condition every way into the join decides
/// ([`GuardFacts::decided_join_selects`]): the phi of the arms the ways
/// choose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinSelect {
    /// The join.
    pub block: usize,
    /// The select's position in the join.
    pub instruction: usize,
    /// What the select defines, which the phi defines instead.
    pub dest: SsaVarId,
    /// The value each predecessor's edge chooses, by predecessor.
    pub arms: Vec<(usize, SsaVarId)>,
}

/// What the guard facts read of a function's values: each value's defining
/// operation, whether it may hold a float, its declared width, and, where the
/// function's memory is read ([`Self::reading`]), which loads read one cell in
/// one state.
pub struct Definitions<T: Target> {
    /// Index mapping variables to their defining block, instruction, and op.
    pub index: DefUseIndex<T>,
    /// Variables whose declared type is floating point, or unknown.
    ///
    /// Self-cancelling identities do not hold for floats: `x == x` is *false*
    /// when `x` is NaN, and `x - x` is NaN rather than zero. `Ceq` and `Sub`
    /// carry no float discriminator - `ConstValue` folds `F32`/`F64` pairs for
    /// both - so the declared type is the only way to tell. An unknown type is
    /// treated as possibly-float.
    ///
    /// (`Xor` is deliberately *not* gated on this: `x ^ x` compares bit patterns,
    /// and a value's bits equal themselves even when they encode NaN.)
    maybe_float: VarSet,
    /// Each variable's declared bit width; absent where the type has no known
    /// width.
    widths: VarMap<u32>,
    /// Each load's read: the cell it reads and the state of memory it reads
    /// it in ([`MemorySsa::version_before`]). Two loads with one read hold
    /// one value. Empty until [`Self::reading`] reads the memory.
    reads: VarMap<(usize, u32)>,
    /// Each phi's operands, the values it joins.
    joins: VarMap<Vec<SsaVarId>>,
}

impl<T: Target> Definitions<T> {
    /// Whether `value` is a boolean: what a comparison or `bool.not` defines,
    /// read through copies. A value only known nonzero is none, since an
    /// integer a guard tested may be `5`, which `true` is not.
    pub fn is_boolean(&self, value: SsaVarId) -> bool {
        let mut current = value;
        for _ in 0..Question::DEPTH {
            match self.get_definition(current) {
                Some(SsaOp::Copy { src, .. }) => current = *src,
                Some(
                    SsaOp::BoolNot { .. }
                    | SsaOp::Ceq { .. }
                    | SsaOp::Clt { .. }
                    | SsaOp::Cgt { .. },
                ) => return true,
                _ => return false,
            }
        }
        false
    }

    /// The boolean `left == right` tests, and the truth it tests it for, where
    /// one side is a boolean ([`Self::is_boolean`]) and the other a truth
    /// value: `true` or `1`, `false` or `0`.
    ///
    /// Such a test asks the boolean's own question, or its negation: `(x < 0)
    /// == false` is `x >= 0`, as a flag condition read back as a value and
    /// tested against zero states it. Another number is no truth value, and
    /// a boolean equals it never.
    fn truth_test(&self, left: SsaVarId, right: SsaVarId) -> Option<(SsaVarId, bool)> {
        let truth = |var: SsaVarId| match self.get_definition(var)? {
            SsaOp::Const {
                value: ConstValue::True,
                ..
            } => Some(true),
            SsaOp::Const {
                value: ConstValue::False,
                ..
            } => Some(false),
            SsaOp::Const { value, .. } => match value.as_i64() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                _ => None,
            },
            _ => None,
        };
        match (truth(right), truth(left)) {
            (Some(truth), _) if self.is_boolean(left) => Some((left, truth)),
            (_, Some(truth)) if self.is_boolean(right) => Some((right, truth)),
            _ => None,
        }
    }

    /// Reads the definitions of `ssa`'s values.
    #[must_use]
    pub fn of(ssa: &SsaFunction<T>) -> Self {
        let bound = ssa.var_id_bound();
        let mut maybe_float = VarSet::new(bound);
        let mut widths = VarMap::new(bound);
        for var in ssa.variables() {
            let ty = var.var_type();
            if T::is_floating(ty) || T::is_unknown(ty) {
                maybe_float.insert(var.id());
            }
            if let Some(width) = T::bit_width(ty) {
                widths.insert(var.id(), width);
            }
        }
        let mut joins = VarMap::new(bound);
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                joins.insert(
                    phi.result(),
                    phi.operands().iter().map(PhiOperand::value).collect(),
                );
            }
        }
        Self {
            index: DefUseIndex::build_with_ops(ssa),
            maybe_float,
            widths,
            reads: VarMap::new(bound),
            joins,
        }
    }

    /// Reads what each load of `memory`, the function's memory, reads: the
    /// cell and the state it is read in, so that two loads of one cell with
    /// no write to it between are one value ([`Self::same_value`]).
    #[must_use]
    pub fn reading(mut self, memory: &MemorySsa<T>) -> Self {
        for op in memory.operations() {
            if let MemoryOp::Load {
                location,
                dest,
                block,
                instr,
            } = op
                && let Some(id) = memory.location_id(location)
                && let Some(version) = memory.version_before_id(id, *block, *instr)
            {
                self.reads.insert(*dest, (id, version));
            }
        }
        self
    }

    /// Whether `a` and `b` are loads of one cell in one state of memory
    /// ([`Self::reading`]).
    #[must_use]
    pub fn same_read(&self, a: SsaVarId, b: SsaVarId) -> bool {
        matches!((self.reads.get(a), self.reads.get(b)), (Some(x), Some(y)) if x == y)
    }

    /// The loads that read the cell `cell` ([`MemorySsa::location_id`]) in
    /// the state `version` ([`MemorySsa::version_before_id`]).
    pub fn loads_reading(&self, cell: usize, version: u32) -> impl Iterator<Item = SsaVarId> + '_ {
        self.reads
            .iter()
            .filter(move |(_, read)| **read == (cell, version))
            .map(|(load, _)| load)
    }

    /// Returns `true` when `var` may hold a floating-point value.
    ///
    /// Self-cancelling identities (`x == x`, `x - x`) are unsound for floats, so
    /// their analyzers gate on this. Membership is seeded once at
    /// [`of`](Self::of) time from each variable's declared type; a type
    /// that is unknown is treated as possibly-float, so this errs toward `true`
    /// rather than miscategorising an unknown as safe.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable whose declared type is being queried.
    ///
    /// # Returns
    ///
    /// `true` if `var`'s declared type is floating point or unknown, `false` otherwise.
    #[must_use]
    pub fn maybe_float(&self, var: SsaVarId) -> bool {
        self.maybe_float.contains(var)
    }
    /// Returns `var`'s declared bit width, or `None` when its type has none.
    #[must_use]
    pub fn bit_width(&self, var: SsaVarId) -> Option<u32> {
        self.widths.get(var).copied()
    }
    /// Looks up the defining operation for a variable.
    ///
    /// # Returns
    ///
    /// The [`SsaOp`] that defines `var`, or `None` when `var` has no defining
    /// instruction (it is phi-defined or a parameter).
    #[must_use]
    pub fn get_definition(&self, var: SsaVarId) -> Option<&SsaOp<T>> {
        self.index.def_op(var)
    }
    /// Whether `a` and `b` hold one value: one variable, two constants alike
    /// ([`ConstValue`]'s equality: one arm, the same bits), or two loads of
    /// one cell in one state ([`Self::same_read`]). A compiler materialises a
    /// literal afresh before each test that reads it, and the two definitions
    /// of `3` in two blocks are the one number; it loads a variable afresh
    /// after a loop that loaded it last, and the two loads are the one value.
    #[must_use]
    pub fn same_value(&self, a: SsaVarId, b: SsaVarId) -> bool {
        a == b
            || matches!(
                (self.get_definition(a), self.get_definition(b)),
                (Some(SsaOp::Const { value: x, .. }), Some(SsaOp::Const { value: y, .. }))
                    if x == y
            )
            || self.same_read(a, b)
    }

    /// The bits and width of the integer literal `var`, when a constant with
    /// an integer value defines it, or a phi each of whose values is that one
    /// literal: the paths that set a flag to `1` and meet hold `1`, and a
    /// loop's phi of `1` and itself never holds anything else.
    #[must_use]
    pub fn literal(&self, var: SsaVarId, ptr_size: PointerSize) -> Option<(u64, u32)> {
        let mut seen = BTreeSet::from([var]);
        let mut pending = vec![var];
        let mut found = None;
        while let Some(next) = pending.pop() {
            if let Some(operands) = self.joins.get(next) {
                pending.extend(operands.iter().copied().filter(|value| seen.insert(*value)));
                continue;
            }
            let literal = match self.get_definition(next)? {
                SsaOp::Const { value, .. } => value.integer_pattern(ptr_size)?,
                _ => return None,
            };
            if *found.get_or_insert(literal) != literal {
                return None;
            }
        }
        found
    }
}

/// What a two-way terminator or a select's condition asks, in one form across
/// the ways the IR spells it: a flags value's condition, a comparison of two
/// values, or a boolean value itself.
///
/// A boolean is read through what defines it - copies, `bool.not`, and the
/// comparisons `ceq`, `clt`, `cgt` - so `select (bool.not (ceq a b))` and
/// `branchcmp a == b` ask one question, with opposite answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Question {
    /// A flags value's condition (`branchflags`).
    Flags {
        /// The flags value.
        flags: SsaVarId,
        /// The condition read from it.
        condition: FlagCondition,
    },
    /// A comparison of two values.
    Compare {
        /// The left operand.
        left: SsaVarId,
        /// The right operand.
        right: SsaVarId,
        /// The relation.
        cmp: CmpKind,
        /// Whether the operands compare unsigned; `false` for an equality,
        /// which signedness does not change.
        unsigned: bool,
    },
    /// A boolean value whose definition says no more.
    Value(SsaVarId),
}

/// One side of a comparison as [`Question::relates`] reads it
/// ([`Question::sides`]).
#[derive(Debug, Clone, Copy)]
enum Compared {
    /// A value.
    Value(SsaVarId),
    /// An integer literal: its bits, and the width it is compared at.
    Literal(u64, u32),
}

impl Compared {
    /// Whether the two are one operand to a comparison: one value
    /// ([`Definitions::same_value`]), or literals of one integer at one
    /// width, whatever the signedness each is written with.
    fn same<T: Target>(self, other: Self, cache: &Definitions<T>) -> bool {
        match (self, other) {
            (Self::Value(one), Self::Value(other)) => cache.same_value(one, other),
            (Self::Literal(one, width), Self::Literal(other, other_width)) => {
                one == other && width == other_width
            }
            _ => false,
        }
    }
}

impl Question {
    /// How many definitions [`Self::of_value`] reads through.
    const DEPTH: usize = 16;

    /// The question asked of `value(v)` for each value `v` it asks of: the
    /// same question where a path holds another value in each one's place.
    #[must_use]
    pub fn reading(self, value: impl Fn(SsaVarId) -> SsaVarId) -> Self {
        match self {
            Self::Flags { flags, condition } => Self::Flags {
                flags: value(flags),
                condition,
            },
            Self::Compare {
                left,
                right,
                cmp,
                unsigned,
            } => Self::Compare {
                left: value(left),
                right: value(right),
                cmp,
                unsigned,
            },
            Self::Value(held) => Self::Value(value(held)),
        }
    }

    /// The question `terminator` branches on, and the answer that takes its
    /// true target; `None` for a terminator that asks none.
    pub fn of_terminator<T: Target>(
        terminator: &SsaOp<T>,
        cache: &Definitions<T>,
    ) -> Option<(Self, bool)> {
        match terminator {
            SsaOp::BranchFlags {
                flags, condition, ..
            } => Some((
                Self::Flags {
                    flags: *flags,
                    condition: *condition,
                },
                true,
            )),
            SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                ..
            } => match (cmp, cache.truth_test(*left, *right)) {
                (CmpKind::Eq | CmpKind::Ne, Some((boolean, truth))) => {
                    let (question, holds) = Self::of_value(boolean, cache);
                    Some((question, holds == (truth == (*cmp == CmpKind::Eq))))
                }
                _ => Some((Self::compare(*left, *right, *cmp, *unsigned), true)),
            },
            SsaOp::Branch { condition, .. } => Some(Self::of_value(*condition, cache)),
            _ => None,
        }
    }

    /// The question the boolean `value` answers, and the answer for which it
    /// is true.
    pub fn of_value<T: Target>(value: SsaVarId, cache: &Definitions<T>) -> (Self, bool) {
        let mut current = value;
        let mut holds = true;
        for _ in 0..Self::DEPTH {
            match cache.get_definition(current) {
                Some(SsaOp::Copy { src, .. }) => current = *src,
                Some(SsaOp::BoolNot { value, .. }) => {
                    current = *value;
                    holds = !holds;
                }
                Some(SsaOp::Ceq { left, right, .. }) => {
                    let Some((boolean, truth)) = cache.truth_test(*left, *right) else {
                        return (Self::compare(*left, *right, CmpKind::Eq, false), holds);
                    };
                    current = boolean;
                    if !truth {
                        holds = !holds;
                    }
                }
                Some(SsaOp::Clt {
                    left,
                    right,
                    unsigned,
                    ..
                }) => return (Self::compare(*left, *right, CmpKind::Lt, *unsigned), holds),
                Some(SsaOp::Cgt {
                    left,
                    right,
                    unsigned,
                    ..
                }) => return (Self::compare(*left, *right, CmpKind::Gt, *unsigned), holds),
                _ => break,
            }
        }
        (Self::Value(current), holds)
    }

    /// The comparison `cmp` of `left` and `right`.
    pub const fn compare(left: SsaVarId, right: SsaVarId, cmp: CmpKind, unsigned: bool) -> Self {
        Self::Compare {
            left,
            right,
            cmp,
            unsigned: unsigned && !matches!(cmp, CmpKind::Eq | CmpKind::Ne),
        }
    }

    /// Whether `asked` holds exactly when this question does (`Some(true)`),
    /// exactly when it does not (`Some(false)`), or neither is known.
    ///
    /// Only questions over the *same operands* relate: the same flags value,
    /// the same pair compared, in either order, each operand one value or
    /// literals of one integer at one width, the same boolean. A value
    /// zero-extended and compared with a literal is the value itself
    /// compared with it, where the literal fits the value's width and the
    /// comparison is an equality or unsigned: `*p == 0` on a byte and
    /// `c != 0` on the word a loop widened it to ask one question. Compiled code routinely holds complementary relations over
    /// different pairs, and nothing follows from those. SSA makes the check
    /// exact: a flags value is single-assignment, so naming it twice is the
    /// guarantee nothing clobbered the flags between.
    pub fn relates<T: Target>(
        self,
        asked: Self,
        cache: &Definitions<T>,
        ptr_size: PointerSize,
    ) -> Option<bool> {
        match (self, asked) {
            (
                Self::Flags {
                    flags: known_flags,
                    condition: known,
                },
                Self::Flags {
                    flags: asked_flags,
                    condition: asked,
                },
            ) => {
                if known_flags != asked_flags {
                    return None;
                }
                // Every `FlagCondition` tests one bit set or clear, so the
                // complement is exact whatever produced the flags - an unordered
                // float compare still leaves `ZF` in a definite state.
                if asked == known {
                    Some(true)
                } else if asked == known.negated() {
                    Some(false)
                } else {
                    None
                }
            }
            (
                Self::Compare {
                    left: known_left,
                    right: known_right,
                    cmp: known_cmp,
                    unsigned: known_unsigned,
                },
                Self::Compare {
                    left,
                    right,
                    cmp,
                    unsigned,
                },
            ) => {
                // The pair compared the other way round asks the swapped
                // relation (`new <= lim` is `lim >= new`), for floats too:
                // each side is false exactly where the other is.
                let (known_one, known_other) = Self::sides(
                    known_left,
                    known_right,
                    known_cmp,
                    known_unsigned,
                    cache,
                    ptr_size,
                );
                let (one, other) = Self::sides(left, right, cmp, unsigned, cache, ptr_size);
                let same = |a: Compared, b: Compared| a.same(b, cache);
                let cmp = if same(known_one, one) && same(known_other, other) {
                    cmp
                } else if same(known_one, other) && same(known_other, one) {
                    cmp.swapped()
                } else {
                    return None;
                };
                if known_unsigned != unsigned {
                    return None;
                }
                if cmp == known_cmp {
                    return Some(true);
                }
                if cmp != known_cmp.negated() {
                    return None;
                }
                // `Lt`/`Ge` and `Gt`/`Le` stop partitioning once an operand can
                // be NaN: both sides are false, so neither implies the other.
                // `Eq`/`Ne` survive, `Ne` being the unordered-or-unequal
                // relation. See `CmpKind::negated`.
                let ordered = !matches!(known_cmp, CmpKind::Eq | CmpKind::Ne);
                if ordered && (cache.maybe_float(known_left) || cache.maybe_float(known_right)) {
                    return None;
                }
                Some(false)
            }
            (Self::Value(known), Self::Value(asked)) => (known == asked).then_some(true),
            _ => None,
        }
    }

    /// The two sides of `left cmp right`, compared `unsigned` or not, as
    /// [`Self::relates`] reads them: each a value, or an integer literal by
    /// its bits and width ([`Definitions::literal`]). Where one side is a
    /// literal and the other zero-extends a narrower integer, and the literal
    /// fits that width, the comparison is the narrower value's with the
    /// literal at that width, provided it is an equality or compares
    /// unsigned: zero extension keeps every value and its unsigned order. A
    /// signed order is not kept (a byte's `0x80` widened is `128`, read
    /// signed at the byte's width it is `-128`), nor a literal past the
    /// narrower width, which no widened value equals.
    fn sides<T: Target>(
        left: SsaVarId,
        right: SsaVarId,
        cmp: CmpKind,
        unsigned: bool,
        cache: &Definitions<T>,
        ptr_size: PointerSize,
    ) -> (Compared, Compared) {
        let side = |var: SsaVarId| {
            cache
                .literal(var, ptr_size)
                .map_or(Compared::Value(var), |(bits, width)| {
                    Compared::Literal(bits, width)
                })
        };
        let (mut one, mut other) = (side(left), side(right));
        if !(unsigned || matches!(cmp, CmpKind::Eq | CmpKind::Ne)) {
            return (one, other);
        }
        // The narrower value a zero extension of `var` widens, with its
        // width, where `bits` fits it.
        let narrowed = |var: SsaVarId, bits: u64| -> Option<(SsaVarId, u32)> {
            let SsaOp::IntConv {
                operand,
                unsigned: true,
                ..
            } = cache.get_definition(var)?
            else {
                return None;
            };
            let narrow = cache.bit_width(*operand)?;
            let wide = cache.bit_width(var)?;
            let fits = bits.checked_shr(narrow).unwrap_or(0) == 0;
            (narrow < wide && fits && !cache.maybe_float(*operand)).then_some((*operand, narrow))
        };
        for _ in 0..Self::DEPTH {
            match (one, other) {
                (Compared::Value(var), Compared::Literal(bits, _)) => {
                    let Some((inner, width)) = narrowed(var, bits) else {
                        break;
                    };
                    (one, other) = (Compared::Value(inner), Compared::Literal(bits, width));
                }
                (Compared::Literal(bits, _), Compared::Value(var)) => {
                    let Some((inner, width)) = narrowed(var, bits) else {
                        break;
                    };
                    (one, other) = (Compared::Literal(bits, width), Compared::Value(inner));
                }
                _ => break,
            }
        }
        (one, other)
    }

    /// What `asked` comes to where this question has the answer `held`.
    pub fn decides<T: Target>(
        self,
        held: bool,
        asked: Self,
        cache: &Definitions<T>,
        ptr_size: PointerSize,
    ) -> Option<bool> {
        self.relates(asked, cache, ptr_size)
            .map(|same| same == held)
    }

    /// The value this question, answered `held`, proves equal to a number:
    /// a comparison with a literal answered equal, a zero flag set on what
    /// its producer computed, or a branch on an integer not taken.
    ///
    /// A floating-point value is equated with nothing: `-0.0 == 0.0`, so the
    /// comparison does not fix the value's bits.
    pub fn equality<T: Target>(
        self,
        held: bool,
        ssa: &SsaFunction<T>,
        cache: &Definitions<T>,
        ptr_size: PointerSize,
    ) -> Option<Equality> {
        let equality = match self {
            Self::Compare {
                left, right, cmp, ..
            } => {
                let equal = match cmp {
                    CmpKind::Eq => held,
                    CmpKind::Ne => !held,
                    _ => return None,
                };
                if !equal {
                    return None;
                }
                let (value, (bits, width)) = match (
                    cache.literal(left, ptr_size),
                    cache.literal(right, ptr_size),
                ) {
                    (None, Some(literal)) => (left, literal),
                    (Some(literal), None) => (right, literal),
                    _ => return None,
                };
                Equality { value, bits, width }
            }
            Self::Flags { flags, condition } => {
                let zero = match condition {
                    FlagCondition::Zero => held,
                    FlagCondition::NotZero => !held,
                    _ => return None,
                };
                if !zero {
                    return None;
                }
                let value = ssa.flag_source(flags)?;
                Equality {
                    value,
                    bits: 0,
                    width: cache.bit_width(value)?,
                }
            }
            Self::Value(value) => {
                if held || cache.is_boolean(value) {
                    return None;
                }
                Equality {
                    value,
                    bits: 0,
                    width: cache.bit_width(value)?,
                }
            }
        };
        (cache.bit_width(equality.value) == Some(equality.width)
            && !cache.maybe_float(equality.value))
        .then_some(equality)
    }
}

/// A value a guard proves equal to a number: `value` holds the integer whose
/// bits are `bits` at `width`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Equality {
    /// The value proved.
    value: SsaVarId,
    /// The number's bits, in the low `width` bits.
    bits: u64,
    /// The number's width, which is the value's.
    width: u32,
}

impl Equality {
    /// The equality of `value` with the integer whose bits are `bits` at
    /// `width`, the value's width: one a path proves, rather than a guard.
    #[must_use]
    pub const fn new(value: SsaVarId, bits: u64, width: u32) -> Self {
        Self { value, bits, width }
    }

    /// The number's bits, in the low `width` bits.
    #[must_use]
    pub const fn bits(&self) -> u64 {
        self.bits
    }

    /// The value proved.
    #[must_use]
    pub const fn value(&self) -> SsaVarId {
        self.value
    }

    /// Whether the number is zero: at a pointer, the null pointer.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.bits == 0
    }

    /// How many definitions [`Self::implied`] follows.
    const DEPTH: usize = 16;

    /// This equality and those it implies through the value's definition: a
    /// value proved equal to a number proves what it was computed from
    /// equal to one wherever the computation can be undone exactly.
    ///
    /// `d = x - c` equal to `k` makes `x` equal to `k + c`, and so do sums,
    /// differences from a literal, an exclusive or with one, negation and
    /// complement, a copy, `x & x` and `x | x` (a test of `x` against itself),
    /// and the reinterpretations between an integer and a pointer of the same
    /// width. A widening conversion is undone where the number is one the
    /// narrower value widens to. Every step is modulo the width, as the
    /// machine computes it. A value that is itself a literal, or may be a
    /// float, ends the walk.
    pub fn implied<T: Target>(self, cache: &Definitions<T>, ptr_size: PointerSize) -> Vec<Self> {
        let mut implied = Vec::new();
        let mut current = self;
        for _ in 0..Self::DEPTH {
            if cache.literal(current.value, ptr_size).is_some() || cache.maybe_float(current.value)
            {
                break;
            }
            implied.push(current);
            let Some(next) = current.undone(cache, ptr_size) else {
                break;
            };
            current = next;
        }
        implied
    }

    /// What the definition of this equality's value, undone, proves of the
    /// value it was computed from.
    pub fn undone<T: Target>(self, cache: &Definitions<T>, ptr_size: PointerSize) -> Option<Self> {
        let Self { bits, width, .. } = self;
        let mask = BitWidth::new(width).mask();
        // The literal `var` at this width, as bits.
        let literal = |var: SsaVarId| {
            cache
                .literal(var, ptr_size)
                .and_then(|(bits, literal_width)| (literal_width == width).then_some(bits))
        };
        let at = |value: SsaVarId, bits: u64| {
            (cache.bit_width(value) == Some(width)).then_some(Self {
                value,
                bits: bits & mask,
                width,
            })
        };
        match cache.get_definition(self.value)? {
            SsaOp::Copy { src, .. }
            | SsaOp::PtrToInt { operand: src, .. }
            | SsaOp::IntToPtr { operand: src, .. } => at(*src, bits),
            SsaOp::And { left, right, .. } | SsaOp::Or { left, right, .. } if left == right => {
                at(*left, bits)
            }
            SsaOp::Sub { left, right, .. } => match (literal(*left), literal(*right)) {
                (None, Some(subtrahend)) => at(*left, bits.wrapping_add(subtrahend)),
                (Some(minuend), None) => at(*right, minuend.wrapping_sub(bits)),
                _ => None,
            },
            SsaOp::Add { left, right, .. } => match (literal(*left), literal(*right)) {
                (None, Some(addend)) => at(*left, bits.wrapping_sub(addend)),
                (Some(addend), None) => at(*right, bits.wrapping_sub(addend)),
                _ => None,
            },
            SsaOp::Xor { left, right, .. } => match (literal(*left), literal(*right)) {
                (None, Some(mask)) => at(*left, bits ^ mask),
                (Some(mask), None) => at(*right, bits ^ mask),
                _ => None,
            },
            SsaOp::Neg { operand, .. } => at(*operand, bits.wrapping_neg()),
            SsaOp::Not { operand, .. } => at(*operand, !bits),
            SsaOp::IntConv {
                operand, unsigned, ..
            } => {
                let narrow = cache.bit_width(*operand)?;
                if narrow == width {
                    return at(*operand, bits);
                }
                if narrow > width {
                    return None;
                }
                let narrow_mask = BitWidth::new(narrow).checked_mask()?;
                let truncated = bits & narrow_mask;
                let sign = 1u64.checked_shl(narrow.checked_sub(1)?)?;
                let widened = if !*unsigned && truncated & sign != 0 {
                    (truncated | !narrow_mask) & mask
                } else {
                    truncated
                };
                (widened == bits).then_some(Self {
                    value: *operand,
                    bits: truncated,
                    width: narrow,
                })
            }
            _ => None,
        }
    }

    /// Whether `op` only passes its operand on: a copy, or a conversion that
    /// reinterprets or resizes it.
    pub fn passes_on<T: Target>(op: &SsaOp<T>) -> bool {
        matches!(
            op,
            SsaOp::Copy { .. }
                | SsaOp::IntConv { .. }
                | SsaOp::PtrToInt { .. }
                | SsaOp::IntToPtr { .. }
                | SsaOp::Bitcast { .. }
        )
    }

    /// What `value` is proved equal to where `proved` holds: its own
    /// equality, or the number a decided value it was passed on from
    /// ([`Self::passes_on`]) carries to it - the same bits through a copy or
    /// a reinterpretation of its width, and those bits extended or truncated
    /// through an integer conversion, as the conversion computes them.
    pub fn read<T: Target>(
        proved: &Proved<'_>,
        value: SsaVarId,
        cache: &Definitions<T>,
    ) -> Option<Self> {
        if let Some(equality) = proved.get(value) {
            return Some(equality);
        }
        let mut chain = Vec::new();
        let mut current = value;
        let source = loop {
            if chain.len() >= Self::DEPTH {
                return None;
            }
            let operand = match cache.get_definition(current)? {
                SsaOp::Copy { src, .. } => *src,
                SsaOp::IntConv { operand, .. }
                | SsaOp::PtrToInt { operand, .. }
                | SsaOp::IntToPtr { operand, .. }
                | SsaOp::Bitcast { operand, .. } => *operand,
                _ => return None,
            };
            chain.push(current);
            if let Some(equality) = proved.get(operand) {
                break equality;
            }
            current = operand;
        };
        chain
            .into_iter()
            .rev()
            .try_fold(source, |carried, dest| carried.passed_to(dest, cache))
    }

    /// This number as the definition of `dest`, which passes this equality's
    /// value on ([`Self::passes_on`]), computes it.
    fn passed_to<T: Target>(self, dest: SsaVarId, cache: &Definitions<T>) -> Option<Self> {
        let width = cache.bit_width(dest)?;
        if cache.maybe_float(dest) {
            return None;
        }
        let mask = |bits: u32| {
            1u64.checked_shl(bits)
                .map_or(u64::MAX, |bit| bit.wrapping_sub(1))
        };
        let bits = match cache.get_definition(dest)? {
            SsaOp::IntConv { unsigned, .. } if width > self.width => {
                let sign = 1u64.checked_shl(self.width.checked_sub(1)?)?;
                if !*unsigned && self.bits & sign != 0 {
                    (self.bits | !mask(self.width)) & mask(width)
                } else {
                    self.bits
                }
            }
            SsaOp::IntConv { .. } => self.bits & mask(width),
            _ if width == self.width => self.bits,
            _ => return None,
        };
        Some(Self {
            value: dest,
            bits,
            width,
        })
    }

    /// The number as a constant of `ty`, the type of the value read: `None`
    /// where `ty` is no integer of this width.
    fn literal_of<T: Target>(self, ty: &T::Type) -> Option<ConstValue<T>> {
        (T::bit_width(ty) == Some(self.width))
            .then(|| ConstValue::from_bits_of_type(self.bits, ty))
            .flatten()
    }
}

/// What the facts holding at one point prove equal to a number: the
/// equalities of each block whose facts hold there, nearest first, then
/// those of the facts an edge adds ([`GuardReader::equalities`]).
///
/// The first layer proving a value wins, as the first fact proving one does
/// among facts read in that order.
pub struct Proved<'a> {
    /// The layers, in the order they are read.
    layers: Vec<&'a BTreeMap<SsaVarId, Equality>>,
}

impl<'a> Proved<'a> {
    /// What `asked` comes to where `proved` holds: a comparison of two
    /// values each proved equal to a number or a literal, at one width,
    /// or a boolean value proved a number. A float is never proved one
    /// ([`Question::equality`]), and a literal of a float type is no integer.
    #[must_use]
    pub fn evaluated<T: Target>(
        &self,
        asked: Question,
        cache: &Definitions<T>,
        ptr_size: PointerSize,
    ) -> Option<bool> {
        let number = |value: SsaVarId| {
            Equality::read(self, value, cache)
                .map(|equality| (equality.bits, equality.width))
                .or_else(|| cache.literal(value, ptr_size))
        };
        match asked {
            Question::Compare {
                left,
                right,
                cmp,
                unsigned,
            } => {
                if cache.maybe_float(left) || cache.maybe_float(right) {
                    return None;
                }
                let ((first, width), (second, other)) = (number(left)?, number(right)?);
                if width != other {
                    return None;
                }
                let mask = BitWidth::new(width).checked_mask()?;
                let (first, second) = (first & mask, second & mask);
                let order = if unsigned {
                    first.cmp(&second)
                } else {
                    let signed = |bits: u64| {
                        let shift = 64u32.checked_sub(width)?;
                        i64::from_ne_bytes(bits.checked_shl(shift)?.to_ne_bytes())
                            .checked_shr(shift)
                    };
                    signed(first)?.cmp(&signed(second)?)
                };
                Some(match cmp {
                    CmpKind::Eq => order.is_eq(),
                    CmpKind::Ne => order.is_ne(),
                    CmpKind::Lt => order.is_lt(),
                    CmpKind::Le => order.is_le(),
                    CmpKind::Gt => order.is_gt(),
                    CmpKind::Ge => order.is_ge(),
                })
            }
            Question::Value(value) => number(value).map(|(bits, _)| bits != 0),
            Question::Flags { .. } => None,
        }
    }

    /// What `equalities` alone prove.
    #[must_use]
    pub fn of(equalities: &'a BTreeMap<SsaVarId, Equality>) -> Self {
        Self {
            layers: vec![equalities],
        }
    }

    /// The equality the first layer proving `value` holds of it.
    #[must_use]
    pub fn get(&self, value: SsaVarId) -> Option<Equality> {
        self.layers
            .iter()
            .find_map(|layer| layer.get(&value).copied())
    }
}

/// What [`GuardFacts::of`] reads of a function to learn what the ways
/// into a join share past it.
struct Ways<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Its graph, a throw's ways in among the terminators'.
    cfg: &'a SsaCfg<'a, T>,
    /// Its values' definitions.
    cache: &'a Definitions<T>,
    /// The width of its pointers.
    ptr_size: PointerSize,
    /// Its memory, where it is read.
    memory: Option<&'a MemorySsa<T>>,
    /// Dominance from its entry.
    dominators: &'a DominatorTree,
}

impl<T: Target> Clone for Ways<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Target> Copy for Ways<'_, T> {}

impl<T: Target> Ways<'_, T> {
    /// What every way into `join` that its edge answering `question` with
    /// `held` can come from shares, on the way into it: nothing unless the
    /// edge rules out some way. A way is the facts of a predecessor with
    /// that predecessor's own edge answer; one the edge's answer contradicts
    /// ([`Question::decides`], or the numbers its facts prove) never takes
    /// the edge. A way whose facts are not read yet (a back edge) knows
    /// nothing, so nothing is shared past it.
    fn shared(
        &self,
        known: &GuardFacts,
        done: &[bool],
        (join, question, held): (usize, Question, bool),
    ) -> Vec<(Question, bool)> {
        let Self {
            ssa,
            cfg,
            cache,
            ptr_size,
            memory,
            ..
        } = *self;
        let mut ways: Vec<usize> = cfg.block_predecessors(join).to_vec();
        ways.sort_unstable();
        ways.dedup();
        if ways.len() < 2 {
            return Vec::new();
        }
        let mut shared: Option<Vec<(Question, bool)>> = None;
        let mut ruled_out = false;
        let reader = GuardReader::new(ssa, cache, ptr_size);
        for way in ways {
            if !done.get(way).copied().unwrap_or(false) {
                return Vec::new();
            }
            let mut facts: Vec<(Question, bool)> = known.facts(way).collect();
            facts.extend(reader.edge_answer(way, join));
            let read = memory
                .map(|memory| self.reads(memory, join, way))
                .unwrap_or_default();
            let asked = question.reading(|value| read.get(&value).copied().unwrap_or(value));
            let decided = facts
                .iter()
                .find_map(|(known, answer)| known.decides(*answer, asked, cache, ptr_size))
                .or_else(|| {
                    let equalities = reader.equalities(&facts);
                    Proved::of(&equalities).evaluated(asked, cache, ptr_size)
                });
            if decided == Some(!held) {
                ruled_out = true;
                continue;
            }
            shared = Some(match shared {
                None => facts,
                Some(mut common) => {
                    let holds: FxHashSet<(Question, bool)> = facts.into_iter().collect();
                    common.retain(|fact| holds.contains(fact));
                    common
                }
            });
        }
        if ruled_out {
            shared.unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Each load of `join` its test may read, with nothing written to its
    /// cell in `join` before it, mapped to a load at or above `way` of the
    /// cell in the state `way` leaves it in: on that way the two are one value.
    fn reads(
        &self,
        memory: &MemorySsa<T>,
        join: usize,
        way: usize,
    ) -> BTreeMap<SsaVarId, SsaVarId> {
        let (ssa, cache, dominators) = (self.ssa, self.cache, self.dominators);
        let leaving = ssa.block(way).map_or(0, |block| block.instructions().len());
        let mut read = BTreeMap::new();
        for op in memory.operations() {
            let MemoryOp::Load {
                location,
                dest,
                block,
                instr,
            } = op
            else {
                continue;
            };
            if *block != join {
                continue;
            }
            let Some(cell) = memory.location_id(location) else {
                continue;
            };
            // Nothing written to the cell in the join before the load.
            if memory.version_before_id(cell, join, *instr)
                != memory.version_before_id(cell, join, 0)
            {
                continue;
            }
            let Some(version) = memory.version_before_id(cell, way, leaving) else {
                continue;
            };
            if let Some(load) = cache.loads_reading(cell, version).find(|load| {
                ssa.variable(*load).is_some_and(|var| {
                    dominators.dominates(NodeId::new(var.def_site().block()), NodeId::new(way))
                })
            }) {
                read.insert(*dest, load);
            }
        }
        read
    }
}

/// What each block is entered knowing: the answers the edges entering the
/// blocks that dominate it gave.
///
/// A block entered along exactly one edge, and no way in other than that
/// edge, is entered knowing that edge's answer to its terminator's question,
/// and so is every block it dominates. A terminator or a select re-asking the
/// question there is decided:
///
/// ```text
/// b0: branch_flags v4, zero,     true -> b3, false -> b1
/// b1: branch_flags v4, not_zero, true -> b3, false -> b2
/// ```
///
/// `b1` is entered only when `zero` on `v4` is *false*, so `not_zero` on `v4`
/// is true and `b1` goes to `b3` unconditionally. That is `jz L; jnz L` with
/// a junk byte behind it, a common anti-disassembly idiom, but nothing here is
/// about obfuscation: it is the ordinary fact that a dominated block knows its
/// guard.
///
/// # Ways in that no terminator names
///
/// The predecessor relation lists terminator edges only, and two kinds of
/// block are entered without one: the function entry, which the caller
/// enters, and an exception handler or filter entry, which the runtime enters
/// on a throw. Either can hold exactly one terminator predecessor and still be
/// reached with the guard's condition in any state, so neither learns from its
/// edge. In a function with such entries the dominance the terminators give
/// is not the dominance a throw keeps, so a block learns from its own edge
/// alone there.
pub struct GuardFacts {
    /// The facts each block is entered knowing beyond those of the blocks it
    /// inherits from, by block index: a question and its answer.
    entered: Vec<Vec<(Question, bool)>>,
    /// The nearest block whose facts each block inherits that is entered
    /// knowing any: its nearest such dominator, where facts pass down
    /// dominance, and `None` past the last.
    inherits: Vec<Option<usize>>,
    /// What each block's own entered facts prove equal to a number, by block
    /// index ([`GuardReader::equalities`]).
    equalities: Vec<BTreeMap<SsaVarId, Equality>>,
    /// The nearest block whose facts each block inherits whose own facts
    /// prove any number.
    proves: Vec<Option<usize>>,
    /// The width of the function's pointers, at which a literal of the
    /// native integer types is read.
    ptr_size: PointerSize,
}

/// What one function's guards are read from: its definitions, and the width
/// of its pointers, at which a literal of the native integer types is read.
#[derive(Clone, Copy)]
pub struct GuardReader<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Its definitions.
    cache: &'a Definitions<T>,
    /// The width of its pointers.
    ptr_size: PointerSize,
}

impl<'a, T: Target> GuardReader<'a, T> {
    /// Reads the guards of `ssa`, whose definitions are `cache` and whose
    /// pointers are `ptr_size` wide.
    #[must_use]
    pub const fn new(
        ssa: &'a SsaFunction<T>,
        cache: &'a Definitions<T>,
        ptr_size: PointerSize,
    ) -> Self {
        Self {
            ssa,
            cache,
            ptr_size,
        }
    }

    /// The answer of `from`'s two-way terminator on its edge to `to`, when
    /// its two targets differ.
    fn edge_answer(&self, from: usize, to: usize) -> Option<(Question, bool)> {
        let guard = self.ssa.block(from)?.control_terminator()?;
        let (true_target, false_target) = guard.two_way_targets()?;
        if true_target == false_target {
            return None;
        }
        let (question, takes_true) = Question::of_terminator(guard, self.cache)?;
        if true_target == to {
            Some((question, takes_true))
        } else if false_target == to {
            Some((question, !takes_true))
        } else {
            None
        }
    }

    /// Every value `facts` prove equal to a number, with the number: the
    /// first fact that proves one wins, as every fact holds at once.
    pub fn equalities(&self, facts: &[(Question, bool)]) -> BTreeMap<SsaVarId, Equality> {
        let mut equalities = BTreeMap::new();
        for equality in facts
            .iter()
            .filter_map(|(question, held)| {
                question.equality(*held, self.ssa, self.cache, self.ptr_size)
            })
            .flat_map(|equality| equality.implied(self.cache, self.ptr_size))
        {
            equalities.entry(equality.value).or_insert(equality);
        }
        equalities
    }
}

impl GuardFacts {
    /// Reads the facts of every block of `ssa`, whose pointers are
    /// `ptr_size` wide, reading `memory` where it is given (`None` reads no
    /// memory).
    ///
    /// A block is entered knowing the answer of the one edge entering it and
    /// what its dominators know. Where that edge leaves a join, it knows
    /// more: a way into the join whose facts contradict the edge's answer
    /// never takes the edge, and what every other way knows on its way into
    /// the join, its own edge's answer with it, holds past the edge. A loop
    /// broken out of past the test after it, the test entered again from the
    /// break: the edge out of the test the break never takes knows what the
    /// loop's own exit knew (`argp == 0`).
    ///
    /// With `memory`, a load the join's test reads, with nothing written to
    /// its cell in the join before it, is read on each way in as a load at or
    /// above that way of the cell in the state the way leaves it in
    /// ([`Definitions::reading`] of the same memory): the test reading `*cell`
    /// again is the break's own `*cell != 0` on the break's way.
    pub fn of<T: Target>(
        ssa: &SsaFunction<T>,
        cfg: &SsaCfg<'_, T>,
        exception_blocks: &ExceptionBlocks<'_, T>,
        cache: &Definitions<T>,
        ptr_size: PointerSize,
        memory: Option<&MemorySsa<T>>,
    ) -> Self {
        let count = ssa.block_count();
        // What the one edge entering each block answered, and where from.
        let own: Vec<Option<(usize, Question, bool)>> = (0..count)
            .map(|block| {
                if block == 0 || exception_blocks.is_runtime_entry(block) {
                    return None;
                }
                // Exactly one *edge*, not one predecessor: `block_predecessors`
                // records multiplicity, so a predecessor branching here on both
                // arms appears twice and is refused - such an edge carries no
                // answer, because both lead here.
                let [predecessor] = *cfg.block_predecessors(block) else {
                    return None;
                };
                if predecessor == block {
                    return None;
                }
                let (question, held) =
                    GuardReader::new(ssa, cache, ptr_size).edge_answer(predecessor, block)?;
                Some((predecessor, question, held))
            })
            .collect();
        let inherit = exception_blocks.is_empty();
        let dominators = compute_dominators(cfg, cfg.entry());
        let entry = cfg.entry();
        // Each block keeps what it is entered knowing beyond the blocks it
        // inherits from - its own edge's answer, and what the ways into a
        // join it leaves share that its dominators do not know - and reads
        // the rest down its chain of dominators.
        let mut known = Self {
            entered: vec![Vec::new(); count],
            inherits: vec![None; count],
            equalities: Vec::new(),
            proves: vec![None; count],
            ptr_size,
        };
        let mut done = vec![false; count];
        let order = reverse_postorder(cfg, entry);
        for &node in &order {
            let block = node.index();
            if block >= count {
                continue;
            }
            if !dominators.dominates(entry, node) {
                if let Some(flag) = done.get_mut(block) {
                    *flag = true;
                }
                continue;
            }
            let parent = match dominators.immediate_dominator(node) {
                Some(next) if inherit && next != node => Some(next.index()),
                _ => None,
            };
            let mut facts: Vec<(Question, bool)> = Vec::new();
            if let Some((predecessor, question, held)) = own.get(block).copied().flatten() {
                facts.push((question, held));
                if inherit {
                    let ways = Ways {
                        ssa,
                        cfg,
                        cache,
                        ptr_size,
                        memory,
                        dominators: &dominators,
                    };
                    let shared = ways.shared(&known, &done, (predecessor, question, held));
                    if !shared.is_empty() {
                        // A fact the chain or this block already holds is not
                        // held twice: kept, each join's share would hold its
                        // dominators' facts again, doubling down a run of joins.
                        let mut seen: FxHashSet<(Question, bool)> =
                            parent.into_iter().flat_map(|up| known.facts(up)).collect();
                        seen.insert((question, held));
                        facts.extend(shared.into_iter().filter(|fact| seen.insert(*fact)));
                    }
                }
            }
            let inherited = parent.and_then(|up| {
                if known.entered.get(up).is_some_and(|facts| !facts.is_empty()) {
                    Some(up)
                } else {
                    known.inherits.get(up).copied().flatten()
                }
            });
            if let Some(slot) = known.inherits.get_mut(block) {
                *slot = inherited;
            }
            if let Some(slot) = known.entered.get_mut(block) {
                *slot = facts;
            }
            if let Some(flag) = done.get_mut(block) {
                *flag = true;
            }
        }
        known.equalities = known
            .entered
            .iter()
            .map(|facts| GuardReader::new(ssa, cache, ptr_size).equalities(facts))
            .collect();
        for node in order {
            let block = node.index();
            let proves = known.inherits.get(block).copied().flatten().and_then(|up| {
                if known
                    .equalities
                    .get(up)
                    .is_some_and(|proved| !proved.is_empty())
                {
                    Some(up)
                } else {
                    known.proves.get(up).copied().flatten()
                }
            });
            if let Some(slot) = known.proves.get_mut(block) {
                *slot = proves;
            }
        }
        known
    }

    /// The facts `block` is entered knowing, its own first and then those of
    /// each block it inherits from, nearest first.
    fn facts(&self, block: usize) -> impl Iterator<Item = (Question, bool)> + '_ {
        let mut next = Some(block);
        iter::from_fn(move || {
            let at = next?;
            next = self.inherits.get(at).copied().flatten();
            self.entered.get(at)
        })
        .flatten()
        .copied()
    }

    /// What the facts `block` is entered knowing prove equal to a number,
    /// then what `edge` proves beneath them.
    fn proved<'a>(
        &'a self,
        block: usize,
        edge: Option<&'a BTreeMap<SsaVarId, Equality>>,
    ) -> Proved<'a> {
        let mut layers = Vec::new();
        let mut next = Some(block);
        while let Some(at) = next {
            if let Some(proved) = self.equalities.get(at).filter(|proved| !proved.is_empty()) {
                layers.push(proved);
            }
            next = self.proves.get(at).copied().flatten();
        }
        layers.extend(edge);
        Proved { layers }
    }

    /// What `asked` comes to on entry to `block`, where one of its facts
    /// decides it.
    pub fn answer<T: Target>(
        &self,
        block: usize,
        asked: Question,
        cache: &Definitions<T>,
    ) -> Option<bool> {
        self.facts(block)
            .find_map(|(known, held)| known.decides(held, asked, cache, self.ptr_size))
            .or_else(|| {
                self.proved(block, None)
                    .evaluated(asked, cache, self.ptr_size)
            })
    }

    /// Whether a fact `block` is entered knowing asks `asked` or its
    /// negation ([`Question::relates`]), rather than deciding it through a
    /// value proved a number.
    pub fn asks<T: Target>(&self, block: usize, asked: Question, cache: &Definitions<T>) -> bool {
        self.facts(block)
            .any(|(known, _)| known.relates(asked, cache, self.ptr_size).is_some())
    }

    /// The value `phi`, a phi of block `at`, holds on the way into `block`:
    /// an edge into `at` on which a fact `block` is entered knowing comes
    /// the other way ([`Self::edge_answer`]) leads to no path into `block`,
    /// and the edges left bring one value. `cdcmd` in dash enters its
    /// copied `strtok` from a block that re-asks `if (c)` past a join of
    /// `lim` and `lim + 1`, the `+ 1` on the way where `c` held: on the way
    /// where it did not, the join is `lim`. `None` where the edges left bring
    /// more than one value, or none.
    ///
    /// A fact names one value of each operand where every path into `block`
    /// enters `at` once, after the operands are defined: the caller's to
    /// establish, as `at` dominating `block` and lying on no cycle.
    pub fn phi_on_way<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        block: usize,
        at: usize,
        phi: &PhiNode,
        cache: &Definitions<T>,
    ) -> Option<SsaVarId> {
        let known: Vec<(Question, bool)> = self.facts(block).collect();
        let mut taken = None;
        for operand in phi.operands() {
            let from = operand.predecessor();
            if known.iter().any(|&(question, held)| {
                self.edge_answer(ssa, from, at, question, cache) == Some(!held)
            }) {
                continue;
            }
            match taken {
                None => taken = Some(operand.value()),
                Some(value) if value == operand.value() => {}
                Some(_) => return None,
            }
        }
        taken
    }

    /// What `asked` comes to on the edge from `from` into `to`: the answer of
    /// `from`'s two-way terminator that leads there, when its two targets
    /// differ, else what `from` holds on entry.
    pub fn edge_answer<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        from: usize,
        to: usize,
        asked: Question,
        cache: &Definitions<T>,
    ) -> Option<bool> {
        if let Some(terminator) = ssa.block(from).and_then(SsaBlock::control_terminator)
            && let Some((true_target, false_target)) = terminator.two_way_targets()
            && true_target != false_target
            && let Some((question, takes_true)) = Question::of_terminator(terminator, cache)
        {
            let held = if true_target == to {
                Some(takes_true)
            } else if false_target == to {
                Some(!takes_true)
            } else {
                None
            };
            if let Some(answer) =
                held.and_then(|held| question.decides(held, asked, cache, self.ptr_size))
            {
                return Some(answer);
            }
            if let Some(answer) = self.answer(from, asked, cache) {
                return Some(answer);
            }
            // The edge's own answer may prove a value equal to a number.
            let proved = question.equality(held?, ssa, cache, self.ptr_size)?;
            let mut edge = BTreeMap::new();
            for equality in proved.implied(cache, self.ptr_size) {
                edge.entry(equality.value).or_insert(equality);
            }
            return self
                .proved(from, Some(&edge))
                .evaluated(asked, cache, self.ptr_size);
        }
        self.answer(from, asked, cache)
    }

    /// The phi operands that read a select whose condition the edge they come
    /// in on decides, each as `(block, phi, predecessor, chosen)`: on that
    /// edge the select is the arm its answer chooses.
    ///
    /// A predicated write followed by a branch on its own condition is the
    /// shape: Thumb's `it eq; moveq r0, #23; beq L` lifts as
    /// `r0 = select(eq, 23, r0)`, and the edge to `L` is taken only where `eq`
    /// holds, so `L` is entered with `23`. Read whole, the phi at `L` reads
    /// the `r0` the function was entered with on an edge that never carries
    /// it, which makes it a parameter no caller passes. The chosen arm is an
    /// operand of the select, so it is available wherever the select is.
    /// Only terminator edges answer a question: a throw may leave a block at
    /// any point.
    pub fn decided_phi_operands<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        cache: &Definitions<T>,
    ) -> Vec<(usize, usize, usize, SsaVarId)> {
        let mut decided = Vec::new();
        for (block_idx, block) in ssa.iter_blocks() {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                for operand in phi.operands() {
                    if operand.edge() != PhiEdge::Normal {
                        continue;
                    }
                    let Some(SsaOp::Select {
                        condition,
                        true_val,
                        false_val,
                        ..
                    }) = cache.index.def_op(operand.value())
                    else {
                        continue;
                    };
                    let (question, holds) = Question::of_value(*condition, cache);
                    let Some(answer) =
                        self.edge_answer(ssa, operand.predecessor(), block_idx, question, cache)
                    else {
                        continue;
                    };
                    let chosen = if answer == holds {
                        *true_val
                    } else {
                        *false_val
                    };
                    decided.push((block_idx, phi_idx, operand.predecessor(), chosen));
                }
            }
        }
        decided
    }

    /// The target `block`'s terminator takes, when a fact decides it, and the
    /// answer it takes it on.
    pub fn decided_branch<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        block: usize,
        cache: &Definitions<T>,
    ) -> Option<(usize, bool)> {
        let terminator = ssa.block(block)?.control_terminator()?;
        let (true_target, false_target) = terminator.two_way_targets()?;
        let (question, takes_true) = Question::of_terminator(terminator, cache)?;
        let holds = self.answer(block, question, cache)? == takes_true;
        Some((if holds { true_target } else { false_target }, holds))
    }

    /// The reads of a decided value in the blocks every way into which
    /// decided it, each as `(block, instruction, value, constant)`: the
    /// constant is what every read of the value there reads.
    ///
    /// Two kinds of value are decided. A boolean - what a comparison or
    /// `bool.not` defines - whose answer is known on entry is that answer.
    /// Selects and two-way terminators are decided whole on it
    /// ([`Self::decided_selects`], [`Self::decided_branch`]); this is every
    /// other reader, `c & x` or a store of `c`. A value only known nonzero is
    /// no boolean: an integer the guard tested may be `5`, which `true` is not.
    ///
    /// And a value a fact proves equal to a number ([`Question::equality`],
    /// [`Equality::implied`]) is that number, at the value's own type, in
    /// every reader but a two-way terminator, which asks a question and is
    /// decided whole. A compiler reuses the register a null pointer was just
    /// tested in for the integer `0` it passes next, and read as the pointer
    /// the value names, the `0` would tell the callee its parameter is one.
    ///
    /// A phi operand is no such reader, and neither is a copy or a
    /// conversion ([`Equality::passes_on`]): each passes the value on to where
    /// it joins, often from a block that holds nothing else, and the empty
    /// block is what the structurer joins two tests over with `&&`. With the
    /// constant stated there, `x != 0 && *x == 0` reads as two `if`s and an
    /// `else { x = 0; }`. What reads the value passed on reads the number it
    /// carries ([`Equality::read`]).
    pub fn decided_reads<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        cache: &Definitions<T>,
    ) -> Vec<(usize, usize, SsaVarId, ConstValue<T>)> {
        let mut decided = Vec::new();
        for (block_idx, block) in ssa.iter_blocks() {
            if self.facts(block_idx).next().is_none() {
                continue;
            }
            let proved = self.proved(block_idx, None);
            for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                let op = instruction.op();
                if op.two_way_targets().is_some() {
                    continue;
                }
                let select = matches!(op, SsaOp::Select { .. });
                let mut seen: Vec<SsaVarId> = Vec::new();
                for used in op.uses() {
                    if seen.contains(&used) {
                        continue;
                    }
                    seen.push(used);
                    let constant = if cache.is_boolean(used) {
                        if select {
                            continue;
                        }
                        let (question, holds) = Question::of_value(used, cache);
                        self.answer(block_idx, question, cache)
                            .map(|answer| ConstValue::from_bool(answer == holds))
                    } else if Equality::passes_on(op) {
                        continue;
                    } else {
                        Equality::read(&proved, used, cache).and_then(|equality| {
                            equality.literal_of(ssa.variable(used)?.var_type())
                        })
                    };
                    if let Some(constant) = constant {
                        decided.push((block_idx, instr_idx, used, constant));
                    }
                }
            }
        }
        decided
    }

    /// The reads of a select in the blocks every way into which decided its
    /// condition, each as `(block, instruction, select, chosen)`: the read
    /// there reads the arm the answer chooses.
    ///
    /// A predicated write a branch on its own condition follows is the shape:
    /// Thumb's `it ne; movne r4, sp; beq L` lifts as `r4 = select(ne, sp,
    /// r4)`, and the block the branch falls into is entered only where `ne`
    /// holds, so every read of `r4` there reads `sp`. Read whole, the select
    /// hands the callee the `r4` the function was entered with on a path that
    /// never carries it. Both arms are operands of the select, so each is
    /// available wherever the select is, and so in every block it reaches.
    ///
    /// A select its own block decides, or one this read alone reads, is
    /// [`Self::decided_selects`]'s, which decides the select whole, and a phi
    /// operand is [`Self::decided_phi_operands`]'s, which the edge it comes
    /// in on decides.
    pub fn decided_select_reads<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        cache: &Definitions<T>,
    ) -> Vec<(usize, usize, SsaVarId, SsaVarId)> {
        let mut decided = Vec::new();
        for (block_idx, block) in ssa.iter_blocks() {
            if self.facts(block_idx).next().is_none() {
                continue;
            }
            for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                let mut seen: Vec<SsaVarId> = Vec::new();
                for used in instruction.op().uses() {
                    if seen.contains(&used) {
                        continue;
                    }
                    seen.push(used);
                    let Some((
                        at_block,
                        _,
                        SsaOp::Select {
                            condition,
                            true_val,
                            false_val,
                            ..
                        },
                    )) = cache.index.full_definition(used)
                    else {
                        continue;
                    };
                    // One its own block decides, or this read alone, is
                    // decided whole ([`Self::decided_selects`]).
                    if at_block == block_idx || cache.index.use_count(used) == 1 {
                        continue;
                    }
                    let (question, holds) = Question::of_value(*condition, cache);
                    let Some(answer) = self.answer(block_idx, question, cache) else {
                        continue;
                    };
                    let chosen = if answer == holds {
                        *true_val
                    } else {
                        *false_val
                    };
                    decided.push((block_idx, instr_idx, used, chosen));
                }
            }
        }
        decided
    }

    /// The selects at a join whose condition every way into the join decides,
    /// though not all alike: each is the phi of the arms the ways choose
    /// ([`JoinSelect`]).
    ///
    /// Thumb's `cmp r0, #0; ite lt; movlt r0, #0; ldrge r0, [r1, r0, lsl #2]`
    /// loads from an arm of its own, entered where `lt` fails
    /// ([`Self::decided_select_reads`]'s shape), and joins the way past it
    /// where `lt` holds: the join holds `r0 = select(lt, 0, phi(load, r0))`,
    /// and each way in chooses one arm, the load where `lt` failed and `0`
    /// where it held. A chosen arm that is a phi of the join is the operand
    /// its edge brings; a constant the join defines is stated again at the end
    /// of the way in; any other value is defined before the join, so it is
    /// available at the end of each way into it. One the join computes is
    /// not, and leaves the select as it is.
    pub fn decided_join_selects<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        cfg: &SsaCfg<'_, T>,
        cache: &Definitions<T>,
    ) -> Vec<JoinSelect> {
        let mut decided = Vec::new();
        for (block_idx, block) in ssa.iter_blocks() {
            let preds = cfg.block_predecessors(block_idx);
            let distinct: BTreeSet<usize> = preds.iter().copied().collect();
            if preds.len() < 2 || distinct.len() != preds.len() {
                continue;
            }
            // Only edges the predecessors' terminators state answer: a throw
            // may leave a block at any point.
            let stated = preds.iter().all(|&pred| {
                ssa.block(pred)
                    .and_then(SsaBlock::control_terminator)
                    .is_some_and(|terminator| terminator.successors().contains(&block_idx))
            });
            if !stated {
                continue;
            }
            for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                let SsaOp::Select {
                    dest,
                    condition,
                    true_val,
                    false_val,
                } = instruction.op()
                else {
                    continue;
                };
                let (question, holds) = Question::of_value(*condition, cache);
                if self.answer(block_idx, question, cache).is_some() {
                    continue;
                }
                let arm = |pred: usize| {
                    let answer = self.edge_answer(ssa, pred, block_idx, question, cache)?;
                    let chosen = if answer == holds {
                        *true_val
                    } else {
                        *false_val
                    };
                    let site = ssa.variable(chosen)?.def_site();
                    if site.block() != block_idx {
                        return Some(chosen);
                    }
                    if site.index().is_some() {
                        // A constant the join defines can be stated again on
                        // the way in.
                        return matches!(cache.get_definition(chosen), Some(SsaOp::Const { .. }))
                            .then_some(chosen);
                    }
                    let joined = block.phi_nodes().iter().find(|phi| phi.result() == chosen);
                    match joined {
                        Some(phi) => {
                            let mut brought = phi
                                .operands()
                                .iter()
                                .filter(|operand| operand.predecessor() == pred);
                            let operand = brought.next()?;
                            (brought.next().is_none() && operand.edge() == PhiEdge::Normal)
                                .then(|| operand.value())
                        }
                        // The function's entry defines it, ahead of every way in.
                        None => Some(chosen),
                    }
                };
                let arms: Option<Vec<(usize, SsaVarId)>> = preds
                    .iter()
                    .map(|&pred| arm(pred).map(|value| (pred, value)))
                    .collect();
                if let Some(arms) = arms {
                    decided.push(JoinSelect {
                        block: block_idx,
                        instruction: instr_idx,
                        dest: *dest,
                        arms,
                    });
                }
            }
        }
        decided
    }

    /// The phi operands a fact of the edge they come in on proves equal to a
    /// number, each as `(block, phi, predecessor, equality)`: on that edge
    /// the operand is the number ([`Equality::read`]). The facts of an edge
    /// are the predecessor's and its own terminator's answer for the edge;
    /// only terminator edges carry them, since a throw may leave a block at
    /// any point.
    ///
    /// What the phi joins on that edge is the number, whatever the operand's
    /// value is elsewhere: `p == 0 ? p : strchr(p, c) + 1` joins the null
    /// pointer and a string, and the record `p` points at is no part of the
    /// result.
    pub fn phi_operand_equalities<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        definitions: &Definitions<T>,
    ) -> Vec<(usize, usize, usize, Equality)> {
        let mut decided = Vec::new();
        for (block_idx, block) in ssa.iter_blocks() {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                for operand in phi.operands() {
                    if operand.edge() != PhiEdge::Normal {
                        continue;
                    }
                    let predecessor = operand.predecessor();
                    let mut facts: Vec<(Question, bool)> = self.facts(predecessor).collect();
                    if let Some(terminator) = ssa
                        .block(predecessor)
                        .and_then(SsaBlock::control_terminator)
                        && let Some((true_target, false_target)) = terminator.two_way_targets()
                        && true_target != false_target
                        && let Some((question, takes_true)) =
                            Question::of_terminator(terminator, definitions)
                    {
                        if true_target == block_idx {
                            facts.push((question, takes_true));
                        } else if false_target == block_idx {
                            facts.push((question, !takes_true));
                        }
                    }
                    if facts.is_empty() {
                        continue;
                    }
                    let equalities =
                        GuardReader::new(ssa, definitions, self.ptr_size).equalities(&facts);
                    if let Some(equality) =
                        Equality::read(&Proved::of(&equalities), operand.value(), definitions)
                    {
                        decided.push((block_idx, phi_idx, predecessor, equality));
                    }
                }
            }
        }
        decided
    }

    /// The selects of `ssa` whose condition is decided where they are
    /// evaluated, each as `(block, instruction, dest, chosen)`: by a fact of
    /// its block, by the arm of another select it computes for alone, or by
    /// the facts of the one block whose instruction it computes for alone.
    ///
    /// The arm of `select c, t, f` is evaluated only for `c`, so whatever
    /// computes `t` for that arm and nothing else asks `c` knowing it true:
    /// `select c, (select c, x, y) + k, z` adds `k` to `x`. So does what an
    /// instruction reads in a block entered only where `c` holds: Thumb's
    /// `itttt eq; addeq r3, r0, #0x68; ...; addeq r3, sp; strbeq r2, [r3,
    /// #-86]` stores from an arm of its own ([`Self::decided_select_reads`]),
    /// and the `r3 + sp` it stores through reads the first `addeq`'s select
    /// for that store alone. The computation is followed down while each
    /// value it reads is read by it alone, and only through operations
    /// without effect.
    pub fn decided_selects<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        cache: &Definitions<T>,
    ) -> Vec<(usize, usize, SsaVarId, SsaVarId)> {
        let mut decided: Vec<(usize, usize, SsaVarId, SsaVarId)> = Vec::new();
        let mut seen: BTreeSet<(usize, usize)> = BTreeSet::new();
        for (block_idx, block) in ssa.iter_blocks() {
            for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                let SsaOp::Select {
                    dest,
                    condition,
                    true_val,
                    false_val,
                } = instruction.op()
                else {
                    continue;
                };
                let (question, holds) = Question::of_value(*condition, cache);
                if let Some(answer) = self.answer(block_idx, question, cache) {
                    let chosen = if answer == holds {
                        *true_val
                    } else {
                        *false_val
                    };
                    if seen.insert((block_idx, instr_idx)) {
                        decided.push((block_idx, instr_idx, *dest, chosen));
                    }
                    continue;
                }
                // Each arm knows the question's answer that selects it.
                for (arm, answer) in [(*true_val, holds), (*false_val, !holds)] {
                    Self::decide_alone(
                        arm,
                        |asked| question.decides(answer, asked, cache, self.ptr_size),
                        cache,
                        &mut seen,
                        &mut decided,
                    );
                }
            }
        }
        // What an instruction reads in a block that knows something knows
        // it too, where the instruction reads it alone.
        for (block_idx, block) in ssa.iter_blocks() {
            if self.facts(block_idx).next().is_none() {
                continue;
            }
            for instruction in block.instructions() {
                for used in instruction.op().uses() {
                    Self::decide_alone(
                        used,
                        |asked| self.answer(block_idx, asked, cache),
                        cache,
                        &mut seen,
                        &mut decided,
                    );
                }
            }
        }
        decided
    }

    /// Follows the computation of `start`, read by one reader alone, and
    /// records into `decided` each select on the way whose condition `known`
    /// answers, as [`Self::decided_selects`] states them; `seen` holds the
    /// selects recorded already.
    ///
    /// The computation is followed down while each value it reads is read by
    /// it alone, and only through operations without effect: a value another
    /// reader uses is computed for both, and knows nothing, and a load or a
    /// division runs whatever the condition is. A decided select is followed
    /// on into the arm it chooses.
    fn decide_alone<T: Target>(
        start: SsaVarId,
        known: impl Fn(Question) -> Option<bool>,
        cache: &Definitions<T>,
        seen: &mut BTreeSet<(usize, usize)>,
        decided: &mut Vec<(usize, usize, SsaVarId, SsaVarId)>,
    ) {
        /// How many values a computation is followed through.
        const WALK: usize = 64;
        let mut pending = vec![start];
        let mut walked = 0usize;
        while let Some(value) = pending.pop() {
            walked = walked.saturating_add(1);
            if walked > WALK || cache.index.use_count(value) != 1 {
                continue;
            }
            let Some((at_block, at_instr, op)) = cache.index.full_definition(value) else {
                continue;
            };
            if let SsaOp::Select {
                dest,
                condition,
                true_val,
                false_val,
            } = op
            {
                let (question, holds) = Question::of_value(*condition, cache);
                if let Some(answer) = known(question) {
                    let chosen = if answer == holds {
                        *true_val
                    } else {
                        *false_val
                    };
                    if seen.insert((at_block, at_instr)) {
                        decided.push((at_block, at_instr, *dest, chosen));
                    }
                    pending.push(chosen);
                    continue;
                }
            }
            // Reading `p` where only `q` was valid would change what a load
            // does.
            if op.is_pure() {
                pending.extend(op.uses());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            instruction::SsaInstruction,
            phi::{PhiNode, PhiOperand},
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// **A value zero-extended and compared with a literal asks what the
    /// value itself does.** `c == 0` on a byte and `zext(c) != 0` on the
    /// word are one question, the other way round, as `zext(c) < 5` unsigned
    /// is `c < 5`. The controls relate nothing: a literal past the byte
    /// (`zext(c) == 256` is never true, `c == 0` sometimes), a signed order
    /// (`0x80` widened is `128`, read signed as a byte it is `-128`), and a
    /// sign extension, which this reads no further.
    #[test]
    fn a_widened_value_asks_what_the_value_asks() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 8);
        let c = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::U8,
        );
        let mut defined = Vec::new();
        let mut local = |ssa: &mut SsaFunction<MockTarget>,
                         ty: MockType,
                         op: &dyn Fn(SsaVarId) -> SsaOp<MockTarget>| {
            let index = defined.len();
            let var = ssa.create_variable(
                VariableOrigin::Local(u16::try_from(index).unwrap_or(u16::MAX)),
                0,
                DefSite::instruction(0, index),
                ty,
            );
            defined.push(SsaInstruction::synthetic(op(var)));
            var
        };
        let widened = local(&mut ssa, MockType::I32, &|dest| SsaOp::IntConv {
            dest,
            operand: c,
            target: MockType::I32,
            overflow_check: false,
            unsigned: true,
        });
        let extended = local(&mut ssa, MockType::I32, &|dest| SsaOp::IntConv {
            dest,
            operand: c,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        });
        let literal = |value: ConstValue<MockTarget>| {
            move |dest| SsaOp::Const {
                dest,
                value: value.clone(),
            }
        };
        let zero_8 = local(&mut ssa, MockType::U8, &literal(ConstValue::U8(0)));
        let five_8 = local(&mut ssa, MockType::U8, &literal(ConstValue::U8(5)));
        let zero_32 = local(&mut ssa, MockType::I32, &literal(ConstValue::I32(0)));
        let five_32 = local(&mut ssa, MockType::I32, &literal(ConstValue::I32(5)));
        let wide_32 = local(&mut ssa, MockType::I32, &literal(ConstValue::I32(256)));
        let mut block = SsaBlock::new(0);
        for row in defined {
            block.add_instruction(row);
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();
        let cache = Definitions::of(&ssa);
        let ptr_size = PointerSize::Bit64;
        let relates = |known: Question, asked: Question| known.relates(asked, &cache, ptr_size);
        let byte_zero = Question::compare(c, zero_8, CmpKind::Eq, false);
        assert_eq!(
            relates(
                byte_zero,
                Question::compare(widened, zero_32, CmpKind::Ne, false)
            ),
            Some(false)
        );
        assert_eq!(
            relates(
                byte_zero,
                Question::compare(zero_32, widened, CmpKind::Eq, false)
            ),
            Some(true)
        );
        assert_eq!(
            relates(
                Question::compare(c, five_8, CmpKind::Lt, true),
                Question::compare(widened, five_32, CmpKind::Lt, true)
            ),
            Some(true)
        );
        assert_eq!(
            relates(
                byte_zero,
                Question::compare(widened, wide_32, CmpKind::Eq, false)
            ),
            None
        );
        assert_eq!(
            relates(
                Question::compare(c, five_8, CmpKind::Lt, false),
                Question::compare(widened, five_32, CmpKind::Lt, false)
            ),
            None
        );
        assert_eq!(
            relates(
                byte_zero,
                Question::compare(extended, zero_32, CmpKind::Eq, false)
            ),
            None
        );
    }

    /// **An edge out of a join knows what the ways it can come from know.**
    /// `b0` tests `x == 0`; `b1`, where it holds, goes on to the join `b3`
    /// only where `y == 7`, and `b2`, where it fails, goes there straight.
    /// `b3` asks `x == 0` again: its edge to `b4` is never taken from `b2`, so
    /// `b4` knows `y == 7`, which only `b1`'s way into the join knew. The
    /// edge to `b5` is never taken from `b1`, and `b2`'s way knows nothing of
    /// `y`, so `b5` knows nothing of it either.
    #[test]
    fn an_edge_out_of_a_join_knows_what_its_ways_in_know() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
        let local = |ssa: &mut SsaFunction<MockTarget>, index: u16, block: usize, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(block, at),
                MockType::I32,
            )
        };
        let x = local(&mut ssa, 0, 0, 0);
        let y = local(&mut ssa, 1, 0, 1);
        let zero = local(&mut ssa, 2, 0, 2);
        let seven = local(&mut ssa, 3, 1, 0);
        let compare = |left: SsaVarId, right: SsaVarId, to: (usize, usize)| {
            SsaInstruction::synthetic(SsaOp::BranchCmp {
                left,
                right,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: to.0,
                false_target: to.1,
            })
        };
        let rows: Vec<Vec<SsaInstruction<MockTarget>>> = vec![
            vec![
                SsaInstruction::synthetic(SsaOp::LoadArg {
                    dest: x,
                    arg_index: 0,
                }),
                SsaInstruction::synthetic(SsaOp::LoadArg {
                    dest: y,
                    arg_index: 1,
                }),
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: zero,
                    value: ConstValue::I32(0),
                }),
                compare(x, zero, (1, 2)),
            ],
            vec![
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: seven,
                    value: ConstValue::I32(7),
                }),
                compare(y, seven, (3, 5)),
            ],
            vec![SsaInstruction::synthetic(SsaOp::Jump { target: 3 })],
            vec![compare(x, zero, (4, 5))],
            vec![SsaInstruction::synthetic(SsaOp::returns(None))],
            vec![SsaInstruction::synthetic(SsaOp::returns(None))],
        ];
        for (index, rows) in rows.into_iter().enumerate() {
            let mut block = SsaBlock::new(index);
            for row in rows {
                block.add_instruction(row);
            }
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let definitions = Definitions::of(&ssa);
        let cfg = SsaCfg::from_ssa(&ssa);
        let exception_blocks = ssa.exception_blocks();
        let facts = GuardFacts::of(
            &ssa,
            &cfg,
            &exception_blocks,
            &definitions,
            PointerSize::Bit32,
            None,
        );
        let sevens = Question::Compare {
            left: y,
            right: seven,
            cmp: CmpKind::Eq,
            unsigned: false,
        };
        assert_eq!(facts.answer(4, sevens, &definitions), Some(true));
        assert_eq!(facts.answer(5, sevens, &definitions), None);
    }

    /// **A run of joins holds each fact once.** Each of `RUN` sections tests
    /// `x == 0` at its head, goes on to its join where `y == 7` holds and
    /// straight there where `x == 0` fails, and leaves the join for the next
    /// head where `x == 0` holds again: that edge is never taken from the
    /// failing way, so the next head knows `y == 7` from the other way.
    /// Every head knows both, and the blocks hold a number of facts that grows
    /// with the run, not doubling at each join as a share holding its
    /// dominators' facts again would.
    #[test]
    fn a_run_of_joins_holds_each_fact_once() {
        const RUN: usize = 16;
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
        let local = |ssa: &mut SsaFunction<MockTarget>, index: u16, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, at),
                MockType::I32,
            )
        };
        let x = local(&mut ssa, 0, 0);
        let y = local(&mut ssa, 1, 1);
        let zero = local(&mut ssa, 2, 2);
        let seven = local(&mut ssa, 3, 3);
        let compare = |left: SsaVarId, right: SsaVarId, to: (usize, usize)| {
            SsaInstruction::synthetic(SsaOp::BranchCmp {
                left,
                right,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: to.0,
                false_target: to.1,
            })
        };
        let last = 4 * RUN;
        let exit = last + 1;
        let mut rows: Vec<Vec<SsaInstruction<MockTarget>>> = Vec::new();
        for section in 0..RUN {
            let head = 4 * section;
            let mut first = Vec::new();
            if section == 0 {
                first.extend([
                    SsaInstruction::synthetic(SsaOp::LoadArg {
                        dest: x,
                        arg_index: 0,
                    }),
                    SsaInstruction::synthetic(SsaOp::LoadArg {
                        dest: y,
                        arg_index: 1,
                    }),
                    SsaInstruction::synthetic(SsaOp::Const {
                        dest: zero,
                        value: ConstValue::I32(0),
                    }),
                    SsaInstruction::synthetic(SsaOp::Const {
                        dest: seven,
                        value: ConstValue::I32(7),
                    }),
                ]);
            }
            first.push(compare(x, zero, (head + 1, head + 2)));
            rows.push(first);
            rows.push(vec![compare(y, seven, (head + 3, exit))]);
            rows.push(vec![SsaInstruction::synthetic(SsaOp::Jump {
                target: head + 3,
            })]);
            rows.push(vec![compare(x, zero, (head + 4, exit))]);
        }
        rows.push(vec![SsaInstruction::synthetic(SsaOp::returns(None))]);
        rows.push(vec![SsaInstruction::synthetic(SsaOp::returns(None))]);
        for (index, rows) in rows.into_iter().enumerate() {
            let mut block = SsaBlock::new(index);
            for row in rows {
                block.add_instruction(row);
            }
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let definitions = Definitions::of(&ssa);
        let cfg = SsaCfg::from_ssa(&ssa);
        let exception_blocks = ssa.exception_blocks();
        let facts = GuardFacts::of(
            &ssa,
            &cfg,
            &exception_blocks,
            &definitions,
            PointerSize::Bit32,
            None,
        );
        let ask = |right: SsaVarId, left: SsaVarId| Question::Compare {
            left,
            right,
            cmp: CmpKind::Eq,
            unsigned: false,
        };
        for section in 1..=RUN {
            let head = 4 * section;
            assert_eq!(facts.answer(head, ask(zero, x), &definitions), Some(true));
            assert_eq!(facts.answer(head, ask(seven, y), &definitions), Some(true));
        }
        let held: usize = facts.entered.iter().map(Vec::len).sum();
        assert!(held <= 4 * ssa.block_count(), "{held} facts held");
    }

    /// **A phi operand its edge proves zero is zero there.** `b0` tests
    /// `x == 0` and goes to `b2` on it, where a phi takes `x` from `b0` and
    /// `y` from `b1`: on the edge from `b0` the operand is zero, which is the
    /// null pointer a `return p == 0 ? p : q` joins. The operand from `b1`,
    /// where nothing is known of `y`, is proved nothing.
    #[test]
    fn a_phi_operand_its_edge_proves_zero_is_zero_there() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
        let local = |ssa: &mut SsaFunction<MockTarget>, index: u16, block: usize, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(block, at),
                MockType::I32,
            )
        };
        let x = local(&mut ssa, 0, 0, 0);
        let zero = local(&mut ssa, 1, 0, 1);
        let y = local(&mut ssa, 2, 1, 0);
        let joined = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(2), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: x,
            arg_index: 0,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: x,
            right: zero,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 2,
            false_target: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: y,
            arg_index: 1,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(joined, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(x, 0));
        phi.add_operand(PhiOperand::new(y, 1));
        b2.add_phi(phi);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(joined))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let definitions = Definitions::of(&ssa);
        let cfg = SsaCfg::from_ssa(&ssa);
        let exception_blocks = ssa.exception_blocks();
        let facts = GuardFacts::of(
            &ssa,
            &cfg,
            &exception_blocks,
            &definitions,
            PointerSize::Bit32,
            None,
        );
        let decided: Vec<(usize, usize, usize, SsaVarId, bool)> = facts
            .phi_operand_equalities(&ssa, &definitions)
            .into_iter()
            .map(|(block, phi, predecessor, equality)| {
                (
                    block,
                    phi,
                    predecessor,
                    equality.value(),
                    equality.is_zero(),
                )
            })
            .collect();
        assert_eq!(decided, vec![(2, 0, 0, x, true)]);
    }

    /// What `b1`'s `x != limit` comes to, entered from `b0` where `x == 3`
    /// failed: `b0` and `b1` each define their literal, `3` in `b0` and
    /// `limit` in `b1`.
    fn decided_by_another_literal(limit: i32) -> Option<(usize, bool)> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 8);
        let local = |ssa: &mut SsaFunction<MockTarget>, index: u16, block: usize, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(block, at),
                MockType::I32,
            )
        };
        let x = local(&mut ssa, 0, 0, 0);
        let three = local(&mut ssa, 1, 0, 1);
        let again = local(&mut ssa, 2, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: x,
            arg_index: 0,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: three,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: x,
            right: three,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 2,
            false_target: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: again,
            value: ConstValue::I32(limit),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: x,
            right: again,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 3,
            false_target: 2,
        }));
        ssa.add_block(b1);
        for index in [2, 3] {
            let mut block = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let definitions = Definitions::of(&ssa);
        let cfg = SsaCfg::from_ssa(&ssa);
        let exception_blocks = ssa.exception_blocks();
        GuardFacts::of(
            &ssa,
            &cfg,
            &exception_blocks,
            &definitions,
            PointerSize::Bit32,
            None,
        )
        .decided_branch(&ssa, 1, &definitions)
    }

    /// **Two literals alike are one operand.** `b0` sends `x == 3` away, so
    /// `b1` is entered knowing `x != 3`, and its own `x != 3` - against a `3`
    /// it defines itself - holds: it goes to `b3`.
    #[test]
    fn a_test_against_a_literal_alike_is_decided() {
        assert_eq!(decided_by_another_literal(3), Some((3, true)));
    }

    /// What `b1`'s `x <cmp> second` comes to, entered from `b0` where
    /// `x == first` held: `b1` goes to `b3` where its test holds and to `b2`
    /// otherwise.
    fn decided_from_equality(
        first: i32,
        cmp: CmpKind,
        second: i32,
        unsigned: bool,
    ) -> Option<(usize, bool)> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 8);
        let local = |ssa: &mut SsaFunction<MockTarget>, index: u16, block: usize, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(block, at),
                MockType::I32,
            )
        };
        let x = local(&mut ssa, 0, 0, 0);
        let known = local(&mut ssa, 1, 0, 1);
        let asked = local(&mut ssa, 2, 1, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: x,
            arg_index: 0,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: known,
            value: ConstValue::I32(first),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: x,
            right: known,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: asked,
            value: ConstValue::I32(second),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: x,
            right: asked,
            cmp,
            unsigned,
            true_target: 3,
            false_target: 2,
        }));
        ssa.add_block(b1);
        for index in [2, 3] {
            let mut block = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let definitions = Definitions::of(&ssa);
        let cfg = SsaCfg::from_ssa(&ssa);
        let exception_blocks = ssa.exception_blocks();
        GuardFacts::of(
            &ssa,
            &cfg,
            &exception_blocks,
            &definitions,
            PointerSize::Bit32,
            None,
        )
        .decided_branch(&ssa, 1, &definitions)
    }

    /// **A value proved a number decides every comparison of it with a
    /// literal.** Entered knowing `x == 45`, `x != 0` and `x < 50` hold and
    /// `x > 50` does not, though no test before asked any of them.
    #[test]
    fn a_value_proved_a_number_decides_its_comparisons() {
        assert_eq!(
            decided_from_equality(45, CmpKind::Ne, 0, false),
            Some((3, true))
        );
        assert_eq!(
            decided_from_equality(45, CmpKind::Lt, 50, false),
            Some((3, true))
        );
        assert_eq!(
            decided_from_equality(45, CmpKind::Gt, 50, false),
            Some((2, false))
        );
    }

    /// **The comparison is made as the test makes it.** `x == -1` holds all
    /// ones: below zero signed, above it unsigned.
    #[test]
    fn a_proved_number_compares_signed_or_unsigned_as_asked() {
        assert_eq!(
            decided_from_equality(-1, CmpKind::Lt, 0, false),
            Some((3, true))
        );
        assert_eq!(
            decided_from_equality(-1, CmpKind::Lt, 0, true),
            Some((2, false))
        );
    }

    /// What `b3`'s `flag != 0` comes to, where `flag` joins `first`, set on
    /// `b1`'s way, and `second`, set on `b2`'s, or, with `looped`, `first`
    /// alone around a loop: `b3` takes `flag` from `b0` and from itself.
    fn decided_past_a_join(first: i32, second: i32, looped: bool) -> Option<(usize, bool)> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 8);
        let local = |ssa: &mut SsaFunction<MockTarget>, index: u16, site: DefSite| {
            ssa.create_variable(VariableOrigin::Local(index), 0, site, MockType::I32)
        };
        let x = local(&mut ssa, 0, DefSite::instruction(0, 0));
        let set_first = local(&mut ssa, 1, DefSite::instruction(1, 0));
        let set_second = local(&mut ssa, 1, DefSite::instruction(2, 0));
        let flag = local(&mut ssa, 1, DefSite::phi(3));
        let zero = local(&mut ssa, 2, DefSite::instruction(3, 0));
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::LoadArg {
            dest: x,
            arg_index: 0,
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: x,
            right: x,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        for (index, dest, value) in [(1, set_first, first), (2, set_second, second)] {
            let mut block = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            }));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(block);
        }
        let mut b3 = SsaBlock::new(3);
        let mut join = PhiNode::new(flag, VariableOrigin::Local(1));
        join.add_operand(PhiOperand::new(set_first, 1));
        let other = if looped { flag } else { set_second };
        join.add_operand(PhiOperand::new(other, if looped { 3 } else { 2 }));
        b3.add_phi(join);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: flag,
            right: zero,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 5,
            false_target: if looped { 3 } else { 4 },
        }));
        ssa.add_block(b3);
        for index in [4, 5] {
            let mut block = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let definitions = Definitions::of(&ssa);
        let cfg = SsaCfg::from_ssa(&ssa);
        let exception_blocks = ssa.exception_blocks();
        GuardFacts::of(
            &ssa,
            &cfg,
            &exception_blocks,
            &definitions,
            PointerSize::Bit32,
            None,
        )
        .decided_branch(&ssa, 3, &definitions)
    }

    /// **A join of one literal is that literal.** Both ways into `b3` set
    /// the flag to `1`, each with a `1` of its own, so `flag != 0` holds:
    /// `expmeta` sets `metaflag = 1` on two paths that meet before it leaves
    /// its loop. A loop's phi of `1` and itself never holds anything else.
    #[test]
    fn a_join_of_one_literal_decides_its_comparisons() {
        assert_eq!(decided_past_a_join(1, 1, false), Some((5, true)));
        assert_eq!(decided_past_a_join(1, 1, true), Some((5, true)));
    }

    /// **A join of two literals is neither.** The control of
    /// [`a_join_of_one_literal_decides_its_comparisons`]: a flag set to `1`
    /// on one way and `0` on the other may be either.
    #[test]
    fn a_join_of_two_literals_is_not_decided() {
        assert_eq!(decided_past_a_join(1, 0, false), None);
    }

    /// **A literal of another value relates to nothing.** The control of
    /// [`a_test_against_a_literal_alike_is_decided`]: `x != 3` says nothing of
    /// `x != 4`.
    #[test]
    fn a_test_against_another_literal_is_not_decided() {
        assert_eq!(decided_by_another_literal(4), None);
    }
}
