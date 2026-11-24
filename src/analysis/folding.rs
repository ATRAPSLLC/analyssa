//! Fold legality: which single-use definitions a lowering may emit at their use.
//!
//! A lowering out of SSA, whether to text or to a stack machine, wants to write
//! a value read exactly once inside the expression that reads it rather than
//! give it a row and a name of its own. Doing so *moves* the computation from
//! where it stands to where its consumer is finally emitted, and a move is only
//! sound when nothing it crosses can observe it. This module decides, for a
//! whole function, which definitions may be folded, and where each fold is
//! finally rendered.
//!
//! # The rules
//!
//! The shape is the same for every candidate: not a terminator, exactly one
//! definition, read exactly once, not pinned by the host. What differs is the
//! right to move:
//!
//! - A **pure, non-trapping** computation computes the same value wherever its
//!   operands are available, so it may fold anywhere it is used, a phi operand
//!   included (the phi copy at the end of the predecessor holds the expression).
//! - A **load** may move forward to its use only within its block, and only
//!   across operations that neither may trap (reordering two traps changes which
//!   fault a reader sees) nor may write a location the load may alias. Another
//!   load may be crossed where one of the two cannot fault, reading storage the
//!   loader mapped ([`IndirectLocation::reads_mapped`]): only one of them can
//!   trap then, and a load that does not has no effect to be seen before or
//!   after it. It is never folded into a phi operand: the copy site is not
//!   where the read stood.
//! - Anything else - a call, a store, a trapping division - moves only when the
//!   host allows it ([`FoldPolicy::moves_effects`]), and then only forward
//!   within its block across operations that are pure and cannot trap, which
//!   are the only ones a move of an effect cannot be observed past.
//!
//! The span a move has to clear runs to where the fold is *rendered*, not to its
//! direct user: a load read by a pure conversion that is itself folded is
//! emitted with the conversion's consumer, so the span reaches that far
//! ([`FoldForest::render_site`]).
//!
//! # Vetoes
//!
//! The host can take folds back after the rules have run: values whose single
//! use sits where an expression cannot go ([`FoldPolicy::fixed_operands`]), and
//! trees deeper than a bound ([`FoldPolicy::max_depth`]). Both are decided here,
//! before anything is rendered, because a renderer that drops a folded row and
//! then declines to expand the fold leaves an operand naming a value nothing
//! defines. A veto only gives a value its row back, which moves the render site
//! of every fold below it earlier, never later, so it cannot make a legal move
//! illegal.
//!
//! # What the result is for
//!
//! [`FoldForest::rowless`] is exactly the input
//! [`Gathering`](crate::analysis::webs::Gathering) takes to decide which values
//! may share a variable: a folded computation reads its operands where it is
//! rendered, and the interference has to see that.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    PointerSize,
    analysis::{
        DefUseIndex,
        image::ImageData,
        memory::{IndirectLocation, MemoryLocation, MemoryLocator},
    },
    ir::{
        function::SsaFunction,
        ops::{RuntimeValueKind, SsaEffectKind, SsaOp},
        variable::{SsaVarId, UseSite},
    },
    target::Target,
};

/// What a host decides about folding that the generic rules cannot.
///
/// Every method has a default that folds as much as the rules allow, so a host
/// overrides only what its lowering needs.
pub trait FoldPolicy<T: Target> {
    /// Whether `value` must keep a row of its own whatever its shape: a value
    /// the host has to be able to name, such as an incoming parameter.
    fn pinned(&self, _value: SsaVarId) -> bool {
        false
    }

    /// Whether constants fold. A host that substitutes immediates through a
    /// path of its own excludes them here.
    fn folds_constants(&self) -> bool {
        true
    }

    /// Whether `op` has an expression form the host can render at all.
    fn may_fold(&self, _op: &SsaOp<T>) -> bool {
        true
    }

    /// Whether an operation that is neither pure nor a load (a call, a store, a
    /// trapping division) may move to its use across a span of pure,
    /// non-trapping operations. A lowering to a stack machine wants this: the
    /// value of a call read by the next instruction stays on the stack.
    fn moves_effects(&self) -> bool {
        false
    }

    /// Whether a pure computation may be rendered in a block other than its
    /// own. A lowering that does not want computation moved into loops, or that
    /// keeps a block's code in its block, says no; the fold is then rendered
    /// only where it is defined (a phi operand: only when the copy's
    /// predecessor is the defining block).
    fn folds_across_blocks(&self) -> bool {
        true
    }

    /// The deepest a fold tree may nest, or `None` for no bound.
    fn max_depth(&self) -> Option<u32> {
        None
    }

    /// Reports the values whose single use sits where an expression cannot go,
    /// given the folds decided so far (`rowless`, by value index). They keep
    /// their rows.
    fn fixed_operands(&self, _ssa: &SsaFunction<T>, _rowless: &[bool], _fixed: &mut Vec<SsaVarId>) {
    }
}

/// The folds a function admits, and where each is rendered.
///
/// The tables are sized by the function's
/// [`var_id_bound`](SsaFunction::var_id_bound). A value past it is malformed IR
/// the lowering does not name: it is never folded and has no definition or use
/// recorded, so an id of three billion costs nothing.
#[derive(Debug, Clone, Default)]
pub struct FoldForest {
    /// By value index: whether the value is folded into its use.
    folded: Vec<bool>,
    /// By value index: the definition site of a single-definition value.
    def_site: Vec<Option<(usize, usize)>>,
    /// The single-definition value each instruction defines, by position.
    defined: BTreeMap<(usize, usize), SsaVarId>,
    /// By value index: the single use of a value read exactly once.
    use_site: Vec<Option<UseSite>>,
}

/// How a candidate's move is constrained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mobility {
    /// Pure and non-trapping: folds anywhere.
    Free,
    /// Pure and non-trapping, but reading a value the host pins: the read is
    /// of the value's place, which another pinned definition may write, so it
    /// moves within its block across no pinned definition.
    ReadsPinned,
    /// A load: moves within its block across non-trapping, non-aliasing code.
    Load,
    /// A value the machine supplies where an instruction takes it (a
    /// timestamp, a system register): moves within its block across pure,
    /// non-trapping code only, so no effect comes between where it is taken
    /// and where it renders, and two of them render in the order they are
    /// taken.
    Positioned,
    /// Anything else: moves only by the host's leave, across pure code.
    Effect,
}

impl FoldForest {
    /// Decides the folds of `ssa`.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function being lowered.
    /// * `defuse` - Its definitions and uses.
    /// * `policy` - The host's decisions.
    /// * `ptr_size` - The target's pointer size, for classifying memory
    ///   accesses.
    /// * `image` - The host's image, where it has one: a load of storage it
    ///   maps cannot fault ([`IndirectLocation::reads_mapped`]).
    #[must_use]
    pub fn compute<T: Target>(
        ssa: &SsaFunction<T>,
        defuse: &DefUseIndex<T>,
        policy: &impl FoldPolicy<T>,
        ptr_size: PointerSize,
        image: Option<&dyn ImageData>,
    ) -> Self {
        let count = ssa.var_id_bound();
        let mut forest = Self {
            folded: vec![false; count],
            def_site: vec![None; count],
            defined: BTreeMap::new(),
            use_site: vec![None; count],
        };

        // Candidates by shape, with how each may move.
        let mut candidates: Vec<(usize, usize, SsaVarId, Mobility)> = Vec::new();
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let op = instr.op();
                let mut defs = op.defs();
                let (Some(dest), None) = (defs.next(), defs.next()) else {
                    continue;
                };
                if let Some(slot) = forest.def_site.get_mut(dest.index()) {
                    *slot = Some((block_idx, instr_idx));
                }
                forest.defined.insert((block_idx, instr_idx), dest);
                let Some([site]) = defuse.uses_of(dest) else {
                    continue;
                };
                if let Some(slot) = forest.use_site.get_mut(dest.index()) {
                    *slot = Some(*site);
                }
                if op.is_terminator()
                    || policy.pinned(dest)
                    || !policy.may_fold(op)
                    || (matches!(op, SsaOp::Const { .. }) && !policy.folds_constants())
                {
                    continue;
                }
                let effects = op.effects();
                let mobility = if effects.kind == SsaEffectKind::Pure && !effects.may_throw {
                    if op.uses().iter().any(|used| policy.pinned(*used)) {
                        Mobility::ReadsPinned
                    } else if !policy.folds_across_blocks()
                        && use_block(ssa, site, dest) != Some(block_idx)
                    {
                        continue;
                    } else {
                        Mobility::Free
                    }
                } else if effects.kind == SsaEffectKind::RuntimeDefined {
                    // A value entering the block makes hold (`RuntimeValue`:
                    // the exception object a handler is entered with) keeps
                    // its row at the block's entry, where a host settles it.
                    // One an instruction takes holds because control arrived
                    // there: rendered at its use with only pure code between,
                    // it is taken at the same point among the function's
                    // effects, which is all its position means.
                    if matches!(
                        op,
                        SsaOp::RuntimeValue { kind, .. } if *kind != RuntimeValueKind::Unspecified
                    ) {
                        continue;
                    }
                    Mobility::Positioned
                } else if MemoryLocator::new(ssa, ptr_size)
                    .classify(op, block_idx, instr_idx)
                    .is_some_and(|memory| memory.is_load())
                {
                    Mobility::Load
                } else if policy.moves_effects() {
                    Mobility::Effect
                } else {
                    continue;
                };
                candidates.push((block_idx, instr_idx, dest, mobility));
            }
        }

        // Free folds first: their legality does not depend on where they end
        // up, and the constrained ones need to know which of their users fold.
        for (_, _, dest, mobility) in &candidates {
            if *mobility == Mobility::Free {
                forest.set_folded(*dest, true);
            }
        }

        // The host's position veto before the constrained folds, so each is
        // decided against the chain it will really be rendered in: a pure user
        // the veto gives its row back ends the chain there, and a load checked
        // past it would be refused a fold it is entitled to.
        let mut fixed = Vec::new();
        policy.fixed_operands(ssa, &forest.folded, &mut fixed);
        for value in &fixed {
            forest.set_folded(*value, false);
        }
        let fixed_set: BTreeSet<SsaVarId> = fixed.into_iter().collect();

        // Constrained folds, each block from its end: a value's user later in
        // the block is decided before the value, so its render site is known.
        let mut constrained: Vec<(usize, usize, SsaVarId, Mobility)> = candidates
            .into_iter()
            .filter(|(_, _, _, mobility)| *mobility != Mobility::Free)
            .collect();
        constrained.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
        for (block_idx, instr_idx, dest, mobility) in constrained {
            if fixed_set.contains(&dest) {
                continue;
            }
            let movable = forest
                .render_of_use(dest)
                .is_some_and(|(render_block, render_instr)| {
                    render_block == block_idx
                        && render_instr > instr_idx
                        && Span {
                            ssa,
                            block: block_idx,
                            from: instr_idx,
                            to: render_instr,
                            ptr_size,
                            image,
                        }
                        .clear(mobility, &|var| policy.pinned(var))
                });
            if movable {
                forest.set_folded(dest, true);
            }
        }

        // The host's vetoes again, positions first, for a host whose answer
        // depends on the constrained folds: a value un-folded for its position
        // gives its row back, which can only shorten a deeper tree.
        let mut fixed = Vec::new();
        policy.fixed_operands(ssa, &forest.folded, &mut fixed);
        for value in fixed {
            forest.set_folded(value, false);
        }
        if let Some(bound) = policy.max_depth() {
            forest.veto_deep(ssa, bound);
        }
        forest
    }

    /// For each value index, whether the value is folded into its use: the
    /// `rowless` input of [`Gathering`](crate::analysis::webs::Gathering).
    #[must_use]
    pub fn rowless(&self) -> &[bool] {
        &self.folded
    }

    /// Whether `value` is folded into its use.
    #[must_use]
    pub fn is_folded(&self, value: SsaVarId) -> bool {
        self.folded.get(value.index()).copied().unwrap_or(false)
    }

    /// Where the tree a folded `value` belongs to is rendered, as
    /// `(block, instruction)`: the first user up the chain of folds that keeps
    /// its row. `None` when `value` is not folded, or when the chain ends in a
    /// phi operand, which is rendered as a copy at the end of the predecessor.
    #[must_use]
    pub fn render_site(&self, value: SsaVarId) -> Option<(usize, usize)> {
        if !self.is_folded(value) {
            return None;
        }
        self.render_of_use(value)
    }

    /// Where `value`'s single use is finally rendered, following folded users.
    fn render_of_use(&self, value: SsaVarId) -> Option<(usize, usize)> {
        let mut current = value;
        // A chain of single-use folds is a path through distinct values, so it
        // is at most as long as there are values.
        for _ in 0..self.folded.len() {
            let site = self.use_site.get(current.index()).copied().flatten()?;
            if site.is_phi_operand() {
                return None;
            }
            let user = self.defined_at(site.block(), site.index());
            match user {
                Some(user) if self.is_folded(user) => current = user,
                _ => return Some((site.block(), site.index())),
            }
        }
        None
    }

    /// The single value defined at `(block, instr)`, if any.
    fn defined_at(&self, block: usize, instr: usize) -> Option<SsaVarId> {
        self.defined.get(&(block, instr)).copied()
    }

    fn set_folded(&mut self, value: SsaVarId, folded: bool) {
        if let Some(slot) = self.folded.get_mut(value.index()) {
            *slot = folded;
        }
    }

    /// Gives a row back to every fold nested deeper than `bound`.
    ///
    /// The walk starts at every operation that keeps a row and at every phi
    /// operand, and descends operand trees; a vetoed value's operands start a
    /// tree of their own, from depth zero.
    fn veto_deep<T: Target>(&mut self, ssa: &SsaFunction<T>, bound: u32) {
        let mut stack: Vec<(SsaVarId, u32)> = Vec::new();
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                stack.extend(phi.operands().iter().map(|operand| (operand.value(), 0)));
            }
            for instr in block.instructions() {
                let op = instr.op();
                if op.defs().next().is_some_and(|dest| self.is_folded(dest)) {
                    continue;
                }
                stack.extend(op.uses().into_iter().map(|used| (used, 0)));
            }
        }
        let mut vetoed: BTreeSet<SsaVarId> = BTreeSet::new();
        let mut seen: BTreeSet<(SsaVarId, u32)> = BTreeSet::new();
        while let Some((value, depth)) = stack.pop() {
            if !self.is_folded(value) || !seen.insert((value, depth)) {
                continue;
            }
            let next = if depth >= bound {
                vetoed.insert(value);
                0
            } else {
                depth.saturating_add(1)
            };
            let Some((block, instr)) = self.def_site.get(value.index()).copied().flatten() else {
                continue;
            };
            if let Some(op) = ssa
                .block(block)
                .and_then(|b| b.instructions().get(instr))
                .map(|i| i.op())
            {
                stack.extend(op.uses().into_iter().map(|used| (used, next)));
            }
        }
        for value in vetoed {
            self.set_folded(value, false);
        }
    }
}

/// The block a use of `value` is rendered in: the using instruction's block, or
/// for a phi operand the predecessor the copy is placed in.
fn use_block<T: Target>(ssa: &SsaFunction<T>, site: &UseSite, value: SsaVarId) -> Option<usize> {
    if !site.is_phi_operand() {
        return Some(site.block());
    }
    ssa.block(site.block())?
        .phi_nodes()
        .get(site.index())?
        .operands()
        .iter()
        .find(|operand| operand.value() == value)
        .map(|operand| operand.predecessor())
}

/// The rows a move takes the operation at `(block, from)` over, to just
/// before `to`.
struct Span<'s, T: Target> {
    /// The function.
    ssa: &'s SsaFunction<T>,
    /// The block the move is in.
    block: usize,
    /// Where the operation stands.
    from: usize,
    /// Where it is rendered.
    to: usize,
    /// The target's pointer size, for classifying memory accesses.
    ptr_size: PointerSize,
    /// The host's image, where it has one.
    image: Option<&'s dyn ImageData>,
}

impl<T: Target> Span<'_, T> {
    /// Whether the move crosses nothing that can observe it.
    ///
    /// A definition the host pins keeps its row where it stands, and the host
    /// pins what is observed there - a value a handler reads, a slot a pointer
    /// reads - so a move that may trap or has an effect does not cross one: a
    /// throw on the far side would find the slot already written.
    fn clear(&self, mobility: Mobility, pinned: &dyn Fn(SsaVarId) -> bool) -> bool {
        let Some(instructions) = self.ssa.block(self.block).map(|b| b.instructions()) else {
            return false;
        };
        let Some(between) = instructions.get(self.from.saturating_add(1)..self.to) else {
            return false;
        };
        if mobility != Mobility::Free && between.iter().any(|instr| instr.op().defs().any(pinned)) {
            return false;
        }
        match mobility {
            Mobility::Free | Mobility::ReadsPinned => true,
            Mobility::Effect | Mobility::Positioned => between.iter().all(|instr| {
                let effects = instr.op().effects();
                effects.kind == SsaEffectKind::Pure && !effects.may_throw
            }),
            Mobility::Load => {
                let Some(location) = self
                    .access(self.from)
                    .and_then(|memory| memory.location().cloned())
                else {
                    return false;
                };
                let mapped = self.reads_mapped(&location);
                between.iter().enumerate().all(|(offset, instr)| {
                    let at = self.from.saturating_add(1).saturating_add(offset);
                    let access = self.access(at);
                    // Two loads trade places where one cannot fault.
                    let crossable = !instr.op().may_throw()
                        || access.as_ref().is_some_and(|memory| {
                            memory.is_load()
                                && instr.op().effects().kind != SsaEffectKind::Write
                                && (mapped
                                    || memory
                                        .location()
                                        .is_some_and(|crossed| self.reads_mapped(crossed)))
                        });
                    crossable
                        && !access.is_some_and(|memory| {
                            memory
                                .defined_locations()
                                .any(|defined| defined.may_alias(&location))
                        })
                })
            }
        }
    }

    /// The memory access of the operation at `at`, where it makes one.
    fn access(&self, at: usize) -> Option<crate::analysis::memory::MemoryOp<T>> {
        let instr = self.ssa.block(self.block)?.instructions().get(at)?;
        MemoryLocator::new(self.ssa, self.ptr_size).classify(instr.op(), self.block, at)
    }

    /// Whether a read of `location` cannot fault
    /// ([`IndirectLocation::reads_mapped`]).
    fn reads_mapped(&self, location: &MemoryLocation<T>) -> bool {
        let word_bits = i64::try_from(self.ptr_size.bytes())
            .unwrap_or(8)
            .saturating_mul(8);
        matches!(location, MemoryLocation::Indirect(cell)
            if IndirectLocation::reads_mapped(cell, self.ssa, self.image, word_bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Endianness,
        analysis::image::{ImageRegion, Mutability},
        ir::{
            CallData, Callee,
            block::SsaBlock,
            instruction::SsaInstruction,
            ops::{AccessQualifiers, NativeKindedData, RuntimeValueKind, SystemOpKind},
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// Writable data the loader maps at `0x3000..0x3100`, whose addresses are
    /// the program's integers.
    struct Data;

    impl ImageData for Data {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            (0x3000..0x3100).contains(&address).then_some(ImageRegion {
                start: 0x3000,
                end: 0x3100,
                mutability: Mutability::Writable,
            })
        }

        fn bytes(&self, _address: u64, _len: usize) -> Option<&[u8]> {
            None
        }

        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    /// An image that serves no byte but maps `0x3000..0x3100`, as a host
    /// maps code whose bytes a program may rewrite; or, `mapping` false, maps
    /// nothing it does not serve either.
    struct Code {
        mapping: bool,
    }

    impl ImageData for Code {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, _address: u64) -> Option<ImageRegion> {
            None
        }

        fn mapped(&self, address: u64, len: u64) -> bool {
            self.mapping
                && address >= 0x3000
                && address.checked_add(len).is_some_and(|end| end <= 0x3100)
        }

        fn bytes(&self, _address: u64, _len: usize) -> Option<&[u8]> {
            None
        }

        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    /// The default policy, and one for a stack machine.
    struct Host {
        moves_effects: bool,
        pinned: Vec<SsaVarId>,
        max_depth: Option<u32>,
    }

    impl Host {
        fn text() -> Self {
            Self {
                moves_effects: false,
                pinned: Vec::new(),
                max_depth: None,
            }
        }

        fn stack() -> Self {
            Self {
                moves_effects: true,
                ..Self::text()
            }
        }
    }

    impl FoldPolicy<MockTarget> for Host {
        fn pinned(&self, value: SsaVarId) -> bool {
            self.pinned.contains(&value)
        }

        fn moves_effects(&self) -> bool {
            self.moves_effects
        }

        fn max_depth(&self) -> Option<u32> {
            self.max_depth
        }
    }

    /// Builds a function whose blocks hold `blocks`, over values `v0..vars`.
    fn function(vars: usize, blocks: Vec<Vec<SsaOp<MockTarget>>>) -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 0);
        for _ in 0..vars {
            let _ = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);
        }
        for (index, ops) in blocks.into_iter().enumerate() {
            let mut block = SsaBlock::new(index);
            for op in ops {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        ssa
    }

    fn v(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    fn forest(ssa: &SsaFunction<MockTarget>, host: &Host) -> FoldForest {
        let defuse = DefUseIndex::build(ssa);
        FoldForest::compute(ssa, &defuse, host, PointerSize::Bit64, None)
    }

    fn konst(dest: usize, value: i32) -> SsaOp<MockTarget> {
        SsaOp::Const {
            dest: v(dest),
            value: ConstValue::I32(value),
        }
    }

    fn add(dest: usize, left: usize, right: usize) -> SsaOp<MockTarget> {
        SsaOp::Add {
            dest: v(dest),
            left: v(left),
            right: v(right),
            flags: None,
        }
    }

    fn call(dest: Option<usize>, method: u32, args: &[usize]) -> SsaOp<MockTarget> {
        SsaOp::from(CallData::new(
            dest.map(v),
            Callee::Direct(method),
            args.iter().copied().map(v).collect(),
        ))
    }

    fn load(dest: usize, field: u32) -> SsaOp<MockTarget> {
        SsaOp::LoadStaticField {
            access: AccessQualifiers::FLAT,
            dest: v(dest),
            field,
        }
    }

    fn store(field: u32, value: usize) -> SsaOp<MockTarget> {
        SsaOp::StoreStaticField {
            access: AccessQualifiers::FLAT,
            field,
            value: v(value),
        }
    }

    fn ret(value: usize) -> SsaOp<MockTarget> {
        SsaOp::returns(Some(v(value)))
    }

    #[test]
    fn a_pure_chain_folds_into_its_consumer() {
        // v0 = 1; v1 = 2; v2 = v0 + v1; return v2
        let ssa = function(
            3,
            vec![vec![konst(0, 1), konst(1, 2), add(2, 0, 1), ret(2)]],
        );
        let forest = forest(&ssa, &Host::text());
        assert!(forest.is_folded(v(0)) && forest.is_folded(v(1)) && forest.is_folded(v(2)));
        assert_eq!(forest.render_site(v(0)), Some((0, 3)));
    }

    #[test]
    fn a_value_read_twice_keeps_its_row() {
        // v0 = 1; v1 = v0 + v0; return v1
        let ssa = function(2, vec![vec![konst(0, 1), add(1, 0, 0), ret(1)]]);
        let forest = forest(&ssa, &Host::text());
        assert!(!forest.is_folded(v(0)));
    }

    #[test]
    fn a_pinned_value_keeps_its_row() {
        let ssa = function(2, vec![vec![konst(0, 1), konst(1, 2), ret(0)]]);
        let host = Host {
            pinned: vec![v(0)],
            ..Host::text()
        };
        assert!(!forest(&ssa, &host).is_folded(v(0)));
    }

    #[test]
    fn a_load_does_not_move_past_a_store_to_the_same_field() {
        // v0 = F1; F1 = v1; return v0
        let ssa = function(2, vec![vec![konst(1, 5), load(0, 1), store(1, 1), ret(0)]]);
        assert!(!forest(&ssa, &Host::text()).is_folded(v(0)));
    }

    #[test]
    fn a_load_moves_past_pure_code() {
        // v0 = F1; v1 = 5; v2 = v1 + v1; v3 = v0 + v2; return v3
        let ssa = function(
            4,
            vec![vec![
                load(0, 1),
                konst(1, 5),
                add(2, 1, 1),
                add(3, 0, 2),
                ret(3),
            ]],
        );
        assert!(forest(&ssa, &Host::text()).is_folded(v(0)));
    }

    #[test]
    fn a_load_does_not_move_past_a_store_that_may_trap() {
        // A store to another field still may trap, and reordering two traps
        // changes which fault a reader sees: v0 = F1; F2 = v1; return v0
        let ssa = function(2, vec![vec![konst(1, 5), load(0, 1), store(2, 1), ret(0)]]);
        assert!(!forest(&ssa, &Host::text()).is_folded(v(0)));
    }

    /// **A load of mapped data moves past another load.** dash's
    /// `sub_2800` copies two globals into a record: `v2 = [0x3000]; v3 =
    /// [0x3004]; [v4] = v2; [v5] = v3`. A load may trap, and reordering two
    /// traps changes which fault a reader sees; but the image maps both
    /// globals, so neither read faults, and the first folds into its store
    /// past the second. The second stays: the first store may trap.
    #[test]
    fn a_load_of_mapped_data_moves_past_another_load() {
        let address = |dest: usize, value: i64| SsaOp::Const {
            dest: v(dest),
            value: ConstValue::I64(value),
        };
        let load_at = |dest: usize, addr: usize| SsaOp::LoadIndirect {
            dest: v(dest),
            addr: v(addr),
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        };
        let store_at = |addr: usize, value: usize| SsaOp::StoreIndirect {
            addr: v(addr),
            value: v(value),
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        };
        let ssa = function(
            6,
            vec![vec![
                address(0, 0x3000),
                address(1, 0x3004),
                address(4, 0x9000),
                address(5, 0x9004),
                load_at(2, 0),
                load_at(3, 1),
                store_at(4, 2),
                store_at(5, 3),
                SsaOp::returns(None),
            ]],
        );
        let defuse = DefUseIndex::build(&ssa);
        let mapped = FoldForest::compute(
            &ssa,
            &defuse,
            &Host::text(),
            PointerSize::Bit64,
            Some(&Data),
        );
        assert!(mapped.is_folded(v(2)));
        assert!(!mapped.is_folded(v(3)));
        // Control: with no image to say so, either load may fault.
        assert!(!forest(&ssa, &Host::text()).is_folded(v(2)));
    }

    /// **A load of storage the image maps but does not serve moves past
    /// another load.** [`a_load_of_mapped_data_moves_past_another_load`]'s
    /// function over an image that maps its globals as a host maps code -
    /// read, never served - folds the first load all the same: mapped storage
    /// is no fault whether or not its bytes are known. Control: an image that
    /// maps nothing it does not serve leaves it.
    #[test]
    fn a_load_of_mapped_unserved_storage_moves_past_another_load() {
        let address = |dest: usize, value: i64| SsaOp::Const {
            dest: v(dest),
            value: ConstValue::I64(value),
        };
        let load_at = |dest: usize, addr: usize| SsaOp::LoadIndirect {
            dest: v(dest),
            addr: v(addr),
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        };
        let store_at = |addr: usize, value: usize| SsaOp::StoreIndirect {
            addr: v(addr),
            value: v(value),
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        };
        let ssa = function(
            6,
            vec![vec![
                address(0, 0x3000),
                address(1, 0x3004),
                address(4, 0x9000),
                address(5, 0x9004),
                load_at(2, 0),
                load_at(3, 1),
                store_at(4, 2),
                store_at(5, 3),
                SsaOp::returns(None),
            ]],
        );
        let defuse = DefUseIndex::build(&ssa);
        let folded = |mapping: bool| {
            FoldForest::compute(
                &ssa,
                &defuse,
                &Host::text(),
                PointerSize::Bit64,
                Some(&Code { mapping }),
            )
            .is_folded(v(2))
        };
        assert!(folded(true));
        assert!(!folded(false));
    }

    #[test]
    fn a_load_does_not_move_past_a_call() {
        // v0 = F1; call M(); return v0
        let ssa = function(1, vec![vec![load(0, 1), call(None, 1, &[]), ret(0)]]);
        assert!(!forest(&ssa, &Host::text()).is_folded(v(0)));
    }

    #[test]
    fn a_load_into_a_phi_operand_keeps_its_row() {
        // B0: v0 = F1; jump B1 / B1: v1 = phi(v0); return v1
        let mut ssa = function(
            2,
            vec![vec![load(0, 1), SsaOp::Jump { target: 1 }], vec![ret(1)]],
        );
        if let Some(block) = ssa.block_mut(1) {
            let mut phi = PhiNode::new(v(1), VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(v(0), 0));
            block.add_phi(phi);
        }
        ssa.recompute_uses();
        assert!(!forest(&ssa, &Host::text()).is_folded(v(0)));
    }

    #[test]
    fn a_call_folds_only_where_the_host_moves_effects() {
        // v0 = A(); return v0
        let ssa = function(1, vec![vec![call(Some(0), 1, &[]), ret(0)]]);
        assert!(!forest(&ssa, &Host::text()).is_folded(v(0)));
        assert!(forest(&ssa, &Host::stack()).is_folded(v(0)));
    }

    /// **A value the runtime supplies renders where it is taken.** A machine
    /// query folds into its use across pure code, whatever the host says of
    /// effects, and not across a call or another query: of two timestamps
    /// subtracted, the first keeps its row, so the listing reads them in the
    /// order the function takes them. The exception object a handler is
    /// entered with keeps its row, where a host settles it on entry.
    #[test]
    fn a_runtime_value_renders_at_its_use_across_pure_code_only() {
        let query = |dest: usize| {
            SsaOp::SystemOp(Box::new(NativeKindedData {
                kind: SystemOpKind::Timestamp { aux: false },
                mnemonic: "rdtsc".to_string(),
                metadata: None,
                outputs: vec![v(dest)],
                inputs: Vec::new(),
            }))
        };
        // v0 = now; v1 = 1; v2 = v1 + v1; v3 = v0 + v2; return v3
        let ssa = function(
            4,
            vec![vec![
                query(0),
                konst(1, 1),
                add(2, 1, 1),
                add(3, 0, 2),
                ret(3),
            ]],
        );
        assert!(forest(&ssa, &Host::text()).is_folded(v(0)));
        // v0 = now; F() ; return v0
        let ssa = function(1, vec![vec![query(0), call(None, 1, &[]), ret(0)]]);
        assert!(!forest(&ssa, &Host::stack()).is_folded(v(0)));
        // v0 = now; v1 = now; v2 = v1 - v0; return v2
        let ssa = function(
            3,
            vec![vec![
                query(0),
                query(1),
                SsaOp::Sub {
                    dest: v(2),
                    left: v(1),
                    right: v(0),
                    flags: None,
                },
                ret(2),
            ]],
        );
        let ordered = forest(&ssa, &Host::text());
        assert!(!ordered.is_folded(v(0)), "the first query keeps its row");
        assert!(ordered.is_folded(v(1)));
        // v0 = exception; v1 = 1; v2 = v0 + v1; return v2
        let entered = SsaOp::RuntimeValue {
            dest: v(0),
            kind: RuntimeValueKind::ExceptionObject,
        };
        let ssa = function(3, vec![vec![entered, konst(1, 1), add(2, 0, 1), ret(2)]]);
        for host in [Host::text(), Host::stack()] {
            assert!(!forest(&ssa, &host).is_folded(v(0)), "settled on entry");
        }
        // v0 = unspecified; v1 = 1; v2 = v0 + v1; return v2: a choice made
        // where control reaches it renders at its use.
        let chosen = SsaOp::RuntimeValue {
            dest: v(0),
            kind: RuntimeValueKind::Unspecified,
        };
        let ssa = function(3, vec![vec![chosen, konst(1, 1), add(2, 0, 1), ret(2)]]);
        assert!(
            forest(&ssa, &Host::text()).is_folded(v(0)),
            "taken at its use"
        );
    }

    #[test]
    fn a_call_does_not_move_past_another_call() {
        // v0 = A(); B(); C(v0)
        let ssa = function(
            1,
            vec![vec![
                call(Some(0), 1, &[]),
                call(None, 2, &[]),
                call(None, 3, &[0]),
                SsaOp::returns(None),
            ]],
        );
        assert!(!forest(&ssa, &Host::stack()).is_folded(v(0)));
    }

    #[test]
    fn a_call_moves_past_pure_code_only() {
        // v0 = A(); v1 = 1; v2 = v1 + v1; B(v0, v2)
        let ssa = function(
            3,
            vec![vec![
                call(Some(0), 1, &[]),
                konst(1, 1),
                add(2, 1, 1),
                call(None, 2, &[0, 2]),
                SsaOp::returns(None),
            ]],
        );
        assert!(forest(&ssa, &Host::stack()).is_folded(v(0)));
    }

    #[test]
    fn a_load_under_a_folded_conversion_clears_the_span_to_the_render_site() {
        // The load's user folds, so the span runs to the user's consumer, past
        // the store: v0 = F1; v2 = 1; v1 = v0 + v2; F1 = v2; return v1
        let ssa = function(
            3,
            vec![vec![
                load(0, 1),
                konst(2, 1),
                add(1, 0, 2),
                store(1, 2),
                ret(1),
            ]],
        );
        let forest = forest(&ssa, &Host::text());
        // v2 is read twice, so it keeps its row; v1 folds into the return,
        // which stands past the store to F1, so the load must not fold.
        assert!(forest.is_folded(v(1)));
        assert!(!forest.is_folded(v(0)));
    }

    #[test]
    fn a_pure_value_stays_in_its_block_when_the_host_says_so() {
        // B0: v0 = 1; jump B1 / B1: return v0
        let ssa = function(
            1,
            vec![vec![konst(0, 1), SsaOp::Jump { target: 1 }], vec![ret(0)]],
        );
        assert!(forest(&ssa, &Host::text()).is_folded(v(0)));
        struct Local;
        impl FoldPolicy<MockTarget> for Local {
            fn folds_across_blocks(&self) -> bool {
                false
            }
        }
        let defuse = DefUseIndex::build(&ssa);
        let local = FoldForest::compute(&ssa, &defuse, &Local, PointerSize::Bit64, None);
        assert!(!local.is_folded(v(0)));
    }

    #[test]
    fn a_load_folds_up_to_a_user_the_host_keeps_a_row_for() {
        // v0 = F1; v1 = 1; v2 = v0 + v1; v3 = *v2; return v3
        //
        // v2 is the second load's address, which the host renders only as a
        // named value. Decided against v2 folding, the first load's chain would
        // run on through the second load (which may trap) to the return, and be
        // refused; with v2 keeping its row the first load sits right before it.
        struct Addresses;
        impl FoldPolicy<MockTarget> for Addresses {
            fn fixed_operands(
                &self,
                ssa: &SsaFunction<MockTarget>,
                _rowless: &[bool],
                fixed: &mut Vec<SsaVarId>,
            ) {
                for block in ssa.blocks() {
                    for instr in block.instructions() {
                        if let SsaOp::LoadIndirect { addr, .. } = instr.op() {
                            fixed.push(*addr);
                        }
                    }
                }
            }
        }
        let ssa = function(
            4,
            vec![vec![
                load(0, 1),
                konst(1, 1),
                add(2, 0, 1),
                SsaOp::LoadIndirect {
                    dest: v(3),
                    addr: v(2),
                    value_type: MockType::I32,
                    access: AccessQualifiers::FLAT,
                },
                ret(3),
            ]],
        );
        let defuse = DefUseIndex::build(&ssa);
        let forest = FoldForest::compute(&ssa, &defuse, &Addresses, PointerSize::Bit64, None);
        assert!(!forest.is_folded(v(2)));
        assert!(
            forest.is_folded(v(0)),
            "the first load is rendered at v2's row"
        );
    }

    #[test]
    fn a_call_does_not_move_past_a_pinned_definition() {
        // v0 = A(); v1 = 1 (pinned: a handler reads it); B(v0)
        let ssa = function(
            2,
            vec![vec![
                call(Some(0), 1, &[]),
                konst(1, 1),
                call(None, 2, &[0]),
                SsaOp::returns(None),
            ]],
        );
        let host = Host {
            pinned: vec![v(1)],
            ..Host::stack()
        };
        assert!(!forest(&ssa, &host).is_folded(v(0)));
        assert!(forest(&ssa, &Host::stack()).is_folded(v(0)));
    }

    #[test]
    fn a_read_of_a_pinned_value_does_not_move_past_a_pinned_definition() {
        // v0 = A() (pinned); v1 = v0; v2 = 7 (pinned, v0's place); B(v1)
        //
        // A host pins values that live in a shared place: v0 and v2 are
        // versions a handler reads, in one slot. Folded into B's row, the
        // copy would read the slot after v2 wrote it.
        let ssa = function(
            3,
            vec![vec![
                call(Some(0), 1, &[]),
                SsaOp::Copy {
                    dest: v(1),
                    src: v(0),
                },
                konst(2, 7),
                call(None, 2, &[1]),
                SsaOp::returns(None),
            ]],
        );
        let host = Host {
            pinned: vec![v(0), v(2)],
            ..Host::stack()
        };
        assert!(!forest(&ssa, &host).is_folded(v(1)));

        // With nothing written in between, the read moves.
        let host = Host {
            pinned: vec![v(0)],
            ..Host::stack()
        };
        assert!(forest(&ssa, &host).is_folded(v(1)));
    }

    #[test]
    fn a_tree_deeper_than_the_bound_gets_rows_back() {
        // v0 = 1; v1 = 2; v2 = v0 + v1; v3 = 3; v4 = v2 + v3; return v4
        let ssa = function(
            5,
            vec![vec![
                konst(0, 1),
                konst(1, 2),
                add(2, 0, 1),
                konst(3, 3),
                add(4, 2, 3),
                ret(4),
            ]],
        );
        let host = Host {
            max_depth: Some(1),
            ..Host::text()
        };
        let forest = forest(&ssa, &host);
        // v4 is at depth 0 under the return and v2 at depth 1, the bound: v2
        // gets its row back and its operands start a tree of their own.
        assert!(forest.is_folded(v(4)));
        assert!(!forest.is_folded(v(2)));
        assert!(forest.is_folded(v(0)));
    }
}
