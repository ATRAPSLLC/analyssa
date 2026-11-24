//! Drops the passthrough of a controlled vector operation no reader observes.
//!
//! A [`SsaOp::Compute`] reads the old value of its destination for every
//! element it does not execute: a prestart element, a tail or inactive element
//! the policy leaves undisturbed, one an agnostic policy may leave as it was,
//! and every element of an execution whose restart is at or beyond its length
//! (RISC-V with `vl = 0` changes nothing, tail-agnostic or not). So the old
//! value is an input of the operation even where the policy is agnostic, and a
//! recovered function that loads into a register it never wrote reads that
//! register.
//!
//! Whether any reader sees those elements is a question about the readers. An
//! elementwise reader under the same restart and length reads exactly the
//! elements the operation executes; a store under them writes exactly those.
//! Where every reader of every result reads only executed elements, the old
//! value is unobservable in every execution, the empty one included (the
//! readers read nothing then either), and the operation states
//! [`LaneControl::no_passthrough`]: no old value, an unspecified one in
//! the elements nobody reads. A reader that reaches past the length (a gather's
//! table, a slide's source) or reads through anything this pass does not model
//! observes the old value, and the operation keeps it.
//!
//! # Demand
//!
//! Each vector value gets the elements its readers read: none, the body of an
//! execution (a restart, a length, optionally a governing predicate, at a lane
//! width), or anything. A bit-exact reinterpretation hands its readers' demand
//! on unchanged, since a body is stated at its lane width, not the view's. A
//! phi hands on only "none": a restart or length named by the readers of a phi
//! may be another iteration's, so any other demand is "anything" there. The
//! demands are the least fixpoint, from "none" up, so a loop that feeds an
//! operation's result back as its own next passthrough drops it when nothing
//! else observes it.

use rustc_hash::FxHashMap;

use crate::{
    Result,
    ir::{
        function::SsaFunction,
        ops::{Computation, Kernel, KindedVecData, LaneControl, SsaOp, VectorInactivePolicy},
        variable::SsaVarId,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// `data` with no passthrough inputs: every element it does not execute holds
/// an unspecified value ([`LaneControl::no_passthrough`]).
#[must_use]
pub fn without_passthrough<T: Target>(data: &KindedVecData<Computation>) -> SsaOp<T> {
    let kind = data.kind;
    let control = kind.control_input_count();
    let inputs = data
        .inputs
        .iter()
        .take(control)
        .chain(
            data.inputs
                .iter()
                .skip(control.saturating_add(kind.passthrough_count())),
        )
        .copied()
        .collect();
    SsaOp::compute(
        Computation {
            lanes: kind.lanes.map(|lanes| LaneControl {
                no_passthrough: true,
                ..lanes
            }),
            ..kind
        },
        data.outputs.clone(),
        inputs,
    )
}

/// The controlled vector operation at `block`:`instr`, if that is one.
fn execution_at<T: Target>(
    ssa: &SsaFunction<T>,
    block: usize,
    instr: usize,
) -> Option<&KindedVecData<Computation>> {
    match ssa.block(block)?.instructions().get(instr)?.op() {
        SsaOp::Compute(data) if data.kind.lanes.is_some() => Some(data),
        _ => None,
    }
}

/// The elements of a value its readers read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Demand {
    /// No reader reads any element.
    Nothing,
    /// The readers read elements of one execution's body, and no other.
    Body(Body),
    /// Some reader may read any element.
    Anything,
}

impl Demand {
    /// The demand of readers reading `self` and `other`.
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Nothing, demand) | (demand, Self::Nothing) => demand,
            (Self::Body(left), Self::Body(right))
                if left.start == right.start
                    && left.length == right.length
                    && left.lane_bits == right.lane_bits =>
            {
                // Readers under two predicates read at most the whole body.
                Self::Body(Body {
                    predicate: left.predicate.filter(|_| left.predicate == right.predicate),
                    ..left
                })
            }
            _ => Self::Anything,
        }
    }
}

/// The body of an execution: the elements from `start` below `length`, those
/// `predicate` selects where it is stated, at `lane_bits` per element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Body {
    /// The restart index.
    start: Bound,
    /// The active length.
    length: Bound,
    /// The governing predicate the reader reads under, when it reads only the
    /// elements it selects.
    predicate: Option<SsaVarId>,
    /// The width of an element, in bits.
    lane_bits: u32,
}

/// A restart or length: the value one variable holds, or a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    /// The constant value.
    Constant(u64),
    /// The variable's value.
    Value(SsaVarId),
}

impl Bound {
    /// `var` as a bound: its constant where a constant defines it.
    fn of<T: Target>(ssa: &SsaFunction<T>, var: SsaVarId) -> Self {
        match ssa
            .recorded_definition(var)
            .map(|definition| definition.op())
        {
            Some(SsaOp::Const { value, .. }) => {
                value.as_u64().map_or(Self::Value(var), Self::Constant)
            }
            _ => Self::Value(var),
        }
    }
}

/// How one reader reads a value.
#[derive(Debug, Clone, Copy)]
enum Reader {
    /// A phi defining `result`.
    Phi { result: SsaVarId },
    /// A bit-exact reinterpretation defining `dest`.
    Bitcast { dest: SsaVarId },
    /// Input `position` of the controlled vector operation at `block`:`instr`.
    Execute {
        block: usize,
        instr: usize,
        position: usize,
    },
    /// Any other reader.
    Other,
}

/// Every reader of every value, and where the controlled vector operations
/// are.
struct Readers {
    /// The readers of each value read at all.
    of: FxHashMap<SsaVarId, Vec<Reader>>,
    /// Each controlled vector operation's first result, with its site.
    executions: Vec<(SsaVarId, (usize, usize))>,
}

impl Readers {
    /// The readers of every value of `ssa`.
    fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let mut of: FxHashMap<SsaVarId, Vec<Reader>> = FxHashMap::default();
        let mut executions = Vec::new();
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    of.entry(operand.value()).or_default().push(Reader::Phi {
                        result: phi.result(),
                    });
                }
            }
            for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                match instruction.op() {
                    SsaOp::Bitcast { dest, operand, .. } => {
                        of.entry(*operand)
                            .or_default()
                            .push(Reader::Bitcast { dest: *dest });
                    }
                    SsaOp::Compute(data) if data.kind.lanes.is_some() => {
                        for (position, input) in data.inputs.iter().enumerate() {
                            of.entry(*input).or_default().push(Reader::Execute {
                                block: block_idx,
                                instr: instr_idx,
                                position,
                            });
                        }
                        if let Some(first) = data.outputs.first() {
                            executions.push((*first, (block_idx, instr_idx)));
                        }
                    }
                    op => {
                        for input in op.uses() {
                            of.entry(input).or_default().push(Reader::Other);
                        }
                    }
                }
            }
        }
        Self { of, executions }
    }

    /// The least fixpoint of every value's demand.
    fn demands<T: Target>(&self, ssa: &SsaFunction<T>) -> Demands {
        let mut demands = Demands {
            of: FxHashMap::default(),
        };
        loop {
            let mut changed = false;
            for (value, readers) in &self.of {
                let demand = readers.iter().fold(Demand::Nothing, |demand, reader| {
                    demand.join(demands.read_by(ssa, *value, *reader))
                });
                if demands.of(*value) != demand {
                    demands.of.insert(*value, demand);
                    changed = true;
                }
            }
            if !changed {
                return demands;
            }
        }
    }
}

/// What each value's readers read, as far as the fixpoint has come.
struct Demands {
    /// The demand of each value read at all; an unread value demands nothing.
    of: FxHashMap<SsaVarId, Demand>,
}

impl Demands {
    /// The demand on `value`.
    fn of(&self, value: SsaVarId) -> Demand {
        self.of.get(&value).copied().unwrap_or(Demand::Nothing)
    }

    /// The elements of `value` that `reader` reads.
    fn read_by<T: Target>(&self, ssa: &SsaFunction<T>, value: SsaVarId, reader: Reader) -> Demand {
        match reader {
            Reader::Other => Demand::Anything,
            Reader::Bitcast { dest } => self.of(dest),
            Reader::Phi { result } => match self.of(result) {
                Demand::Nothing => Demand::Nothing,
                Demand::Body(_) | Demand::Anything => Demand::Anything,
            },
            Reader::Execute {
                block,
                instr,
                position,
            } => execution_at(ssa, block, instr).map_or(Demand::Anything, |data| {
                self.read_by_execution(ssa, data, value, position)
            }),
        }
    }

    /// The elements of `value` that input `position` of `data` reads.
    fn read_by_execution<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        data: &KindedVecData<Computation>,
        value: SsaVarId,
        position: usize,
    ) -> Demand {
        let kind = data.kind;
        let control = kind.control_input_count();
        let passthroughs = kind.passthrough_count();
        let results = || data.outputs.iter().take(kind.data_outputs());
        let Some(lane_bits) = lane_bits(ssa, value) else {
            return Demand::Anything;
        };
        let body = |predicate| {
            let (Some(start), Some(length)) = (data.inputs.first(), data.inputs.get(1)) else {
                return Demand::Anything;
            };
            Demand::Body(Body {
                start: Bound::of(ssa, *start),
                length: Bound::of(ssa, *length),
                predicate,
                lane_bits,
            })
        };
        // The elements a mask-tail computation reads past the length land in
        // the result's tail: read only where the result's tail is.
        let tail_unread = || {
            !kind.mask_tail()
                || results().all(|output| self.observes_only_executed(ssa, data, *output))
        };
        let Some(slot) = position.checked_sub(control) else {
            // The governing predicate, read at every body element, or a
            // scalar control.
            return if kind.inactive().is_some() && position == 2 && tail_unread() {
                body(None)
            } else {
                Demand::Anything
            };
        };
        let Some(input) = slot.checked_sub(passthroughs) else {
            // The passthrough reaches the readers of the result only in the
            // elements the operation does not execute.
            return match data.outputs.get(slot) {
                Some(output) if self.observes_only_executed(ssa, data, *output) => Demand::Nothing,
                _ => Demand::Anything,
            };
        };
        if !tail_unread() {
            return Demand::Anything;
        }
        let predicate = data
            .inputs
            .get(2)
            .filter(|_| kind.inactive().is_some())
            .copied();
        match DataRead::of(kind.kernel, input) {
            DataRead::Executed => body(predicate),
            DataRead::Body => body(None),
            DataRead::Seed => match data.inputs.first().map(|start| Bound::of(ssa, *start)) {
                Some(Bound::Constant(0)) => body(None),
                _ => Demand::Anything,
            },
            DataRead::Anywhere => Demand::Anything,
        }
    }

    /// Whether every element of `output`, a data result of `data`, that a
    /// reader reads is one `data` executes.
    fn observes_only_executed<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        data: &KindedVecData<Computation>,
        output: SsaVarId,
    ) -> bool {
        let body = match self.of(output) {
            Demand::Nothing => return true,
            Demand::Anything => return false,
            Demand::Body(body) => body,
        };
        let kind = data.kind;
        let (Some(start), Some(length)) = (data.inputs.first(), data.inputs.get(1)) else {
            return false;
        };
        let selects = match kind.inactive() {
            None | Some(VectorInactivePolicy::Zero) => true,
            Some(VectorInactivePolicy::Undisturbed | VectorInactivePolicy::Agnostic) => {
                body.predicate.is_some() && body.predicate == data.inputs.get(2).copied()
            }
        };
        kind.kernel.executes_its_body()
            && selects
            && lane_bits(ssa, output) == Some(body.lane_bits)
            && body.start == Bound::of(ssa, *start)
            && body.length == Bound::of(ssa, *length)
    }
}

/// The width of one element of `value`'s vector or predicate type.
fn lane_bits<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> Option<u32> {
    let ty = ssa.variable(value)?.var_type();
    T::vector_descriptor(ty)
        .map(|descriptor| descriptor.lane_bits())
        .or_else(|| T::vector_mask_descriptor(ty).map(|descriptor| descriptor.lane_bits()))
}

/// Which elements of a data input a kernel reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataRead {
    /// The element of each result element the operation executes, and no
    /// other: an elementwise kernel.
    Executed,
    /// Elements of the body, executed or not.
    Body,
    /// Element zero, read whenever the body is not empty: a reduction's seed,
    /// which shares its kernel's data inputs with the reduced source.
    Seed,
    /// Any element, the body's bounds aside.
    Anywhere,
}

impl DataRead {
    /// How `kernel` reads its data input `input`.
    const fn of(kernel: Kernel, input: usize) -> Self {
        match kernel {
            Kernel::Move
            | Kernel::Unary(_)
            | Kernel::Binary(_)
            | Kernel::IntegerDivide { .. }
            | Kernel::ModuloShift { .. }
            | Kernel::MultiplyHigh { .. }
            | Kernel::Cast { .. }
            | Kernel::FloatClassify
            | Kernel::FloatToInteger { .. }
            | Kernel::Estimate7 { .. }
            | Kernel::Extend { .. }
            | Kernel::SignInject { .. }
            | Kernel::NarrowShift { .. }
            | Kernel::Fixed { .. }
            | Kernel::CarrylessMultiply { .. }
            | Kernel::Compare { .. }
            | Kernel::WidenBinary { .. }
            | Kernel::FusedMultiplyAdd { .. }
            | Kernel::IntegerMultiplyAdd { .. }
            | Kernel::WidenMultiplyAdd { .. }
            | Kernel::IntegerCarry { .. }
            | Kernel::Select
            | Kernel::MaskBinary(_)
            | Kernel::Memory { .. } => Self::Executed,
            // A gather reads its index elementwise and its table at the
            // indices; a slide reads its source at an offset.
            Kernel::Gather if input == 1 => Self::Executed,
            Kernel::Gather | Kernel::Slide { .. } => Self::Anywhere,
            Kernel::Compress
            | Kernel::MaskPrefix(_)
            | Kernel::MaskIota
            | Kernel::MaskSummary { .. } => Self::Body,
            Kernel::Reduce { .. } | Kernel::ReduceWide { .. } => Self::Seed,
            // An element-group kernel may read one group for every group
            // (`vaesem.vs`), and a scalar insertion reads no vector. A kernel
            // of a whole value runs under no lane control, and reads all of
            // its inputs.
            Kernel::Crypto(_)
            | Kernel::InsertScalar
            | Kernel::Index
            | Kernel::BitDeposit
            | Kernel::BitExtract
            | Kernel::BitGroup
            | Kernel::CounterPredicate { .. }
            | Kernel::PredicateCounter { .. }
            | Kernel::Checksum(_)
            | Kernel::Random { .. }
            | Kernel::PointerAuth(_)
            | Kernel::ScalarCipher(_)
            | Kernel::FloatToIntModular
            | Kernel::MemoryTag(_)
            | Kernel::VectorShapeSupported
            | Kernel::VectorLengthChoice
            | Kernel::VectorFloatSupported
            | Kernel::VectorCryptoSupported
            | Kernel::DecimalAdjust { .. }
            | Kernel::FloatHelper(_)
            | Kernel::TernaryLogic { .. }
            | Kernel::MultiSad { .. }
            | Kernel::DoubleBlockSad { .. }
            | Kernel::StringCompare { .. }
            | Kernel::Bitfield { .. }
            | Kernel::Intersect
            | Kernel::ShuffleBits
            | Kernel::HorizontalMinPos
            | Kernel::FloatControlled { .. } => Self::Anywhere,
        }
    }
}

/// Pass that drops the passthrough of a controlled vector operation whose
/// non-executed elements no reader observes.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - only replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnreadPassthroughPass;

impl UnreadPassthroughPass {
    /// Creates a new unread-passthrough pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for UnreadPassthroughPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "unread-passthrough"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Drop the old value of a vector operation no reader observes"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Drops the passthrough of every controlled vector operation whose
    /// non-executed elements no reader observes.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `_env` - The pass environment, which this pass does not read.
    ///
    /// # Returns
    ///
    /// `true` when at least one operation dropped its passthrough.
    fn run(&self, ssa: &mut SsaFunction<T>, _env: &PassEnv<'_, T>) -> Result<bool> {
        let readers = Readers::of(ssa);
        if readers.executions.is_empty() {
            return Ok(false);
        }
        let demands = readers.demands(ssa);
        let drops: Vec<(usize, usize)> = readers
            .executions
            .iter()
            .filter(|(_, (block, instr))| {
                execution_at(ssa, *block, *instr).is_some_and(|data| {
                    data.kind.passthrough_count() > 0
                        && data
                            .outputs
                            .iter()
                            .take(data.kind.data_outputs())
                            .all(|output| demands.observes_only_executed(ssa, data, *output))
                })
            })
            .map(|(_, site)| *site)
            .collect();
        if drops.is_empty() {
            return Ok(false);
        }
        let mut dropped = false;
        let report = ssa.edit(Default::default(), |editor| {
            for &(block, instr) in &drops {
                let Some(data) = execution_at(editor.function(), block, instr) else {
                    continue;
                };
                let op = without_passthrough(data);
                editor.replace_instruction_op(block, instr, op)?;
                dropped = true;
            }
            Ok(())
        });
        Ok(report.is_ok() && dropped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{SsaVerifier, VerifyLevel, context::AnalysisContext},
        events::NullListener,
        ir::{
            ConstValue, PhiNode, PhiOperand, SsaBlock, SsaInstruction,
            ops::{
                VectorBinaryKind, VectorElement, VectorFaultMode, VectorPredicateLayout,
                VectorRounding, VectorStatus,
            },
            variable::{DefSite, VariableOrigin},
        },
        target::VectorTailPolicy,
        testing::{MACHINE, MockTarget, MockType},
    };

    /// A controlled operation of `kernel` over 32-bit integers, tail-agnostic,
    /// under `inactive`.
    fn kind(kernel: Kernel, inactive: Option<VectorInactivePolicy>) -> Computation {
        Computation {
            kernel,
            element: VectorElement::integer(32, None),
            predicate: VectorPredicateLayout::Packed,
            rounding: VectorRounding::None,
            status: VectorStatus::None,
            canonical_nan: false,
            lanes: Some(LaneControl {
                mask_tail: false,
                tail: VectorTailPolicy::Agnostic,
                inactive,
                no_passthrough: false,
            }),
        }
    }

    /// A unit-stride load or store of one field.
    const fn memory(load: bool) -> Kernel {
        Kernel::Memory {
            load,
            access_bits: 32,
            sign_extend: false,
            byte_stream: false,
            coalescible: false,
            indexed: false,
            ordered: false,
            fields: 1,
            fault: VectorFaultMode::Normal,
        }
    }

    /// Builds a function of entry values and the operations stated over them.
    struct Fixture {
        ssa: SsaFunction<MockTarget>,
        blocks: Vec<SsaBlock<MockTarget>>,
        locals: u16,
    }

    impl Fixture {
        fn new(blocks: usize) -> Self {
            Self {
                ssa: SsaFunction::new(8, 8),
                blocks: (0..blocks).map(SsaBlock::new).collect(),
                locals: 0,
            }
        }

        fn argument(&mut self, index: u16, ty: MockType) -> SsaVarId {
            self.ssa
                .create_variable(VariableOrigin::Argument(index), 0, DefSite::entry(), ty)
        }

        fn local(&mut self, block: usize, ty: MockType) -> SsaVarId {
            let instr = self.blocks[block].instructions().len();
            self.locals += 1;
            self.ssa.create_variable(
                VariableOrigin::Local(self.locals),
                0,
                DefSite::instruction(block, instr),
                ty,
            )
        }

        fn push(&mut self, block: usize, op: SsaOp<MockTarget>) {
            self.blocks[block].add_instruction(SsaInstruction::synthetic(op));
        }

        fn zero(&mut self, block: usize) -> SsaVarId {
            let dest = self.local(block, MockType::I64);
            self.push(
                block,
                SsaOp::Const {
                    dest,
                    value: ConstValue::U64(0),
                },
            );
            dest
        }

        /// A controlled operation of `kind` reading `inputs`, defining one
        /// result of `ty` where `ty` is given.
        fn execute(
            &mut self,
            block: usize,
            kind: Computation,
            inputs: Vec<SsaVarId>,
            ty: Option<MockType>,
        ) -> Option<SsaVarId> {
            let output = ty.map(|ty| self.local(block, ty));
            self.push(
                block,
                SsaOp::compute(kind, output.into_iter().collect(), inputs),
            );
            output
        }

        fn finish(mut self) -> SsaFunction<MockTarget> {
            for block in self.blocks {
                self.ssa.add_block(block);
            }
            self.ssa.refresh_def_sites();
            self.ssa.recompute_uses();
            self.ssa
        }
    }

    /// Runs the pass, checks the function still verifies, and returns whether
    /// it dropped a passthrough and whether `old` is still read.
    ///
    /// The edit renumbers what it orphans, so `old` is found again by its
    /// origin, an argument's.
    fn dropped(mut ssa: SsaFunction<MockTarget>, old: SsaVarId) -> (bool, bool) {
        let origin = ssa.variable(old).map(|old| old.origin());
        let changed = UnreadPassthroughPass.run(
            &mut ssa,
            &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
        ) == Ok(true);
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(errors.is_empty(), "{errors:?}\n{ssa}");
        let is_old = |var: SsaVarId| ssa.variable(var).map(|var| var.origin()) == origin;
        let read = ssa.blocks().iter().any(|block| {
            block
                .phi_nodes()
                .iter()
                .flat_map(|phi| phi.operands())
                .any(|operand| is_old(operand.value()))
                || block
                    .instructions()
                    .iter()
                    .any(|instruction| instruction.op().uses().into_iter().any(is_old))
        });
        (changed, read)
    }

    /// `v = load(0, vl, old, src, 4)`, then `reader` over `v`, returning what
    /// `reader` gives; `old` is handed back for the caller to check.
    fn load_then(
        reader: impl FnOnce(&mut Fixture, SsaVarId, SsaVarId, SsaVarId) -> Option<SsaVarId>,
    ) -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut fixture = Fixture::new(1);
        let length = fixture.argument(0, MockType::I64);
        let source = fixture.argument(1, MockType::I64);
        let old = fixture.argument(2, MockType::NxV4I32);
        let start = fixture.zero(0);
        let loaded = fixture
            .execute(
                0,
                kind(memory(true), None),
                vec![start, length, old, source, source],
                Some(MockType::NxV4I32),
            )
            .unwrap();
        let value = reader(&mut fixture, start, length, loaded);
        fixture.push(0, SsaOp::returns(value));
        (fixture.finish(), old)
    }

    /// **A load only a store under its own length reads keeps no old value.**
    /// The store writes exactly the elements the load executed, in every
    /// execution, `vl = 0` included, so the destination's old value reaches
    /// nothing and the load states none.
    #[test]
    fn a_load_only_a_store_of_its_body_reads_takes_no_old_value() {
        let (ssa, old) = load_then(|fixture, start, length, loaded| {
            let destination = fixture.argument(3, MockType::I64);
            fixture.execute(
                0,
                kind(memory(false), None),
                vec![start, length, destination, destination, loaded],
                None,
            );
            None
        });
        assert_eq!(dropped(ssa, old), (true, false));
    }

    /// **A reader past the body observes the old value.** A returned vector,
    /// a store under another length and a gather's table each read elements the
    /// load may not have executed, so the load keeps its old value; a gather's
    /// index is read elementwise, and the load drops it.
    #[test]
    fn a_reader_past_the_body_keeps_the_old_value() {
        let (ssa, old) = load_then(|_, _, _, loaded| Some(loaded));
        assert_eq!(dropped(ssa, old), (false, true), "returned");

        let (ssa, old) = load_then(|fixture, start, _, loaded| {
            let other = fixture.argument(4, MockType::I64);
            let destination = fixture.argument(3, MockType::I64);
            fixture.execute(
                0,
                kind(memory(false), None),
                vec![start, other, destination, destination, loaded],
                None,
            );
            None
        });
        assert_eq!(dropped(ssa, old), (false, true), "another length");

        // The gather's own old value is unobserved either way: the store
        // reads only its body.
        for (table, expected) in [(true, (true, true)), (false, (true, false))] {
            let (ssa, old) = load_then(|fixture, start, length, loaded| {
                let other = fixture.argument(4, MockType::NxV4I32);
                let into = fixture.argument(5, MockType::NxV4I32);
                let (source, index) = if table {
                    (loaded, other)
                } else {
                    (other, loaded)
                };
                let gathered = fixture
                    .execute(
                        0,
                        kind(Kernel::Gather, None),
                        vec![start, length, into, source, index],
                        Some(MockType::NxV4I32),
                    )
                    .unwrap();
                let destination = fixture.argument(3, MockType::I64);
                fixture.execute(
                    0,
                    kind(memory(false), None),
                    vec![start, length, destination, destination, gathered],
                    None,
                );
                None
            });
            assert_eq!(dropped(ssa, old), expected, "gather table {table}");
        }
    }

    /// **A masked operation keeps the elements its mask leaves.** Under an
    /// undisturbed mask, a reader under the same mask reads only the elements
    /// the operation executed and drops its old value; an unmasked reader reads
    /// the ones it left, which hold the old value, and keeps it. Zeroing
    /// inactive elements needs no old value at all.
    #[test]
    fn a_masked_operation_keeps_what_its_mask_leaves_unread() {
        for (policy, reader_masked, expected) in [
            (VectorInactivePolicy::Undisturbed, true, (true, false)),
            (VectorInactivePolicy::Undisturbed, false, (false, true)),
            (VectorInactivePolicy::Agnostic, false, (false, true)),
            (VectorInactivePolicy::Zero, false, (true, false)),
        ] {
            let mut fixture = Fixture::new(1);
            let length = fixture.argument(0, MockType::I64);
            let mask = fixture.argument(1, MockType::NxMask4);
            let left = fixture.argument(2, MockType::NxV4I32);
            let old = fixture.argument(3, MockType::NxV4I32);
            let destination = fixture.argument(4, MockType::I64);
            let start = fixture.zero(0);
            let sum = fixture
                .execute(
                    0,
                    kind(Kernel::Binary(VectorBinaryKind::Add), Some(policy)),
                    vec![start, length, mask, old, left, left],
                    Some(MockType::NxV4I32),
                )
                .unwrap();
            let (store, inputs) = if reader_masked {
                (
                    kind(memory(false), Some(VectorInactivePolicy::Undisturbed)),
                    vec![start, length, mask, destination, destination, sum],
                )
            } else {
                (
                    kind(memory(false), None),
                    vec![start, length, destination, destination, sum],
                )
            };
            fixture.execute(0, store, inputs, None);
            fixture.push(0, SsaOp::returns(None));
            assert_eq!(
                dropped(fixture.finish(), old),
                expected,
                "{policy:?} read under the mask: {reader_masked}"
            );
        }
    }

    /// **A strip-mined loop drops the old value it carries.** `b1` loads into
    /// the register a phi carries around the loop and stores what it loaded
    /// under the same length; the phi's only reader is the load's own
    /// passthrough, so from "nothing demanded" the fixpoint finds the old value
    /// unobserved, and the value arriving from before the loop is read no more.
    #[test]
    fn a_strip_mined_loop_drops_the_old_value_it_carries() {
        let mut fixture = Fixture::new(3);
        let length = fixture.argument(0, MockType::I64);
        let source = fixture.argument(1, MockType::I64);
        let old = fixture.argument(2, MockType::NxV4I32);
        let more = fixture.argument(3, MockType::I1);
        fixture.push(0, SsaOp::Jump { target: 1 });
        let carried =
            fixture
                .ssa
                .create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::NxV4I32);
        let start = fixture.zero(1);
        let loaded = fixture
            .execute(
                1,
                kind(memory(true), None),
                vec![start, length, carried, source, source],
                Some(MockType::NxV4I32),
            )
            .unwrap();
        fixture.execute(
            1,
            kind(memory(false), None),
            vec![start, length, source, source, loaded],
            None,
        );
        fixture.push(
            1,
            SsaOp::Branch {
                condition: more,
                true_target: 1,
                false_target: 2,
            },
        );
        let mut phi = PhiNode::new(carried, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(old, 0));
        phi.add_operand(PhiOperand::new(loaded, 1));
        fixture.blocks[1].add_phi(phi);
        fixture.push(2, SsaOp::returns(None));
        let ssa = fixture.finish();
        assert_eq!(dropped(ssa, old), (true, false));
    }
}
