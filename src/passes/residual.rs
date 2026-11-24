//! Residualization: replacing what a static execution computed with its
//! result.
//!
//! A [`StaticFacts`] states what holds on every run of a function. What it
//! establishes, this pass writes back into the function, so the computation
//! that produced it no longer has to be read to learn it:
//!
//! - A value that is one constant on every run becomes that constant, where
//!   the instruction defining it does nothing else: it is pure, or it only
//!   reads memory.
//! - A loop that leaves the same bytes and the same live-out values on every
//!   run becomes those writes and those values, and its exit edge is taken
//!   straight from before it: the loop is gone, and what it computed is stated
//!   outright.
//!
//! Facts describe the exact SSA snapshot execution observed. Repair can renumber
//! dense value ids; use [`apply_static_facts`] to combine scalar and loop edits.
//! Independent passes require fresh facts after an edit, even when the result
//! remains valid SSA. Reusing the old value ids can name different definitions.

use std::{
    cell::{Cell, OnceCell},
    collections::{BTreeMap, BTreeSet},
    result,
};

use crate::{
    Error, Result,
    analysis::{
        budget::{Budget, Exhaustion, WorkBudget},
        cfg::SsaCfg,
        execution::{LoopEffect, ReadFootprint, ReplacementProof, Root, StaticFacts},
        image::ImageRead,
        loopanalyzer::LoopAnalyzer,
        loops::{LoopForest, LoopInfo},
    },
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        function::{SsaEditOptions, SsaEditor, SsaFunction, SsaRollbackPolicy},
        instruction::{InstrId, SsaInstruction},
        ops::{AccessQualifiers, AtomicOrdering, Callee, SsaEffectKind, SsaOp},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

/// One instruction replacement - a value's constant, or a call's one executed
/// target - and the image evidence retained after its edit commits.
struct Replacement<T: Target> {
    /// Block containing the instruction.
    block: usize,
    /// Instruction within the block.
    instruction: usize,
    /// Replacement operation.
    op: SsaOp<T>,
    /// The constants of an operation's other results, where it defines
    /// several: inserted after it once every plan naming the snapshot's
    /// positions has been applied.
    others: Vec<SsaOp<T>>,
    /// Original image access, when this replaces a load.
    read: Option<ImageRead<T>>,
}

/// Why a bounded residual request left the original body unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResidualRefusal {
    /// The owner cancelled the request before publication.
    Cancelled,
    /// The complete planned or produced body exceeds one residual budget
    /// ([`Budget::ResidualOperations`] and its siblings).
    Exhausted(Exhaustion),
    /// A representation count cannot be expressed as a host size.
    SizeOverflow,
}

/// Counts of the complete SSA representation at one transaction boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidualSize {
    /// Instructions plus phi nodes.
    pub operations: usize,
    /// Allocated variable slots.
    pub variables: usize,
    /// Block slots.
    pub blocks: usize,
    /// Incoming phi operands.
    pub phi_operands: usize,
}

impl ResidualSize {
    /// Measures the complete body against each residual budget.
    ///
    /// Counts include the original body until repair removes it; deletion
    /// receives no advance credit. The budgets do not bound allocation inside
    /// unrelated passes.
    ///
    /// # Arguments
    ///
    /// * `function` - The body to count.
    /// * `budget` - The residual budgets it is counted against.
    /// * `cancelled` - The owner's cancellation, asked before each block and
    ///   each phi; `None` for a count no owner cancels.
    ///
    /// # Errors
    ///
    /// Returns the first exceeded budget, an unrepresentable count, or
    /// [`ResidualRefusal::Cancelled`] once `cancelled` answers `true`.
    pub fn measure<T: Target>(
        function: &SsaFunction<T>,
        budget: &WorkBudget,
        cancelled: Option<&dyn Fn() -> bool>,
    ) -> result::Result<Self, ResidualRefusal> {
        let cancelled = || cancelled.is_some_and(|cancelled| cancelled());
        if cancelled() {
            return Err(ResidualRefusal::Cancelled);
        }
        let mut size = Self {
            variables: function.variable_count(),
            blocks: function.block_count(),
            ..Self::default()
        };
        size.check(budget)?;
        for block in function.blocks() {
            if cancelled() {
                return Err(ResidualRefusal::Cancelled);
            }
            size.operations = size
                .operations
                .checked_add(block.instruction_count())
                .and_then(|count| count.checked_add(block.phi_count()))
                .ok_or(ResidualRefusal::SizeOverflow)?;
            size.check(budget)?;
            for phi in block.phi_nodes() {
                if cancelled() {
                    return Err(ResidualRefusal::Cancelled);
                }
                size.phi_operands = size
                    .phi_operands
                    .checked_add(phi.operand_count())
                    .ok_or(ResidualRefusal::SizeOverflow)?;
            }
            size.check(budget)?;
        }
        size.check(budget)?;
        Ok(size)
    }

    /// Checks these counts in deterministic budget order.
    ///
    /// # Errors
    ///
    /// Returns the first exceeded budget.
    pub fn check(self, budget: &WorkBudget) -> result::Result<(), ResidualRefusal> {
        for (kind, demand) in [
            (Budget::ResidualOperations, self.operations),
            (Budget::ResidualVariables, self.variables),
            (Budget::ResidualBlocks, self.blocks),
            (Budget::ResidualPhiOperands, self.phi_operands),
        ] {
            if let Some(exhaustion) = budget.check(kind, demand) {
                return Err(ResidualRefusal::Exhausted(exhaustion));
            }
        }
        Ok(())
    }
}

/// The complete result of a bounded immutable residual request.
pub enum ResidualRewrite<T: Target> {
    /// The facts supplied no applicable root-body replacement.
    Unchanged,
    /// A repaired and verified replacement, ready for the owner's next stage.
    Rewritten {
        /// Complete replacement body; the input body remains untouched.
        function: Box<SsaFunction<T>>,
        /// Number of scalar and loop plans committed together.
        replacements: usize,
        /// The stores each decoded run of bytes was written back as, in the
        /// order the loops' runs were applied.
        runs: Vec<ResidualRun>,
    },
    /// The complete request was withheld while its original facts remain valid.
    Refused(ResidualRefusal),
}

/// The stores a rewrite wrote one decoded run of bytes back as: the run is
/// the loop's ([`LoopEffect::writes`]), the stores the returned body's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualRun {
    /// The decoded loop's header in the input body ([`LoopEffect::header`]).
    pub header: usize,
    /// The run's position among the loop's [`LoopEffect::writes`].
    pub run: usize,
    /// One store per byte of the run, in address order, by identity in the
    /// returned body.
    pub stores: Vec<InstrId>,
}

/// Owner cancellation state shared by planning and one complete edit.
struct ResidualControl<'a> {
    /// Owner cancellation predicate, independent of instruction construction.
    cancelled: &'a dyn Fn() -> bool,
    /// Sticky cancellation: a later clear predicate cannot revive this request.
    refused: Cell<bool>,
}

impl ResidualControl<'_> {
    /// Checks the owner without clearing an earlier cancellation.
    fn active(&self) -> bool {
        if self.refused.get() || (self.cancelled)() {
            self.refused.set(true);
            false
        } else {
            true
        }
    }

    /// Aborts an open edit so its unpublished working body can be discarded.
    fn check_edit(&self) -> Result<()> {
        if self.active() {
            Ok(())
        } else {
            Err(Error::new("residual request cancelled"))
        }
    }
}

/// Produces a bounded residual body without modifying its input or execution facts.
///
/// Plans against the original namespace, checks complete peak representation
/// counts, then clones the admitted body once. Cancellation or a refused bound
/// discards the whole replacement. Repair, full verification and read metadata
/// remain one publication boundary. The owner may independently normalize the
/// returned body and must recheck its own output limits afterward.
///
/// # Errors
///
/// Returns an error if the admitted edit or full SSA verification fails.
pub fn rewrite_static_facts<T: Target>(
    ssa: &SsaFunction<T>,
    facts: &StaticFacts<T>,
    budget: &WorkBudget,
    cancelled: &dyn Fn() -> bool,
) -> Result<ResidualRewrite<T>> {
    let control = ResidualControl {
        cancelled,
        refused: Cell::new(false),
    };
    if !control.active() {
        return Ok(ResidualRewrite::Refused(ResidualRefusal::Cancelled));
    }
    if !facts.is_root_entry() || !facts.is_complete() {
        return Ok(ResidualRewrite::Unchanged);
    }
    let initial = match ResidualSize::measure(ssa, budget, Some(&|| !control.active())) {
        Ok(size) => size,
        Err(reason) => return Ok(ResidualRewrite::Refused(reason)),
    };
    let planner = Planner {
        ssa,
        facts,
        control: Some(&control),
    };
    let loops = planner.loops();
    if !control.active() {
        return Ok(ResidualRewrite::Refused(ResidualRefusal::Cancelled));
    }
    match planner
        .peak(initial, &loops.plans)
        .and_then(|size| size.check(budget))
    {
        Ok(()) => {}
        Err(reason) => return Ok(ResidualRewrite::Refused(reason)),
    }
    let mut folds = planner.constants(&loops.covered);
    folds.extend(planner.calls(&loops.covered));
    if !control.active() {
        return Ok(ResidualRewrite::Refused(ResidualRefusal::Cancelled));
    }
    if folds.is_empty() && loops.plans.is_empty() {
        return Ok(ResidualRewrite::Unchanged);
    }
    let mut function = ssa.clone();
    let edited = commit_static_facts(
        &mut function,
        folds,
        loops.plans,
        loops.reads,
        Some(&control),
        SsaRollbackPolicy::Never,
    );
    if !control.active() {
        return Ok(ResidualRewrite::Refused(ResidualRefusal::Cancelled));
    }
    let (replacements, runs) = edited?;
    if let Err(reason) = ResidualSize::measure(&function, budget, Some(&|| !control.active())) {
        return Ok(ResidualRewrite::Refused(reason));
    }
    if !control.active() {
        return Ok(ResidualRewrite::Refused(ResidualRefusal::Cancelled));
    }
    Ok(ResidualRewrite::Rewritten {
        function: Box::new(function),
        replacements,
        runs,
    })
}

/// Applies scalar and loop facts from one unchanged SSA snapshot atomically.
///
/// Both planners run before editing. Scalar definitions inside selected loop
/// bodies are excluded; surviving scalar replacements and loop residuals share
/// one repair boundary, so dense value ids cannot be reinterpreted after an
/// intermediate compaction. The original facts remain unchanged for reporting.
/// Image read witnesses are published only after full verification succeeds.
/// Returns the total number of scalar and loop replacements. Call-scoped facts
/// return zero: observations from one invocation cannot rewrite a shared callee.
///
/// # Errors
///
/// Returns an error when editing or full SSA verification fails. The function,
/// including its read metadata, is restored on failure.
pub fn apply_static_facts<T: Target>(
    ssa: &mut SsaFunction<T>,
    facts: &StaticFacts<T>,
) -> Result<usize> {
    if !facts.is_root_entry() {
        return Ok(0);
    }
    let planner = Planner {
        ssa,
        facts,
        control: None,
    };
    let loops = planner.loops();
    let folds = planner.constants(&loops.covered);
    commit_static_facts(
        ssa,
        folds,
        loops.plans,
        loops.reads,
        None,
        SsaRollbackPolicy::OnFailure,
    )
    .map(|(count, _)| count)
}

/// Replaces every value `facts` proves constant by that constant, where the
/// instruction defining it has no effect but its value. Returns how many were
/// replaced. Each folded read-only image access retains its source and whether
/// control enters the loaded value. Entry provenance is computed before editing,
/// and only when an image access is replaced.
///
/// # Arguments
///
/// * `ssa` - The function `facts` were established on.
/// * `facts` - What execution established on this exact, unmodified snapshot.
///   After another edit, obtain fresh facts or use [`apply_static_facts`].
///
/// # Errors
///
/// Returns an error when the edit cannot be applied; the function is then
/// left as it was.
pub fn fold_constants<T: Target>(
    ssa: &mut SsaFunction<T>,
    facts: &StaticFacts<T>,
) -> Result<usize> {
    if !facts.is_root_entry() {
        return Ok(0);
    }
    let planner = Planner {
        ssa,
        facts,
        control: None,
    };
    let folds = planner.constants(&BTreeSet::new());
    commit_static_facts(
        ssa,
        folds,
        Vec::new(),
        Vec::new(),
        None,
        SsaRollbackPolicy::OnFailure,
    )
    .map(|(count, _)| count)
}

/// How a decoded loop's writes and live-outs are stated once the loop is
/// gone: where to splice them, the exit to jump to, and what to write.
struct LoopResidual<'a, T: Target> {
    /// The block outside the loop that entered it.
    entry: usize,
    /// Whether `entry` branches elsewhere too, so the residual goes into a
    /// block split onto its edge into the loop rather than into `entry`.
    split: bool,
    /// The loop's header, whose entry edge from the preheader is redirected.
    header: usize,
    /// The normal predecessor whose proved exit operands the residual carries.
    exit_from: usize,
    /// Where control leaves the loop.
    exit: usize,
    /// The runs of bytes the loop leaves written, each with where it starts.
    writes: Option<ByteWrites<'a, T>>,
    /// The loop-defined values used after it, and the constant each holds.
    live_outs: Vec<(SsaVarId, ConstValue<T>)>,
    /// The loop's blocks, which the preheader's edge now runs past.
    body: Vec<usize>,
}

/// Nonempty byte-write runs with the target representation required to emit them.
struct ByteWrites<'a, T: Target> {
    /// Supported byte storage type, resolved once before any edit begins.
    byte_type: T::Type,
    /// Consecutive write ranges established by the loop.
    runs: Vec<Write<'a>>,
}

/// One run of bytes a decoded loop leaves written.
struct Write<'a> {
    /// Where the run starts.
    base: Base,
    /// The bytes, from `base` up.
    bytes: &'a [u8],
}

/// Where a run of bytes starts.
enum Base {
    /// `base + offset`: an address past a value the program holds where the
    /// loop is entered, the stack pointer on entry for the stack, and for a
    /// realigned frame a value whose executed address lies in it.
    Relative {
        /// The value the run is addressed from.
        base: SsaVarId,
        /// The offset from it, in bytes.
        offset: i64,
    },
    /// An address in the image, as a bare number.
    Image {
        /// The address.
        address: u64,
    },
}

/// Replaces every loop `facts` decoded by the bytes and values it leaves, and
/// removes it. Returns how many loops were replaced.
///
/// A loop is replaced only when the rewrite is unambiguous and sound: it has
/// one entry from outside and one exit whose incoming phi values are
/// available where it is entered, every loop live-out is constant, and every
/// byte it wrote lands somewhere the residual can name. Anything else is left
/// as it was. An entry that branches elsewhere too keeps its test: the
/// residual goes into a block split onto its edge into the loop.
///
/// # Arguments
///
/// * `ssa` - The function `facts` were established on.
/// * `facts` - What execution established on this exact, unmodified snapshot.
///   After another edit, obtain fresh facts or use [`apply_static_facts`].
///
/// # Errors
///
/// Returns an error when the edit cannot be applied; the function is then left
/// as it was.
pub fn residualize_loops<T: Target>(
    ssa: &mut SsaFunction<T>,
    facts: &StaticFacts<T>,
) -> Result<usize> {
    if !facts.is_root_entry() {
        return Ok(0);
    }
    let planner = Planner {
        ssa,
        facts,
        control: None,
    };
    let plans = planner.loops();
    commit_static_facts(
        ssa,
        Vec::new(),
        plans.plans,
        plans.reads,
        None,
        SsaRollbackPolicy::OnFailure,
    )
    .map(|(count, _)| count)
}

/// The planning of one rewrite: the function planned on, what its execution
/// established, and the owner's cancellation, asked as planning goes.
struct Planner<'p, 'f, T: Target> {
    /// The unmodified function the plans are made against.
    ssa: &'p SsaFunction<T>,
    /// What execution established on `ssa`.
    facts: &'f StaticFacts<T>,
    /// The owner's cancellation, where one can cancel.
    control: Option<&'p ResidualControl<'p>>,
}

impl<'f, T: Target> Planner<'_, 'f, T> {
    /// Whether the owner cancelled: once it has, every later question is
    /// answered the same way ([`ResidualControl::active`]).
    fn cancelled(&self) -> bool {
        self.control.is_some_and(|control| !control.active())
    }

    /// Computes the peak before unreachable loop bodies and old phi inputs are removed.
    fn peak(
        &self,
        mut size: ResidualSize,
        plans: &[LoopResidual<'_, T>],
    ) -> result::Result<ResidualSize, ResidualRefusal> {
        for plan in plans {
            if self.cancelled() {
                return Err(ResidualRefusal::Cancelled);
            }
            if let Some(writes) = &plan.writes {
                for write in &writes.runs {
                    if self.cancelled() {
                        return Err(ResidualRefusal::Cancelled);
                    }
                    let base = usize::from(!matches!(write.base, Base::Relative { offset: 0, .. }));
                    let bytes = write.bytes.len();
                    let pointers = bytes.saturating_sub(1);
                    let operations = bytes
                        .checked_mul(2)
                        .and_then(|n| n.checked_add(pointers))
                        .and_then(|n| n.checked_add(base))
                        .ok_or(ResidualRefusal::SizeOverflow)?;
                    let variables = bytes
                        .checked_add(pointers)
                        .and_then(|n| n.checked_add(base))
                        .ok_or(ResidualRefusal::SizeOverflow)?;
                    size.operations = size
                        .operations
                        .checked_add(operations)
                        .ok_or(ResidualRefusal::SizeOverflow)?;
                    size.variables = size
                        .variables
                        .checked_add(variables)
                        .ok_or(ResidualRefusal::SizeOverflow)?;
                }
            }
            size.operations = size
                .operations
                .checked_add(plan.live_outs.len())
                .ok_or(ResidualRefusal::SizeOverflow)?;
            size.variables = size
                .variables
                .checked_add(plan.live_outs.len())
                .ok_or(ResidualRefusal::SizeOverflow)?;
            let inputs = self
                .ssa
                .block(plan.exit)
                .map_or(0, |block| block.phi_count());
            size.phi_operands = size
                .phi_operands
                .checked_add(inputs)
                .ok_or(ResidualRefusal::SizeOverflow)?;
        }
        Ok(size)
    }

    /// Plans scalar replacements against the unchanged source, excluding replaced loop bodies.
    fn constants(&self, covered: &BTreeSet<usize>) -> Vec<Replacement<T>> {
        let entered = OnceCell::new();
        self.facts
            .constants()
            .into_iter()
            .filter_map(|(value, constant)| {
                if self.cancelled() {
                    return None;
                }
                let definition = self.ssa.recorded_definition(value)?;
                if covered.contains(&definition.block_index()) {
                    return None;
                }
                let op = definition.op();
                // An operation of several results (a double-width division's
                // quotient and remainder, a product's halves, a quotient and the
                // flags it leaves) is planned once, from its first known result,
                // where every result anything reads is a known constant: one
                // constant takes its place, the others follow it, and a result
                // nothing reads, such as flags the machine leaves undefined,
                // needs none.
                let mut defs = op
                    .defs()
                    .filter(|dest| {
                        self.facts.constant(*dest).is_some()
                            || !self.ssa.find_var_user_blocks(*dest, None).is_empty()
                    })
                    .peekable();
                if defs.next() != Some(value) {
                    return None;
                }
                let others = defs
                    .map(|dest| {
                        Some(SsaOp::Const {
                            dest,
                            value: self.facts.constant(dest)?.clone(),
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                let proof = self.facts.replacement_proof(value);
                let proven = op
                    .defs()
                    .filter(|dest| self.facts.constant(*dest).is_some())
                    .all(|dest| self.facts.replacement_proof(dest).is_some());
                let effects = op.effects();
                // A proven atomic access is a relaxed load: it orders nothing, and
                // the proof states its read cannot fault.
                let foldable = (effects.is_pure() || proven)
                    && effects
                        .ordering
                        .is_none_or(|ordering| proven && ordering == AtomicOrdering::Relaxed)
                    && !matches!(op, SsaOp::Const { .. })
                    && match effects.kind {
                        SsaEffectKind::Pure | SsaEffectKind::Read => true,
                        SsaEffectKind::Atomic => proven,
                        _ => false,
                    }
                    && !effects.volatile;
                foldable.then(|| Replacement {
                    block: definition.block_index(),
                    instruction: definition.instruction_index(),
                    op: SsaOp::Const {
                        dest: value,
                        value: constant.clone(),
                    },
                    others,
                    read: match proof {
                        Some(ReplacementProof::ReadOnlyImage { address, len }) => {
                            // The type of what was read: a broadcast read one
                            // element, the lane's width.
                            // A lane-wise load reads elements of its lanes'
                            // width across the span: no one constant is the bytes.
                            let (value_type, whole) = match op {
                                SsaOp::LoadIndirect { value_type, .. }
                                | SsaOp::AtomicLoad { value_type, .. }
                                | SsaOp::VectorLoad {
                                    vector_type: value_type,
                                    ..
                                } => (Some(value_type.clone()), true),
                                SsaOp::VectorBroadcastLoad { vector_type, .. } => (
                                    T::vector_shape(vector_type)
                                        .and_then(|shape| T::integer_type(shape.lane_bits, false)),
                                    true,
                                ),
                                SsaOp::VectorMaskedLoad { vector_type, .. }
                                | SsaOp::VectorGather { vector_type, .. }
                                | SsaOp::VectorFaultingLoad { vector_type, .. }
                                | SsaOp::VectorSegmentLoad { vector_type, .. }
                                | SsaOp::VectorPackLoad { vector_type, .. } => (
                                    T::vector_shape(vector_type)
                                        .and_then(|shape| T::integer_type(shape.lane_bits, false)),
                                    false,
                                ),
                                SsaOp::VectorStructLoadReplicate(data) => {
                                    (T::integer_type(u32::from(data.element_bits), false), false)
                                }
                                _ => (None, false),
                            };
                            value_type.map(|value_type| ImageRead {
                                address,
                                len,
                                instruction: definition.instruction().id(),
                                original: Some(definition.instruction().original().clone()),
                                value_type,
                                value: whole.then(|| constant.clone()),
                                entered: entered
                                    .get_or_init(|| self.ssa.entered_values())
                                    .contains(&value),
                            })
                        }
                        _ => None,
                    },
                })
            })
            .collect()
    }

    /// Plans each call of the root body that every visit of the trial sent to
    /// one method as a direct call to that method, so the executed transfer is
    /// stated where the call graph reads it: an indirect call, and a direct one
    /// that entered another method than it names, the version recovered from
    /// code written at its address.
    ///
    /// A complete root-entry trial visited every path that reaches the call; a
    /// visit that resolved elsewhere, or to nothing, leaves the call as it is.
    fn calls(&self, covered: &BTreeSet<usize>) -> Vec<Replacement<T>> {
        let targets = self.facts.call_targets();
        let mut plans = Vec::new();
        for (block_index, block) in self.ssa.blocks().iter().enumerate() {
            if covered.contains(&block_index) {
                continue;
            }
            for (index, instruction) in block.instructions().iter().enumerate() {
                if self.cancelled() {
                    return Vec::new();
                }
                let op = instruction.op();
                let Some(method) = targets.unique(0, block_index, index) else {
                    continue;
                };
                // A computed target every visit reached, or a direct call every
                // visit entered elsewhere than it names: code written at its
                // address since, recovered as a version of its own.
                let stated = op.as_call().map(|call| &call.callee);
                if !matches!(stated, Some(Callee::Indirect { .. }))
                    && !matches!(stated, Some(Callee::Direct(named)) if named != method)
                {
                    continue;
                }
                let mut op = op.clone();
                if let Some(call) = op.as_call_mut() {
                    call.callee = Callee::Direct(method.clone());
                }
                plans.push(Replacement {
                    block: block_index,
                    instruction: index,
                    op,
                    others: Vec::new(),
                    read: None,
                });
            }
        }
        plans
    }

    /// Plans all supported loops before any repair can renumber the source values.
    fn loops(&self) -> LoopResiduals<'f, T> {
        if self.cancelled() {
            return LoopResiduals::default();
        }
        let headers = self.facts.decoded_loops();
        if headers.is_empty() {
            return LoopResiduals::default();
        }
        let forest = LoopAnalyzer::new(self.ssa).analyze();
        let cfg = SsaCfg::from_ssa(self.ssa);
        if self.cancelled() {
            return LoopResiduals::default();
        }
        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let mut residuals: Vec<LoopResidual<'f, T>> = headers
            .into_iter()
            .filter_map(|header| self.one_loop(&cfg, &dominators, &forest, header))
            .collect();
        if residuals.is_empty() {
            return LoopResiduals::default();
        }
        // Prefer an eligible outer loop; its residual includes the nested effects.
        // Disjoint plans keep stable block identities throughout the transaction.
        residuals.sort_by(|a, b| {
            b.body
                .len()
                .cmp(&a.body.len())
                .then(a.header.cmp(&b.header))
        });
        let mut covered = BTreeSet::new();
        residuals.retain(|residual| {
            if residual.body.iter().any(|block| covered.contains(block)) {
                return false;
            }
            covered.extend(residual.body.iter().copied());
            true
        });
        let mut keys = BTreeSet::new();
        let mut reads = Vec::new();
        let entered = OnceCell::new();
        for residual in &residuals {
            for (key, value) in self.facts.loop_reads(residual.header) {
                if self.cancelled() {
                    return LoopResiduals::default();
                }
                if keys.insert(key) {
                    let Some(read) = residual_read(self.ssa, key, value, &entered) else {
                        return LoopResiduals::default();
                    };
                    reads.push(read);
                }
            }
        }
        LoopResiduals {
            plans: residuals,
            reads,
            covered,
        }
    }

    /// Plans the residual for the loop headed by `header`, or `None` when it may
    /// not be replaced.
    fn one_loop(
        &self,
        cfg: &SsaCfg<'_, T>,
        dominators: &DominatorTree,
        forest: &LoopForest,
        header: usize,
    ) -> Option<LoopResidual<'f, T>> {
        if self.cancelled() {
            return None;
        }
        let info = forest.loop_for_header(NodeId::new(header))?;
        let effect = self.facts.loop_effect(header)?;
        let (exit_from, exit) = effect.exit;
        let in_loop = |block: usize| info.body.contains(block);
        if !in_loop(exit_from)
            || in_loop(exit)
            || !cfg.terminator_successors(exit_from).contains(&exit)
        {
            return None;
        }
        // One entry from outside the loop, whose edge into it the residual
        // takes the place of.
        let mut outside = cfg
            .block_predecessors(header)
            .iter()
            .copied()
            .filter(|pred| !in_loop(*pred));
        let entry = outside.next()?;
        if outside.next().is_some() || !cfg.terminator_successors(entry).contains(&header) {
            return None;
        }
        // Residual writes execute where the loop was entered. A conditional
        // entry would also execute them on its other arm, so its edge into the
        // loop is split and the new block holds them: the exit test of a loop
        // before this one, a guard on the count.
        let split = !matches!(
            self.ssa.block(entry).and_then(|block| block.control_terminator()),
            Some(SsaOp::Jump { target }) if *target == header
        );
        // The preheader's edge skips the loop for the exit; it may not leave a
        // region the loop entry stayed in, nor enter one mid-way. A split block
        // lies in no region, at the next index, and its edges must respect them
        // too, as `SsaEditor::split_edges` checks.
        let regions = if split {
            let block = self.ssa.block_count();
            self.ssa.retarget_respects_regions(entry, header, block)
                && self.ssa.retarget_respects_regions(block, header, header)
                && self.ssa.retarget_respects_regions(block, header, exit)
        } else {
            self.ssa.retarget_respects_regions(entry, header, exit)
        };
        if !regions {
            return None;
        }
        let entry_stack = self.ssa.entry_stack_pointer();
        let writes = if effect.writes.is_empty() {
            None
        } else {
            let byte_type = T::integer_type(8, false)?;
            let runs = effect
                .writes
                .iter()
                .map(|(address, bytes)| {
                    let base = match address.root() {
                        Root::Stack => Base::Relative {
                            base: entry_stack?,
                            offset: address.offset(),
                        },
                        Root::Symbol => Base::Image {
                            address: address.image_address()?,
                        },
                        // A realigned frame lies at no offset of the entry stack:
                        // the run is addressed from a value the program holds
                        // in the frame where the loop is entered.
                        root @ Root::StackAligned { .. } => {
                            let (base, at) = realigned_base(
                                self.ssa,
                                dominators,
                                forest,
                                self.facts,
                                root,
                                (entry, header),
                            )?;
                            Base::Relative {
                                base,
                                offset: address.offset().checked_sub(at)?,
                            }
                        }
                        Root::Number
                        | Root::StackFrame(_)
                        | Root::StackDerived { .. }
                        | Root::Heap(_)
                        | Root::Opaque { .. } => {
                            return None;
                        }
                    };
                    if self.cancelled() {
                        return None;
                    }
                    Some(Write { base, bytes })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(ByteWrites { byte_type, runs })
        };
        let live_outs = self.live_outs(info, effect)?;
        let materialized: BTreeSet<_> = live_outs.iter().map(|(value, _)| *value).collect();
        for phi in self.ssa.block(exit)?.phi_nodes() {
            if self.cancelled() {
                return None;
            }
            let value = phi
                .normal_operands()
                .find(|operand| operand.predecessor() == exit_from)?
                .value();
            let definition = self.ssa.variable(value)?.def_site().block();
            if in_loop(definition) {
                if !materialized.contains(&value) {
                    return None;
                }
            } else if !dominators.dominates(NodeId::new(definition), NodeId::new(entry)) {
                return None;
            }
        }
        Some(LoopResidual {
            entry,
            split,
            header,
            exit_from,
            exit,
            writes,
            live_outs,
            body: info.body.iter().collect(),
        })
    }

    /// The loop's values used after it, each with the constant it holds there, or
    /// `None` when one is used after the loop without being constant.
    fn live_outs(
        &self,
        info: &LoopInfo,
        effect: &LoopEffect<T>,
    ) -> Option<Vec<(SsaVarId, ConstValue<T>)>> {
        let in_loop = |block: usize| info.body.contains(block);
        let mut live_outs = Vec::new();
        let defined = (0..self.ssa.block_count())
            .filter(|block| in_loop(*block))
            .filter(|_| !self.cancelled())
            .filter_map(|block| self.ssa.block(block))
            .flat_map(|block| {
                let phis = block.phi_nodes().iter().map(|phi| phi.result());
                let defs = block
                    .instructions()
                    .iter()
                    .flat_map(|instruction| instruction.op().defs());
                phis.chain(defs)
            });
        for value in defined {
            if self.cancelled() {
                return None;
            }
            let used_outside = self
                .ssa
                .variable(value)
                .into_iter()
                .flat_map(|variable| variable.uses())
                .any(|use_site| !in_loop(use_site.block()));
            if !used_outside {
                continue;
            }
            let constant = effect
                .exit_values
                .iter()
                .find(|(exit_value, _)| *exit_value == value)
                .and_then(|(_, held)| held.constant())?;
            live_outs.push((value, constant.clone()));
        }
        Some(live_outs)
    }
}

/// Complete disjoint loop plans and their original source domains.
struct LoopResiduals<'a, T: Target> {
    /// Selected loop replacements, in deterministic application order.
    plans: Vec<LoopResidual<'a, T>>,
    /// Original reads retained by those plans.
    reads: Vec<ImageRead<T>>,
    /// Original blocks whose scalar definitions the replacements remove.
    covered: BTreeSet<usize>,
}

impl<T: Target> Default for LoopResiduals<'_, T> {
    fn default() -> Self {
        Self {
            plans: Vec::new(),
            reads: Vec::new(),
            covered: BTreeSet::new(),
        }
    }
}

/// Edits one admitted body and publishes read metadata only after its verified boundary.
///
/// Returns how many plans it committed, and the stores each decoded run was
/// written back as.
fn commit_static_facts<'a, T: Target>(
    ssa: &mut SsaFunction<T>,
    folds: Vec<Replacement<T>>,
    residuals: Vec<LoopResidual<'a, T>>,
    mut reads: Vec<ImageRead<T>>,
    control: Option<&ResidualControl<'_>>,
    rollback: SsaRollbackPolicy,
) -> Result<(usize, Vec<ResidualRun>)> {
    let count = folds.len().saturating_add(residuals.len());
    if count == 0 {
        return Ok((0, Vec::new()));
    }
    let mut runs = Vec::new();
    if let Some(control) = control {
        control.check_edit()?;
    }
    ssa.edit(
        SsaEditOptions::new()
            .with_verify(true)
            .with_rollback(rollback),
        |editor| {
            // Replacements preserve instruction positions. Apply them before
            // loop residuals insert into preheaders, while all source ids and
            // positions still name the snapshot used by both planners. The
            // other results of a multi-result operation follow its first
            // once every plan is applied, found again by the instruction's
            // identity.
            let mut following = Vec::new();
            for fold in folds {
                if let Some(control) = control {
                    control.check_edit()?;
                }
                if !fold.others.is_empty() {
                    let replaced = editor
                        .function()
                        .block(fold.block)
                        .and_then(|block| block.instruction(fold.instruction))
                        .and_then(|instruction| {
                            Some((instruction.id()?, instruction.original().clone()))
                        })
                        .ok_or_else(|| {
                            Error::new(format!(
                                "B{}:I{}: a multi-result replacement has no identity",
                                fold.block, fold.instruction
                            ))
                        })?;
                    following.push((replaced, fold.others));
                }
                editor.replace_instruction_op(fold.block, fold.instruction, fold.op)?;
                reads.extend(fold.read);
            }
            let splits = split_entries(editor, &residuals)?;
            for residual in residuals {
                apply_residual(editor, &residual, &splits, control, &mut runs)?;
            }
            for ((id, original), others) in following {
                let (block, at) = editor.function().locate_instruction(id).ok_or_else(|| {
                    Error::new("a multi-result replacement's instruction was removed")
                })?;
                for (offset, op) in others.into_iter().enumerate() {
                    editor.insert_instruction(
                        block,
                        at.saturating_add(1).saturating_add(offset),
                        SsaInstruction::new(original.clone(), op),
                    )?;
                }
            }
            if let Some(control) = control {
                control.check_edit()?;
            }
            Ok(())
        },
    )?;
    if let Some(control) = control {
        control.check_edit()?;
    }
    for read in reads {
        if let Some(control) = control {
            control.check_edit()?;
        }
        T::record_image_read(ssa.facts_mut(), read);
    }
    if let Some(control) = control {
        control.check_edit()?;
    }
    Ok((count, runs))
}

/// Restores a performed read's metadata before any source instruction is edited.
fn residual_read<T: Target>(
    ssa: &SsaFunction<T>,
    key: ReadFootprint,
    value: Option<&ConstValue<T>>,
    entered: &OnceCell<BTreeSet<SsaVarId>>,
) -> Option<ImageRead<T>> {
    let instruction = ssa.block(key.block)?.instruction(key.instruction)?;
    let (value_type, result) = match instruction.op() {
        SsaOp::LoadIndirect {
            value_type, dest, ..
        }
        | SsaOp::VectorLoad {
            vector_type: value_type,
            dest,
            ..
        }
        | SsaOp::VectorBroadcastLoad {
            vector_type: value_type,
            dest,
            ..
        } => (value_type.clone(), Some(*dest)),
        SsaOp::LoadBlk {
            element_bytes,
            value,
            ..
        } => (
            T::integer_type(element_bytes.saturating_mul(8), false).unwrap_or_else(T::unknown_type),
            Some(*value),
        ),
        SsaOp::CompareBlk(data) => (
            T::integer_type(data.element_bytes.saturating_mul(8), false)
                .unwrap_or_else(T::unknown_type),
            None,
        ),
        SsaOp::ScanBlk { element_bytes, .. } => (
            T::integer_type(element_bytes.saturating_mul(8), false).unwrap_or_else(T::unknown_type),
            None,
        ),
        SsaOp::CopyBlk { .. } | SsaOp::MoveBlk { .. } => (
            T::integer_type(8, false).unwrap_or_else(T::unknown_type),
            None,
        ),
        _ => return None,
    };
    Some(ImageRead {
        address: key.address,
        len: key.len,
        instruction: instruction.id(),
        original: Some(instruction.original().clone()),
        value_type,
        value: value.cloned(),
        entered: key.produces
            && result.is_some_and(|result| {
                entered
                    .get_or_init(|| ssa.entered_values())
                    .contains(&result)
            }),
    })
}

/// Applies `residual`: the writes and live-outs go into the preheader, then
/// its edge into the loop is redirected past the loop to the exit. Polls the
/// owner's cancellation, where it has one, inside the byte expansion, and
/// appends the stores each run was written back as to `runs`.
///
/// The preheader is the loop's entry, or the block `splits` put onto the
/// entry's edge into it ([`split_entries`]). An exit edge split that way,
/// into a loop entered from this one's exit test, now leaves through the
/// split block, so the residual goes on into the next one.
fn apply_residual<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    residual: &LoopResidual<'_, T>,
    splits: &BTreeMap<(usize, usize), usize>,
    control: Option<&ResidualControl<'_>>,
    runs: &mut Vec<ResidualRun>,
) -> Result<()> {
    if let Some(control) = control {
        control.check_edit()?;
    }
    let preheader = if residual.split {
        *splits
            .get(&(residual.entry, residual.header))
            .ok_or_else(|| Error::new("the planned loop entry edge was not split"))?
    } else {
        residual.entry
    };
    let exit = splits
        .get(&(residual.exit_from, residual.exit))
        .copied()
        .unwrap_or(residual.exit);
    if let Some(writes) = &residual.writes {
        for (run, write) in writes.runs.iter().enumerate() {
            if let Some(control) = control {
                control.check_edit()?;
            }
            let mut stores = Vec::with_capacity(write.bytes.len());
            let base = materialize_base(editor, preheader, &write.base)?;
            for (index, byte) in write.bytes.iter().enumerate() {
                if index.is_multiple_of(32)
                    && let Some(control) = control
                {
                    control.check_edit()?;
                }
                let offset = i64::try_from(index).unwrap_or(i64::MAX);
                let at = ptr_add(editor, preheader, base, offset)?;
                let value = const_var(
                    editor,
                    preheader,
                    ConstValue::U8(*byte),
                    writes.byte_type.clone(),
                )?;
                let index = editor.insert_before_terminator(
                    preheader,
                    SsaInstruction::synthetic(SsaOp::StoreIndirect {
                        addr: at,
                        value,
                        value_type: writes.byte_type.clone(),
                        access: AccessQualifiers::FLAT,
                    }),
                )?;
                let store = editor
                    .function()
                    .block(preheader)
                    .and_then(|block| block.instructions().get(index))
                    .and_then(SsaInstruction::id)
                    .ok_or_else(|| Error::new("residual store without an identity"))?;
                stores.push(store);
            }
            runs.push(ResidualRun {
                header: residual.header,
                run,
                stores,
            });
        }
    }
    for (old, constant) in &residual.live_outs {
        if let Some(control) = control {
            control.check_edit()?;
        }
        // The constant stands for the value it replaces, of that value's type.
        let var_type = editor
            .function()
            .variable(*old)
            .map(|variable| variable.var_type().clone())
            .filter(|ty| !T::is_unknown(ty))
            .or_else(|| T::result_type_for_const(constant))
            .unwrap_or_else(T::unknown_type);
        let fresh = const_var(editor, preheader, constant.clone(), var_type)?;
        editor.replace_uses_checked(*old, fresh);
    }
    // Copy after substitution so loop-defined operands name their materialized
    // preheader constants. Other incoming edges, including bypasses, keep theirs.
    editor.duplicate_phi_edge(exit, residual.exit_from, preheader)?;
    if !editor.redirect_terminator_target(preheader, residual.header, exit)? {
        return Err(Error::new(
            "the planned loop entry edge changed before residual application",
        ));
    }
    // The loop still ran; what it left is the residual above.
    editor.record_passed_through(preheader, exit, &residual.body);
    // The added normal edge now covers every exit phi, and all loop live-outs
    // have preheader definitions. CFG repair removes unreachable loop edges
    // and their old phi operands; full verification guards the whole transaction.
    editor.declare_repaired();
    Ok(())
}

/// Splits the edge each residual entered by a conditional block takes into
/// its loop ([`LoopResidual::split`]), and returns the new block by the edge
/// it was put on.
///
/// All splits come before any residual is applied: a loop entered from the
/// exit test of another the rewrite replaces leaves that loop's exit edge
/// through its split block, which the other residual then exits to.
///
/// # Errors
///
/// Returns an error if an edge planned for a split no longer splits.
fn split_entries<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    residuals: &[LoopResidual<'_, T>],
) -> Result<BTreeMap<(usize, usize), usize>> {
    let mut splits = BTreeMap::new();
    for residual in residuals.iter().filter(|residual| residual.split) {
        let block = editor
            .split_edges(&[residual.entry], residual.header)?
            .ok_or_else(|| Error::new("the planned loop entry edge no longer splits"))?;
        splits.insert((residual.entry, residual.header), block);
    }
    Ok(splits)
}

/// A value the program holds where the loop headed by `header` is entered
/// from `entry`, whose executed address lies in the realigned frame `root`,
/// and that address's offset in it: the lowest-numbered such pointer whose
/// definition dominates the entry, or such an integer where no pointer is
/// (`and` realigns the integer a pointer converts to).
///
/// A value defined inside another loop is that loop's: a rewrite may replace
/// that loop too, and what it defined goes with it.
fn realigned_base<T: Target>(
    ssa: &SsaFunction<T>,
    dominators: &DominatorTree,
    forest: &LoopForest,
    facts: &StaticFacts<T>,
    root: Root,
    (entry, header): (usize, usize),
) -> Option<(SsaVarId, i64)> {
    let candidates = (0..ssa.var_id_bound())
        .map(SsaVarId::from_index)
        .filter_map(|value| {
            let address = facts.value(value)?.address()?;
            let variable = ssa.variable(value)?;
            let defined = NodeId::new(variable.def_site().block());
            (address.root() == root
                && dominators.dominates(defined, NodeId::new(entry))
                && forest
                    .iter()
                    .all(|other| !other.contains(defined) || other.contains(NodeId::new(header))))
            .then(|| (value, address.offset(), T::is_pointer(variable.var_type())))
        });
    let mut first = None;
    for (value, offset, pointer) in candidates {
        if pointer {
            return Some((value, offset));
        }
        first = first.or(Some((value, offset)));
    }
    first
}

/// Emits the base pointer of a run in `block`, and returns the value holding
/// it.
fn materialize_base<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    block: usize,
    base: &Base,
) -> Result<SsaVarId> {
    match base {
        Base::Relative { base, offset } => ptr_add(editor, block, *base, *offset),
        // Named as the host's lift names an image address, where it can be.
        Base::Image { address } => {
            let value =
                T::symbol_at(*address).map_or(ConstValue::NativeUInt(*address), ConstValue::Symbol);
            let var_type = T::result_type_for_const(&value).unwrap_or_else(T::unknown_type);
            const_var(editor, block, value, var_type)
        }
    }
}

/// Emits `base + offset` in `block`, and returns the value holding it.
fn ptr_add<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    block: usize,
    base: SsaVarId,
    offset: i64,
) -> Result<SsaVarId> {
    if offset == 0 {
        return Ok(base);
    }
    let dest =
        editor.create_fresh_variable(VariableOrigin::Phi, 0, DefSite::entry(), T::unknown_type());
    editor.insert_before_terminator(
        block,
        SsaInstruction::synthetic(SsaOp::PtrAdd {
            dest,
            base,
            index: None,
            stride: 1,
            offset,
            result_type: T::unknown_type(),
        }),
    )?;
    Ok(dest)
}

/// Emits `Const(value)` of `var_type` in `block`, and returns the value
/// holding it.
fn const_var<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    block: usize,
    value: ConstValue<T>,
    var_type: T::Type,
) -> Result<SsaVarId> {
    let dest = editor.create_fresh_variable(VariableOrigin::Phi, 0, DefSite::entry(), var_type);
    editor.insert_before_terminator(
        block,
        SsaInstruction::synthetic(SsaOp::Const { dest, value }),
    )?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        num::{NonZeroU8, NonZeroU16},
        ops::Range,
        process,
    };

    use super::*;
    use crate::{
        analysis::{
            Machine, SsaVerifier, VerifyLevel,
            context::AnalysisContext,
            execution::{
                CallModel, ExecutionHost, FindingKind, LibraryModel, RealignedExtent,
                StaticExecution,
            },
            image::{ImageData, ImageRead, ImageRegion, Mutability},
        },
        ir::{
            Callee,
            division::DivisionRules,
            exception::HandlerKind,
            function::{SsaBlockBuilder, SsaDefSpec, SsaFunctionBuilder},
            ops::{AccessPermission, AtomicAccessWidth, CmpKind, CompareBlkData},
            value::ConstValue,
            variable::VariableRole,
        },
        pointer::PointerSize,
        target::{Endianness, TypeLayout},
        testing::{FactfulTarget, MACHINE, MockSummary, MockTarget, MockType},
    };

    /// Attests the fixtures' 128 bytes of accessible automatic frame storage.
    struct FrameHost;

    impl<T: Target> ExecutionHost<T> for FrameHost {
        fn stack_extent(
            &self,
            _body: &SsaFunction<T>,
            _method: Option<&T::MethodRef>,
        ) -> Option<Range<i64>> {
            Some(-128..0)
        }
    }

    /// States only the frame reserved below a realigned base, where it states
    /// one.
    struct RealignedHost(Option<RealignedExtent>);

    impl<T: Target> ExecutionHost<T> for RealignedHost {
        fn realigned_stack_extent(
            &self,
            _body: &SsaFunction<T>,
            _method: Option<&T::MethodRef>,
        ) -> Option<RealignedExtent> {
            self.0.clone()
        }
    }

    /// Two read-only bytes at `0x1000`.
    struct Image;

    impl ImageData for Image {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            if (0x2000..0x2010).contains(&address) {
                return Some(ImageRegion {
                    start: 0x2000,
                    end: 0x2010,
                    mutability: Mutability::Writable,
                });
            }
            (0x1000..0x1002).contains(&address).then_some(ImageRegion {
                start: 0x1000,
                end: 0x1002,
                mutability: Mutability::ReadOnly,
            })
        }

        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            const BYTES: [u8; 2] = [0x0f, 0xf0];
            let start = usize::try_from(address.checked_sub(0x1000)?).ok()?;
            BYTES.get(start..start.checked_add(len)?)
        }

        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    fn abort<E, R>(_: E) -> R {
        process::abort()
    }

    /// A host that decodes supplied byte types and optionally constructs new byte stores.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct ByteTarget<const BYTES: bool>;

    impl<const BYTES: bool> Target for ByteTarget<BYTES> {
        type TypeRef = u32;
        type MethodRef = u32;
        type FieldRef = u32;
        type SigRef = u32;
        type SymbolRef = u32;
        type ExceptionKind = ();
        type Type = MockType;
        type OriginalInstruction = ();
        type LocalSignature = ();
        type VariableFacts = ();
        type InstructionFacts = ();
        type FunctionFacts = MockSummary;
        type Capability = ();
        fn ptr_bytes(&self) -> u32 {
            8
        }
        fn synthetic_instruction() {}
        fn unknown_type() -> MockType {
            MockType::Unknown
        }
        fn type_layout(ty: &MockType, size: PointerSize) -> Option<TypeLayout> {
            MockTarget::type_layout(ty, size)
        }
        fn is_integer(ty: &MockType) -> bool {
            MockTarget::is_integer(ty)
        }
        fn is_floating(ty: &MockType) -> bool {
            MockTarget::is_floating(ty)
        }
        fn is_signed(ty: &MockType) -> bool {
            MockTarget::is_signed(ty)
        }
        fn is_pointer(ty: &MockType) -> bool {
            MockTarget::is_pointer(ty)
        }
        fn is_reference(ty: &MockType) -> bool {
            MockTarget::is_reference(ty)
        }
        fn is_unknown(ty: &MockType) -> bool {
            MockTarget::is_unknown(ty)
        }
        fn bit_width(ty: &MockType) -> Option<u32> {
            MockTarget::bit_width(ty)
        }
        fn integer_type(bits: u32, signed: bool) -> Option<MockType> {
            if bits == 8 && !BYTES {
                None
            } else {
                MockTarget::integer_type(bits, signed)
            }
        }
        fn instruction_mnemonic(_: &()) -> &'static str {
            "fixture"
        }
        fn instruction_rva(_: &()) -> u64 {
            0
        }
        fn handler_kind(_: &()) -> HandlerKind {
            HandlerKind::Catch
        }
        fn record_image_read(facts: &mut MockSummary, read: ImageRead<Self>) {
            facts.image_reads.push((read.address, read.len));
            if read.entered {
                facts.entered_reads.push(read.address);
            }
        }
    }

    /// Checks decoded writes independently of whether the host can synthesize byte stores.
    fn check_byte_representation<const BYTES: bool>() {
        let (mut ssa, _) = decoder_for::<ByteTarget<BYTES>>();
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete());
        let effect = facts.loop_effect(1).unwrap_or_else(|| abort(()));
        assert_eq!(
            effect
                .writes
                .iter()
                .flat_map(|(_, bytes)| bytes.iter().copied())
                .collect::<Vec<_>>(),
            [0x0f ^ 0x5a, 0xf0 ^ 0x5a]
        );
        let before = format!("{ssa:?}");
        assert_eq!(
            residualize_loops(&mut ssa, &facts).unwrap_or_else(abort),
            usize::from(BYTES)
        );
        if BYTES {
            let bytes: Vec<_> = ssa
                .block(0)
                .into_iter()
                .flat_map(|block| block.instructions())
                .filter_map(|instruction| match instruction.op() {
                    SsaOp::StoreIndirect { value, .. } => ssa.get_var_constant(*value)?.as_u64(),
                    _ => None,
                })
                .collect();
            assert_eq!(bytes, [u64::from(0x0fu8 ^ 0x5a), u64::from(0xf0u8 ^ 0x5a)]);
            assert_eq!(ssa.facts().image_reads, [(0x1000, 1), (0x1001, 1)]);
        } else {
            assert_eq!(format!("{ssa:?}"), before);
            assert!(ssa.facts().image_reads.is_empty());
        }
        ssa.validate().unwrap_or_else(abort);
    }

    /// A declined byte representation preserves every original store and its read evidence.
    #[test]
    fn residual_writes_require_a_supported_byte_representation() {
        check_byte_representation::<false>();
        check_byte_representation::<true>();
    }

    /// A host without byte synthesis can still remove read-only loops without byte stores.
    #[test]
    fn read_only_residuals_do_not_require_byte_synthesis() {
        let mut ssa = arranged_reads_for::<ByteTarget<false>>(false, false);
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete());
        assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 2);
        assert_eq!(ssa.facts().image_reads, [(0x1000, 1), (0x1001, 1)]);
        ssa.validate().unwrap_or_else(abort);
    }

    /// `for i in 0..4 { buf[sp-8+i] = source[i] ^ 0x5a }`, then return.
    /// `source` is `0x1000`, the two read-only bytes doubled by the loop.
    fn decoder() -> (SsaFunction<MockTarget>, SsaVarId) {
        decoder_for::<MockTarget>()
    }

    /// Builds the decoder for a host retaining its own function facts.
    fn decoder_for<T: Target<Type = MockType>>() -> (SsaFunction<T>, SsaVarId) {
        decoder_into::<T>(false)
    }

    /// Builds the decoder, storing through the frame `and sp, -16` realigns
    /// where `realigned`, and returns the value its stores are addressed from.
    fn decoder_into<T: Target<Type = MockType>>(realigned: bool) -> (SsaFunction<T>, SsaVarId) {
        decoder_after::<T>(realigned, |_| Ok(()))
    }

    /// Builds the decoder with `prelude` emitted first in its preheader.
    fn decoder_after<T: Target<Type = MockType>>(
        realigned: bool,
        prelude: impl FnOnce(&mut SsaBlockBuilder<'_, T>) -> Result<()>,
    ) -> (SsaFunction<T>, SsaVarId) {
        let mut builder = SsaFunctionBuilder::<T>::new(0, 0);
        let stack = {
            let ssa = builder.function_mut();
            let stack = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            if let Some(variable) = ssa.variable_mut(stack) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            stack
        };
        let (source, zero, one, count, key, frame) = builder
            .in_block(0, |b| {
                prelude(b)?;
                let source =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let zero = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(0))?;
                let one = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
                let count = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(2))?;
                let key = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x5a))?;
                let frame = if realigned {
                    let integer =
                        b.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::PtrToInt {
                            dest,
                            operand: stack,
                            target: MockType::I64,
                        })?;
                    let mask =
                        b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(-16))?;
                    let aligned = b.and(SsaDefSpec::tmp(MockType::I64), integer, mask)?;
                    b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::IntToPtr {
                        dest,
                        operand: aligned,
                        target: MockType::Ptr,
                    })?
                } else {
                    stack
                };
                b.jump(1)?;
                Ok((source, zero, one, count, key, frame))
            })
            .unwrap_or_else(abort);
        let index = builder
            .in_block(1, |b| b.empty_phi(SsaDefSpec::tmp(MockType::I64)))
            .unwrap_or_else(abort);
        let next = builder
            .in_block(1, |b| {
                let from = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: source,
                    index: Some(index),
                    stride: 1,
                    offset: 0,
                    result_type: MockType::Ptr,
                })?;
                let byte = b.load_indirect(SsaDefSpec::tmp(MockType::U8), from, MockType::U8)?;
                let plain = b.xor(SsaDefSpec::tmp(MockType::U8), byte, key)?;
                let to = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: frame,
                    index: Some(index),
                    stride: 1,
                    offset: -8,
                    result_type: MockType::Ptr,
                })?;
                b.store_indirect(to, plain, MockType::U8)?;
                let next = b.add(SsaDefSpec::tmp(MockType::I64), index, one)?;
                b.branch_cmp(next, count, CmpKind::Lt, false, 1, 2)?;
                Ok(next)
            })
            .unwrap_or_else(abort);
        builder
            .add_phi_operand(1, index, 0, zero)
            .unwrap_or_else(abort);
        builder
            .add_phi_operand(1, index, 1, next)
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        (builder.finish().unwrap_or_else(abort), frame)
    }

    /// How a decoding loop is entered other than by a jump of its own.
    #[derive(Debug, Clone, Copy)]
    enum Entered {
        /// From the exit test of a loop copying the image's bytes to the
        /// frame, which the decoding loop then decodes in place, as `-O2`
        /// lays out two loops one after the other.
        AfterLoop,
        /// From a test the loop's count decides, as a guard `if (n > 0)`
        /// before a rotated loop is.
        Guarded,
    }

    /// Builds a decoder whose loop is entered by a conditional edge
    /// ([`Entered`]), writing `0x0f ^ 0x5a, 0xf0 ^ 0x5a` at `sp - 8`.
    fn entered_decoder(entered: Entered) -> SsaFunction<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let stack = {
            let ssa = builder.function_mut();
            let stack = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            if let Some(variable) = ssa.variable_mut(stack) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            stack
        };
        let (decoder, end) = match entered {
            Entered::AfterLoop => (2, 3),
            Entered::Guarded => (1, 2),
        };
        let (source, zero, one, count, key) = builder
            .in_block(0, |b| {
                let source =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let zero = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(0))?;
                let one = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
                let count = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(2))?;
                let key = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x5a))?;
                match entered {
                    Entered::AfterLoop => b.jump(1)?,
                    Entered::Guarded => b.branch_cmp(zero, count, CmpKind::Lt, false, 1, end)?,
                }
                Ok((source, zero, one, count, key))
            })
            .unwrap_or_else(abort);
        // The image's bytes copied to the frame, a loop of its own.
        if let Entered::AfterLoop = entered {
            let index = builder
                .in_block(1, |b| b.empty_phi(SsaDefSpec::tmp(MockType::I64)))
                .unwrap_or_else(abort);
            let next = builder
                .in_block(1, |b| {
                    let from =
                        b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: source,
                            index: Some(index),
                            stride: 1,
                            offset: 0,
                            result_type: MockType::Ptr,
                        })?;
                    let byte =
                        b.load_indirect(SsaDefSpec::tmp(MockType::U8), from, MockType::U8)?;
                    let to = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: stack,
                        index: Some(index),
                        stride: 1,
                        offset: -8,
                        result_type: MockType::Ptr,
                    })?;
                    b.store_indirect(to, byte, MockType::U8)?;
                    let next = b.add(SsaDefSpec::tmp(MockType::I64), index, one)?;
                    b.branch_cmp(next, count, CmpKind::Lt, false, 1, decoder)?;
                    Ok(next)
                })
                .unwrap_or_else(abort);
            builder
                .add_phi_operand(1, index, 0, zero)
                .unwrap_or_else(abort);
            builder
                .add_phi_operand(1, index, 1, next)
                .unwrap_or_else(abort);
        }
        // The decoding loop: in place in the frame after the copy, straight
        // from the image under a guard.
        let index = builder
            .in_block(decoder, |b| b.empty_phi(SsaDefSpec::tmp(MockType::I64)))
            .unwrap_or_else(abort);
        let next = builder
            .in_block(decoder, |b| {
                let to = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: Some(index),
                    stride: 1,
                    offset: -8,
                    result_type: MockType::Ptr,
                })?;
                let from = match entered {
                    Entered::AfterLoop => to,
                    Entered::Guarded => {
                        b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: source,
                            index: Some(index),
                            stride: 1,
                            offset: 0,
                            result_type: MockType::Ptr,
                        })?
                    }
                };
                let byte = b.load_indirect(SsaDefSpec::tmp(MockType::U8), from, MockType::U8)?;
                let plain = b.xor(SsaDefSpec::tmp(MockType::U8), byte, key)?;
                b.store_indirect(to, plain, MockType::U8)?;
                let next = b.add(SsaDefSpec::tmp(MockType::I64), index, one)?;
                b.branch_cmp(next, count, CmpKind::Lt, false, decoder, end)?;
                Ok(next)
            })
            .unwrap_or_else(abort);
        let entry = match entered {
            Entered::AfterLoop => 1,
            Entered::Guarded => 0,
        };
        builder
            .add_phi_operand(decoder, index, entry, zero)
            .unwrap_or_else(abort);
        builder
            .add_phi_operand(decoder, index, decoder, next)
            .unwrap_or_else(abort);
        builder.in_block(end, |b| b.ret(None)).unwrap_or_else(abort);
        builder.finish().unwrap_or_else(abort)
    }

    /// The decoder as unoptimised code builds it: the index lives in an
    /// eight-byte slot of the frame, stored before the loop and loaded,
    /// advanced and stored back on every iteration. Nothing states the stack
    /// pointer's low bits, so only the frame proves the slot's accesses.
    fn slot_decoder() -> SsaFunction<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let stack = {
            let ssa = builder.function_mut();
            let stack = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            if let Some(variable) = ssa.variable_mut(stack) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            stack
        };
        let (slot, source, one, count, key) = builder
            .in_block(0, |b| {
                let slot = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -16,
                    result_type: MockType::Ptr,
                })?;
                let zero = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(0))?;
                b.store_indirect(slot, zero, MockType::I64)?;
                let source =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let one = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
                let count = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(2))?;
                let key = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x5a))?;
                b.jump(1)?;
                Ok((slot, source, one, count, key))
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let index = b.load_indirect(SsaDefSpec::tmp(MockType::I64), slot, MockType::I64)?;
                let from = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: source,
                    index: Some(index),
                    stride: 1,
                    offset: 0,
                    result_type: MockType::Ptr,
                })?;
                let byte = b.load_indirect(SsaDefSpec::tmp(MockType::U8), from, MockType::U8)?;
                let plain = b.xor(SsaDefSpec::tmp(MockType::U8), byte, key)?;
                let to = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: Some(index),
                    stride: 1,
                    offset: -32,
                    result_type: MockType::Ptr,
                })?;
                b.store_indirect(to, plain, MockType::U8)?;
                let next = b.add(SsaDefSpec::tmp(MockType::I64), index, one)?;
                b.store_indirect(slot, next, MockType::I64)?;
                b.branch_cmp(next, count, CmpKind::Lt, false, 1, 2)
            })
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        builder.finish().unwrap_or_else(abort)
    }

    /// An image-reading instruction placed in a one-iteration natural loop.
    #[derive(Debug, Clone, Copy)]
    enum ReadFixture {
        Scalar { entered: bool },
        Copy { reverse: bool },
        Move,
        Fill,
        Compare { reverse: bool },
        Scan { reverse: bool },
        Load { reverse: bool },
        EnteredLoad { reverse: bool },
    }

    /// Builds an exact read footprint without relying on interpreter results as its oracle.
    fn reading_loop(kind: ReadFixture) -> SsaFunction<FactfulTarget> {
        let reverse = matches!(
            kind,
            ReadFixture::Copy { reverse: true }
                | ReadFixture::Compare { reverse: true }
                | ReadFixture::Scan { reverse: true }
                | ReadFixture::Load { reverse: true }
                | ReadFixture::EnteredLoad { reverse: true }
        );
        let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
        let (source, dest, other, count, zero, needle, backward) = builder
            .in_block(0, |b| {
                let source = b.const_value(
                    SsaDefSpec::tmp(MockType::U64),
                    ConstValue::U64(if reverse { 0x1001 } else { 0x1000 }),
                )?;
                let dest = b.const_value(
                    SsaDefSpec::tmp(MockType::U64),
                    ConstValue::U64(if reverse { 0x2001 } else { 0x2000 }),
                )?;
                let other = b.const_value(
                    SsaDefSpec::tmp(MockType::U64),
                    ConstValue::U64(if reverse { 0x1000 } else { 0x1001 }),
                )?;
                let count = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(2))?;
                let zero = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                let needle = b.const_value(
                    SsaDefSpec::tmp(MockType::U8),
                    ConstValue::U8(if reverse { 0xf0 } else { 0x0f }),
                )?;
                let backward = b.const_value(
                    SsaDefSpec::tmp(MockType::Bool),
                    ConstValue::from_bool(reverse),
                )?;
                b.jump(1)?;
                Ok((source, dest, other, count, zero, needle, backward))
            })
            .unwrap_or_else(abort);
        let loaded = builder
            .in_block(1, |b| {
                let loaded = match kind {
                    ReadFixture::Scalar { .. } => {
                        let loaded =
                            b.load_indirect(SsaDefSpec::tmp(MockType::U16), source, MockType::U16)?;
                        b.store_indirect(dest, loaded, MockType::U16)?;
                        Some(loaded)
                    }
                    ReadFixture::Copy { .. } => {
                        b.emit_no_defs(SsaOp::CopyBlk {
                            dest_addr: dest,
                            src_addr: source,
                            size: count,
                            element_bytes: 1,
                            backward: Some(backward),
                            src_access: AccessQualifiers::FLAT,
                            dest_access: AccessQualifiers::FLAT,
                        })?;
                        None
                    }
                    ReadFixture::Fill => {
                        b.init_blk(dest, needle, count, Some(backward))?;
                        None
                    }
                    ReadFixture::Move => {
                        b.emit_no_defs(SsaOp::MoveBlk {
                            dest_addr: dest,
                            src_addr: source,
                            size: count,
                            src_access: AccessQualifiers::FLAT,
                            dest_access: AccessQualifiers::FLAT,
                        })?;
                        None
                    }
                    ReadFixture::Compare { .. } => {
                        let defs = [
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::U32),
                        ];
                        b.emit_many_defs(&defs, |defs| {
                            SsaOp::CompareBlk(Box::new(CompareBlkData {
                                remaining: defs[0],
                                left_next: defs[1],
                                right_next: defs[2],
                                flags: defs[3],
                                left: source,
                                right: other,
                                count,
                                flags_in: Some(zero),
                                backward: Some(backward),
                                element_bytes: 1,
                                while_equal: true,
                                left_access: AccessQualifiers::FLAT,
                                right_access: AccessQualifiers::FLAT,
                            }))
                        })?;
                        None
                    }
                    ReadFixture::Scan { .. } => {
                        let defs = [
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::U32),
                        ];
                        b.emit_many_defs(&defs, |defs| SsaOp::ScanBlk {
                            remaining: defs[0],
                            next: defs[1],
                            flags: defs[2],
                            address: source,
                            value: needle,
                            count,
                            flags_in: Some(zero),
                            backward: Some(backward),
                            element_bytes: 1,
                            while_equal: false,
                            access: AccessQualifiers::FLAT,
                        })?;
                        None
                    }
                    ReadFixture::Load { .. } | ReadFixture::EnteredLoad { .. } => {
                        let defs = [
                            SsaDefSpec::tmp(MockType::U8),
                            SsaDefSpec::tmp(MockType::U64),
                        ];
                        let values = b.emit_many_defs(&defs, |defs| SsaOp::LoadBlk {
                            value: defs[0],
                            next: defs[1],
                            address: source,
                            count,
                            value_in: Some(needle),
                            backward: Some(backward),
                            element_bytes: 1,
                            access: AccessQualifiers::FLAT,
                        })?;
                        b.store_indirect(dest, values[0], MockType::U8)?;
                        Some(values[0])
                    }
                };
                b.branch_cmp(zero, zero, CmpKind::Lt, false, 1, 2)?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        builder
            .in_block(2, |b| {
                if matches!(
                    kind,
                    ReadFixture::Scalar { entered: true } | ReadFixture::EnteredLoad { .. }
                ) {
                    b.call_void(
                        Callee::Indirect {
                            fptr: loaded.unwrap_or_else(|| abort(())),
                            signature: 0,
                        },
                        vec![],
                    )?;
                }
                b.ret(None)
            })
            .unwrap_or_else(abort);
        builder.finish().unwrap_or_else(abort)
    }

    /// Loop replacement retains each actual image read, including early stops and reverse walks.
    #[test]
    fn residual_loops_retain_exact_scalar_and_block_reads() {
        let cases = [
            (ReadFixture::Scalar { entered: false }, vec![(0x1000, 2)]),
            (ReadFixture::Scalar { entered: true }, vec![(0x1000, 2)]),
            (ReadFixture::Copy { reverse: false }, vec![(0x1000, 2)]),
            (ReadFixture::Copy { reverse: true }, vec![(0x1000, 2)]),
            (ReadFixture::Move, vec![(0x1000, 2)]),
            (
                ReadFixture::Compare { reverse: false },
                vec![(0x1000, 1), (0x1001, 1)],
            ),
            (
                ReadFixture::Compare { reverse: true },
                vec![(0x1000, 1), (0x1001, 1)],
            ),
            (ReadFixture::Scan { reverse: false }, vec![(0x1000, 1)]),
            (ReadFixture::Scan { reverse: true }, vec![(0x1001, 1)]),
            (
                ReadFixture::Load { reverse: false },
                vec![(0x1000, 1), (0x1001, 1)],
            ),
            (
                ReadFixture::Load { reverse: true },
                vec![(0x1000, 1), (0x1001, 1)],
            ),
        ];
        for (kind, expected) in cases {
            let mut ssa = reading_loop(kind);
            let context = AnalysisContext::new(MACHINE).with_image(&Image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert!(facts.is_complete(), "{kind:?}: {:?}", facts.completion());
            assert!(ssa.facts().image_reads.is_empty());
            assert_eq!(
                residualize_loops(&mut ssa, &facts).unwrap_or_else(abort),
                1,
                "{kind:?}"
            );
            assert_eq!(ssa.facts().image_reads, expected, "{kind:?}");
            assert_eq!(
                ssa.facts().entered_reads,
                if matches!(kind, ReadFixture::Scalar { entered: true }) {
                    vec![0x1000]
                } else {
                    vec![]
                }
            );
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            fold_constants(&mut ssa, &facts).unwrap_or_else(abort);
            assert_eq!(
                ssa.facts().image_reads,
                expected,
                "duplicate after fold: {kind:?}"
            );
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 0);
            assert_eq!(ssa.facts().image_reads, expected);
            ssa.validate().unwrap_or_else(abort);
        }
    }

    /// Independent symbol coordinates, mapping and access guarantees for a loop fixture.
    #[derive(Debug, Clone, Copy)]
    struct AlignmentCase {
        shift: Option<i64>,
        source: u32,
        destination: u32,
        other: u32,
        alignment: Option<u8>,
    }

    /// Read-only source and writable destination extents, with independently stated numeric mapping.
    struct AlignmentImage(Option<i64>);

    impl ImageData for AlignmentImage {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn number_offset(&self) -> Option<i64> {
            self.0
        }
        fn region(&self, at: u64) -> Option<ImageRegion> {
            if (0x1000..0x1080).contains(&at) {
                Some(ImageRegion {
                    start: 0x1000,
                    end: 0x1080,
                    mutability: Mutability::ReadOnly,
                })
            } else if (0x2000..0x2080).contains(&at) {
                Some(ImageRegion {
                    start: 0x2000,
                    end: 0x2080,
                    mutability: Mutability::Writable,
                })
            } else {
                None
            }
        }
        fn bytes(&self, at: u64, len: usize) -> Option<&[u8]> {
            const CONTENTS: [u8; 128] = [0x11; 128];
            let start = usize::try_from(at.checked_sub(0x1000)?).ok()?;
            CONTENTS.get(start..start.checked_add(len)?)
        }
    }

    /// Sources with independently specified readability and initial knowledge.
    #[derive(Debug, Clone, Copy)]
    enum TransferSource {
        Image {
            visible: usize,
            writable: bool,
            numeric: bool,
        },
        Stack {
            initialized_elements: usize,
            overlap: bool,
        },
    }

    /// Supplies a readable source extent separately from whether its bytes are immutable.
    struct TransferImage {
        visible: usize,
        writable: bool,
        shift: Option<i64>,
    }

    impl ImageData for TransferImage {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn number_offset(&self) -> Option<i64> {
            self.shift
        }
        fn region(&self, at: u64) -> Option<ImageRegion> {
            if (0x1000..0x1000 + self.visible as u64).contains(&at) {
                Some(ImageRegion {
                    start: 0x1000,
                    end: 0x1000 + self.visible as u64,
                    mutability: if self.writable {
                        Mutability::Writable
                    } else {
                        Mutability::ReadOnly
                    },
                })
            } else if (0x2000..0x2080).contains(&at) {
                Some(ImageRegion {
                    start: 0x2000,
                    end: 0x2080,
                    mutability: Mutability::Writable,
                })
            } else {
                None
            }
        }
        fn bytes(&self, at: u64, len: usize) -> Option<&[u8]> {
            const BYTES: [u8; 32] = [0x11; 32];
            let offset = usize::try_from(at.checked_sub(0x1000)?).ok()?;
            let end = offset.checked_add(len)?;
            (end <= self.visible)
                .then(|| BYTES.get(offset..end))
                .flatten()
        }
    }

    /// Transfer entry points that share the source-read contract.
    #[derive(Clone, Copy)]
    enum TransferKind {
        Stream,
        Snapshot,
        Model,
    }

    /// Binds one catalog copy model without substituting a callee body.
    struct TransferModels;

    impl ExecutionHost<FactfulTarget> for TransferModels {
        fn stack_extent(
            &self,
            body: &SsaFunction<FactfulTarget>,
            method: Option<&u32>,
        ) -> Option<Range<i64>> {
            FrameHost.stack_extent(body, method)
        }
        fn model(&self, method: &u32) -> Option<&dyn CallModel<FactfulTarget>> {
            match method {
                17 => Some(&LibraryModel::Copy),
                18 => Some(&LibraryModel::Length),
                _ => None,
            }
        }
    }

    /// Overwrites a transfer's entire destination after its source reads have executed.
    fn overwritten_transfer_loop(
        source: TransferSource,
        width: u32,
        kind: TransferKind,
        reverse: bool,
        empty: bool,
    ) -> (SsaFunction<FactfulTarget>, SsaVarId) {
        let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
        let stack = matches!(source, TransferSource::Stack { .. }).then(|| {
            let stack = builder.function_mut().create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            builder
                .function_mut()
                .variable_mut(stack)
                .unwrap_or_else(|| abort(()))
                .set_role(VariableRole::EntryStackPointer);
            stack
        });
        let (first, destination, source_low, dest_low, size, full_size, fill, zero, backward) =
            builder
                .in_block(0, |b| {
                    let advance = if reverse { 2 * width } else { 0 };
                    let first = match source {
                        TransferSource::Image { numeric, .. } => b.const_value(
                            SsaDefSpec::tmp(MockType::Ptr),
                            if numeric {
                                ConstValue::U64(u64::from(0x1000 + advance))
                            } else {
                                ConstValue::Symbol(0x1000 + advance)
                            },
                        )?,
                        TransferSource::Stack { .. } => {
                            b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                                dest,
                                base: stack.unwrap_or_else(|| abort(())),
                                index: None,
                                stride: 1,
                                offset: -64 + i64::from(advance),
                                result_type: MockType::Ptr,
                            })?
                        }
                    };
                    let destination =
                        if matches!(source, TransferSource::Stack { overlap: true, .. }) {
                            b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                                dest,
                                base: first,
                                index: None,
                                stride: 1,
                                offset: if reverse {
                                    -i64::from(width)
                                } else {
                                    i64::from(width)
                                },
                                result_type: MockType::Ptr,
                            })?
                        } else {
                            b.const_value(
                                SsaDefSpec::tmp(MockType::Ptr),
                                ConstValue::Symbol(0x2000 + advance),
                            )?
                        };
                    let source_low =
                        b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: first,
                            index: None,
                            stride: 1,
                            offset: -i64::from(advance),
                            result_type: MockType::Ptr,
                        })?;
                    let dest_low =
                        b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: destination,
                            index: None,
                            stride: 1,
                            offset: -i64::from(advance),
                            result_type: MockType::Ptr,
                        })?;
                    if let TransferSource::Stack {
                        initialized_elements,
                        ..
                    } = source
                    {
                        let byte =
                            b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x11))?;
                        for element in 0..initialized_elements {
                            for index in 0..width {
                                let at = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| {
                                    SsaOp::PtrAdd {
                                        dest,
                                        base: first,
                                        index: None,
                                        stride: 1,
                                        offset: if reverse {
                                            -(element as i64) * i64::from(width)
                                        } else {
                                            element as i64 * i64::from(width)
                                        } + i64::from(index),
                                        result_type: MockType::Ptr,
                                    }
                                })?;
                                b.store_indirect(at, byte, MockType::U8)?;
                            }
                        }
                    }
                    let size = b.const_value(
                        SsaDefSpec::tmp(MockType::U64),
                        ConstValue::U64(if empty { 0 } else { u64::from(3 * width) }),
                    )?;
                    let full_size = b.const_value(
                        SsaDefSpec::tmp(MockType::U64),
                        ConstValue::U64(u64::from(3 * width)),
                    )?;
                    let fill =
                        b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x77))?;
                    let zero = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                    let backward = b.const_value(
                        SsaDefSpec::tmp(MockType::Bool),
                        ConstValue::from_bool(reverse),
                    )?;
                    b.jump(1)?;
                    Ok((
                        first,
                        destination,
                        source_low,
                        dest_low,
                        size,
                        full_size,
                        fill,
                        zero,
                        backward,
                    ))
                })
                .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let access = AccessQualifiers::FLAT
                    .with_alignment(NonZeroU8::new(1).unwrap_or_else(|| abort(())));
                if matches!(kind, TransferKind::Model) {
                    b.call_void(Callee::Direct(17), vec![dest_low, source_low, size])?;
                } else if matches!(kind, TransferKind::Snapshot) {
                    b.emit_no_defs(SsaOp::MoveBlk {
                        dest_addr: dest_low,
                        src_addr: source_low,
                        size,
                        src_access: access,
                        dest_access: access,
                    })?;
                } else {
                    b.emit_no_defs(SsaOp::CopyBlk {
                        dest_addr: destination,
                        src_addr: first,
                        size,
                        element_bytes: width,
                        backward: Some(backward),
                        src_access: access,
                        dest_access: access,
                    })?;
                }
                b.emit_no_defs(SsaOp::InitBlk {
                    dest_addr: dest_low,
                    value: fill,
                    size: full_size,
                    backward: None,
                    access: AccessQualifiers::FLAT,
                })?;
                b.branch_cmp(zero, zero, CmpKind::Lt, false, 1, 2)
            })
            .unwrap_or_else(abort);
        let loaded = builder
            .in_block(2, |b| {
                let value =
                    b.load_indirect(SsaDefSpec::tmp(MockType::U8), dest_low, MockType::U8)?;
                b.ret(Some(value))?;
                Ok(value)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        ssa.validate().unwrap_or_else(abort);
        (ssa, loaded)
    }

    /// Known final bytes do not erase an unproved read; readable unknown and expanded streams remain valid.
    #[test]
    fn residual_loops_preserve_overwritten_transfer_reads() {
        for width in [1, 2, 4] {
            let full = 3 * width as usize;
            for source in [
                TransferSource::Image {
                    visible: 0,
                    writable: false,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: width as usize,
                    writable: false,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: full,
                    writable: false,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: full,
                    writable: true,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: full,
                    writable: true,
                    numeric: true,
                },
                TransferSource::Stack {
                    initialized_elements: 0,
                    overlap: false,
                },
                TransferSource::Stack {
                    initialized_elements: 1,
                    overlap: false,
                },
                TransferSource::Stack {
                    initialized_elements: 3,
                    overlap: false,
                },
                TransferSource::Stack {
                    initialized_elements: 1,
                    overlap: true,
                },
                TransferSource::Stack {
                    initialized_elements: 3,
                    overlap: true,
                },
            ] {
                for buffered in [false, true] {
                    for reverse in [false, true] {
                        for empty in [false, true] {
                            for shift in [None, Some(0)] {
                                let (visible, writable, expected) = match source {
                                    TransferSource::Image {
                                        visible,
                                        writable,
                                        numeric,
                                    } => (
                                        visible,
                                        writable,
                                        visible == full && (!numeric || shift.is_some()),
                                    ),
                                    TransferSource::Stack {
                                        initialized_elements,
                                        overlap,
                                    } => (
                                        0,
                                        false,
                                        initialized_elements == 3
                                            || overlap && !buffered && initialized_elements == 1,
                                    ),
                                };
                                let image = TransferImage {
                                    visible,
                                    writable,
                                    shift,
                                };
                                let context = AnalysisContext::new(MACHINE).with_image(&image);
                                let (mut ssa, loaded) = overwritten_transfer_loop(
                                    source,
                                    width,
                                    if buffered {
                                        TransferKind::Snapshot
                                    } else {
                                        TransferKind::Stream
                                    },
                                    reverse,
                                    empty,
                                );
                                let facts = StaticExecution::new(&ssa, &context)
                                    .with_host(&FrameHost)
                                    .run();
                                assert!(
                                    facts.is_complete(),
                                    "{source:?}: {:?}",
                                    facts.completion()
                                );
                                assert_eq!(facts.constant(loaded), Some(&ConstValue::U8(0x77)));
                                let before = format!("{ssa:?}");
                                let changed =
                                    residualize_loops(&mut ssa, &facts).unwrap_or_else(abort);
                                assert_eq!(
                                    changed,
                                    usize::from(expected || empty),
                                    "{source:?} width={width} buffered={buffered} reverse={reverse} empty={empty} shift={shift:?}"
                                );
                                if expected || empty {
                                    let after = StaticExecution::new(&ssa, &context)
                                        .with_host(&FrameHost)
                                        .run();
                                    assert!(after.is_complete());
                                    assert_eq!(after.image_writes(), facts.image_writes());
                                    assert_eq!(after.constant(loaded), Some(&ConstValue::U8(0x77)));
                                } else {
                                    assert_eq!(format!("{ssa:?}"), before);
                                    assert!(ssa.facts().image_reads.is_empty());
                                }
                                ssa.validate().unwrap_or_else(abort);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Catalog copy calls must preserve unreadable source reads even after all output is overwritten.
    #[test]
    fn residual_loops_preserve_overwritten_model_reads() {
        for width in [1, 2, 4] {
            let full = 3 * width as usize;
            for source in [
                TransferSource::Image {
                    visible: 0,
                    writable: false,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: width as usize,
                    writable: false,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: full,
                    writable: false,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: full,
                    writable: true,
                    numeric: false,
                },
                TransferSource::Image {
                    visible: full,
                    writable: true,
                    numeric: true,
                },
                TransferSource::Stack {
                    initialized_elements: 1,
                    overlap: false,
                },
                TransferSource::Stack {
                    initialized_elements: 3,
                    overlap: false,
                },
                TransferSource::Stack {
                    initialized_elements: 1,
                    overlap: true,
                },
                TransferSource::Stack {
                    initialized_elements: 3,
                    overlap: true,
                },
            ] {
                for reverse in [false, true] {
                    for empty in [false, true] {
                        for shift in [None, Some(0)] {
                            let (visible, writable, expected) = match source {
                                TransferSource::Image {
                                    visible,
                                    writable,
                                    numeric,
                                } => (
                                    visible,
                                    writable,
                                    visible == full && (!numeric || shift.is_some()),
                                ),
                                TransferSource::Stack {
                                    initialized_elements,
                                    ..
                                } => (0, false, initialized_elements == 3),
                            };
                            let image = TransferImage {
                                visible,
                                writable,
                                shift,
                            };
                            let context = AnalysisContext::new(MACHINE).with_image(&image);
                            let (mut ssa, loaded) = overwritten_transfer_loop(
                                source,
                                width,
                                TransferKind::Model,
                                reverse,
                                empty,
                            );
                            let facts = StaticExecution::new(&ssa, &context)
                                .with_host(&TransferModels)
                                .run();
                            assert!(facts.is_complete(), "{source:?}: {:?}", facts.completion());
                            assert_eq!(facts.constant(loaded), Some(&ConstValue::U8(0x77)));
                            let before = format!("{ssa:?}");
                            assert_eq!(
                                residualize_loops(&mut ssa, &facts).unwrap_or_else(abort),
                                usize::from(expected || empty),
                                "{source:?} width={width} reverse={reverse} empty={empty} shift={shift:?}"
                            );
                            if expected || empty {
                                let after = StaticExecution::new(&ssa, &context)
                                    .with_host(&TransferModels)
                                    .run();
                                assert!(after.is_complete());
                                assert_eq!(after.image_writes(), facts.image_writes());
                                assert_eq!(after.constant(loaded), Some(&ConstValue::U8(0x77)));
                            } else {
                                assert_eq!(format!("{ssa:?}"), before);
                                assert!(ssa.facts().image_reads.is_empty());
                            }
                            ssa.validate().unwrap_or_else(abort);
                        }
                    }
                }
            }
        }
    }

    /// Read-model inputs whose address and bounded contents are independently established.
    #[derive(Clone, Copy, Debug)]
    enum LengthInput {
        Derived,
        Missing,
        Unknown,
        Known,
    }

    /// Builds a read-only catalog call whose result is unused before known output writes.
    fn unused_length_loop(input: LengthInput) -> (SsaFunction<FactfulTarget>, SsaVarId) {
        let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
        let stack = builder.function_mut().create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        builder
            .function_mut()
            .variable_mut(stack)
            .unwrap_or_else(|| abort(()))
            .set_role(VariableRole::EntryStackPointer);
        let unknown = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let (arguments, destination, fill, zero) = builder
            .in_block(0, |b| {
                let pointer = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -64,
                    result_type: MockType::Ptr,
                })?;
                let first = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(b'x'))?;
                let nul = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                b.store_indirect(pointer, first, MockType::U8)?;
                let last = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: pointer,
                    index: None,
                    stride: 1,
                    offset: 1,
                    result_type: MockType::Ptr,
                })?;
                b.store_indirect(last, nul, MockType::U8)?;
                let arguments = match input {
                    LengthInput::Derived => {
                        let mask = b.const_value(
                            SsaDefSpec::tmp(MockType::Ptr),
                            ConstValue::NativeInt(-16),
                        )?;
                        let pointer =
                            b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::And {
                                dest,
                                left: pointer,
                                right: mask,
                                flags: None,
                            })?;
                        vec![pointer]
                    }
                    LengthInput::Missing => vec![],
                    LengthInput::Unknown => vec![unknown],
                    LengthInput::Known => vec![pointer],
                };
                let destination =
                    b.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::Symbol(0x2000))?;
                let fill = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x77))?;
                let zero = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                b.jump(1)?;
                Ok((arguments, destination, fill, zero))
            })
            .unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                b.call_void(Callee::Direct(18), arguments)?;
                b.store_indirect(destination, fill, MockType::U8)?;
                b.branch_cmp(zero, zero, CmpKind::Lt, false, 1, 2)
            })
            .unwrap_or_else(abort);
        let value = builder
            .in_block(2, |b| {
                let value =
                    b.load_indirect(SsaDefSpec::tmp(MockType::U8), destination, MockType::U8)?;
                b.ret(Some(value))?;
                Ok(value)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        ssa.validate().unwrap_or_else(abort);
        (ssa, value)
    }

    /// An unused string-length result does not authorize bypassing an unresolved source read.
    #[test]
    fn residual_loops_preserve_unresolved_length_reads() {
        let image = AlignmentImage(Some(0));
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        for input in [
            LengthInput::Derived,
            LengthInput::Missing,
            LengthInput::Unknown,
            LengthInput::Known,
        ] {
            let (mut ssa, value) = unused_length_loop(input);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&TransferModels)
                .run();
            assert!(facts.is_complete(), "{input:?}: {:?}", facts.completion());
            assert_eq!(facts.constant(value), Some(&ConstValue::U8(0x77)));
            let before = format!("{ssa:?}");
            let expected = usize::from(matches!(input, LengthInput::Known));
            assert_eq!(
                residualize_loops(&mut ssa, &facts).unwrap_or_else(abort),
                expected,
                "{input:?}"
            );
            if expected == 0 {
                assert_eq!(format!("{ssa:?}"), before);
                assert!(ssa.facts().image_reads.is_empty());
            } else {
                let after = StaticExecution::new(&ssa, &context)
                    .with_host(&TransferModels)
                    .run();
                assert!(after.is_complete());
                assert_eq!(after.image_writes(), facts.image_writes());
                assert_eq!(after.constant(value), Some(&ConstValue::U8(0x77)));
            }
            ssa.validate().unwrap_or_else(abort);
        }
    }

    /// Memory kernels whose typed accesses must survive when alignment is unproved.
    #[derive(Debug, Clone, Copy)]
    enum AlignmentKind {
        Read,
        Write,
        Fill,
        Copy,
        Move,
        Compare,
        Scan,
        Load,
    }

    /// Returns an explicitly specified scalar representation and its all-0x11 byte pattern.
    fn alignment_pattern(width: u32) -> (MockType, ConstValue<FactfulTarget>) {
        match width {
            1 => (MockType::U8, ConstValue::U8(0x11)),
            2 => (MockType::U16, ConstValue::U16(0x1111)),
            4 => (MockType::U32, ConstValue::U32(0x1111_1111)),
            _ => abort(()),
        }
    }

    /// Builds a one-visit natural loop; private sources are seeded only through byte writes.
    fn alignment_loop(
        kind: AlignmentKind,
        width: u32,
        reverse: bool,
        empty: bool,
        private: bool,
        case: AlignmentCase,
    ) -> (SsaFunction<FactfulTarget>, Option<SsaVarId>) {
        let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
        let stack = private.then(|| {
            let stack = builder.function_mut().create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::Ptr,
            );
            builder
                .function_mut()
                .variable_mut(stack)
                .unwrap_or_else(|| abort(()))
                .set_role(VariableRole::EntryStackPointer);
            stack
        });
        let (ty, pattern) = alignment_pattern(width);
        let (addresses, count, size, zero, needle, backward) = builder
            .in_block(0, |b| {
                let mut addresses = Vec::new();
                for (slot, (base, offset)) in [
                    (0x1000u32, case.source),
                    (0x2000, case.destination),
                    (0x1040, case.other),
                ]
                .into_iter()
                .enumerate()
                {
                    let advance = if reverse { width } else { 0 };
                    let pointer = if let Some(stack) = stack {
                        let base = [-256i64, -128, -192][slot];
                        b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                            dest,
                            base: stack,
                            index: None,
                            stride: 1,
                            offset: base + i64::from(offset + advance),
                            result_type: MockType::Ptr,
                        })?
                    } else {
                        b.const_value(
                            SsaDefSpec::tmp(MockType::Ptr),
                            ConstValue::Symbol(base + offset + advance),
                        )?
                    };
                    addresses.push(pointer);
                }
                let zero = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                let count = b.const_value(
                    SsaDefSpec::tmp(MockType::U64),
                    ConstValue::U64(if empty { 0 } else { 2 }),
                )?;
                let size = b.const_value(
                    SsaDefSpec::tmp(MockType::U64),
                    ConstValue::U64(if empty { 0 } else { u64::from(2 * width) }),
                )?;
                let needle = b.const_value(SsaDefSpec::tmp(ty), pattern)?;
                let backward = b.const_value(
                    SsaDefSpec::tmp(MockType::Bool),
                    ConstValue::from_bool(reverse),
                )?;
                if private {
                    let byte =
                        b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x11))?;
                    for source in [addresses[0], addresses[2]] {
                        for index in 0..2 * width {
                            let at =
                                b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                                    dest,
                                    base: source,
                                    index: None,
                                    stride: 1,
                                    offset: i64::from(index)
                                        - if reverse { i64::from(width) } else { 0 },
                                    result_type: MockType::Ptr,
                                })?;
                            b.store_indirect(at, byte, MockType::U8)?;
                        }
                    }
                }
                b.jump(1)?;
                Ok((addresses, count, size, zero, needle, backward))
            })
            .unwrap_or_else(abort);
        let access = case.alignment.map_or(AccessQualifiers::FLAT, |alignment| {
            AccessQualifiers::FLAT
                .with_alignment(NonZeroU8::new(alignment).unwrap_or_else(|| abort(())))
        });
        let mut loaded = None;
        builder
            .in_block(1, |b| {
                match kind {
                    AlignmentKind::Read => {
                        loaded =
                            Some(b.emit_def(SsaDefSpec::tmp(ty), |dest| SsaOp::LoadIndirect {
                                dest,
                                addr: addresses[0],
                                value_type: ty,
                                access,
                            })?);
                    }
                    AlignmentKind::Write => b.emit_no_defs(SsaOp::StoreIndirect {
                        addr: addresses[1],
                        value: needle,
                        value_type: ty,
                        access,
                    })?,
                    AlignmentKind::Fill => b.emit_no_defs(SsaOp::InitBlk {
                        dest_addr: addresses[1],
                        value: needle,
                        size,
                        backward: Some(backward),
                        access,
                    })?,
                    AlignmentKind::Copy => b.emit_no_defs(SsaOp::CopyBlk {
                        dest_addr: addresses[1],
                        src_addr: addresses[0],
                        size,
                        element_bytes: width,
                        backward: Some(backward),
                        src_access: access,
                        dest_access: access,
                    })?,
                    AlignmentKind::Move => b.emit_no_defs(SsaOp::MoveBlk {
                        dest_addr: addresses[1],
                        src_addr: addresses[0],
                        size,
                        src_access: access,
                        dest_access: access,
                    })?,
                    AlignmentKind::Compare => {
                        let defs = [
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::Ptr),
                            SsaDefSpec::tmp(MockType::Ptr),
                            SsaDefSpec::tmp(MockType::U32),
                        ];
                        b.emit_many_defs(&defs, |defs| {
                            SsaOp::CompareBlk(Box::new(CompareBlkData {
                                remaining: defs[0],
                                left_next: defs[1],
                                right_next: defs[2],
                                flags: defs[3],
                                left: addresses[0],
                                right: addresses[2],
                                count,
                                flags_in: Some(zero),
                                backward: Some(backward),
                                element_bytes: width,
                                while_equal: true,
                                left_access: access,
                                right_access: access,
                            }))
                        })?;
                    }
                    AlignmentKind::Scan => {
                        let defs = [
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::Ptr),
                            SsaDefSpec::tmp(MockType::U32),
                        ];
                        b.emit_many_defs(&defs, |defs| SsaOp::ScanBlk {
                            remaining: defs[0],
                            next: defs[1],
                            flags: defs[2],
                            address: addresses[0],
                            value: needle,
                            count,
                            flags_in: Some(zero),
                            backward: Some(backward),
                            element_bytes: width,
                            while_equal: true,
                            access,
                        })?;
                    }
                    AlignmentKind::Load => {
                        let defs = [SsaDefSpec::tmp(ty), SsaDefSpec::tmp(MockType::Ptr)];
                        let results = b.emit_many_defs(&defs, |defs| SsaOp::LoadBlk {
                            value: defs[0],
                            next: defs[1],
                            address: addresses[0],
                            count,
                            value_in: Some(needle),
                            backward: Some(backward),
                            element_bytes: width,
                            access,
                        })?;
                        loaded = Some(results[0]);
                    }
                }
                b.branch_cmp(zero, zero, CmpKind::Lt, false, 1, 2)
            })
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        ssa.validate().unwrap_or_else(abort);
        (ssa, loaded)
    }

    /// Known values do not authorize bypassing misaligned or unknown-base typed memory accesses.
    #[test]
    fn residual_loops_require_proved_memory_alignment() {
        let cases = [
            (Some(0), 0, 0, 0, None),
            (Some(0), 1, 0, 0, None),
            (Some(0), 0, 1, 0, None),
            (Some(0), 0, 0, 1, None),
            (Some(1), 0, 0, 0, None),
            (Some(1), 1, 1, 1, None),
            (None, 0, 0, 0, None),
            (Some(0), 1, 1, 1, Some(1)),
            (None, 1, 1, 1, Some(1)),
            (Some(0), 2, 2, 2, Some(2)),
            (Some(1), 2, 2, 2, Some(2)),
            (None, 0, 0, 0, Some(2)),
        ];
        for kind in [
            AlignmentKind::Read,
            AlignmentKind::Write,
            AlignmentKind::Fill,
            AlignmentKind::Copy,
            AlignmentKind::Move,
            AlignmentKind::Compare,
            AlignmentKind::Scan,
            AlignmentKind::Load,
        ] {
            for width in [1u32, 2, 4] {
                for (shift, source, destination, other, alignment) in cases {
                    let case = AlignmentCase {
                        shift,
                        source,
                        destination,
                        other,
                        alignment,
                    };
                    for private in [false, true] {
                        for reverse in [false, true] {
                            if reverse
                                && matches!(
                                    kind,
                                    AlignmentKind::Read
                                        | AlignmentKind::Write
                                        | AlignmentKind::Move
                                )
                            {
                                continue;
                            }
                            for empty in [false, true] {
                                if empty
                                    && matches!(kind, AlignmentKind::Read | AlignmentKind::Write)
                                {
                                    continue;
                                }
                                let unit = if matches!(kind, AlignmentKind::Move) {
                                    1
                                } else {
                                    width
                                };
                                let required = u32::from(alignment.unwrap_or(unit as u8)).min(unit);
                                // A private destination lies in the frame
                                // the host attests (`-128..0`), which proves
                                // an access there; the other operands lie
                                // below it and prove nothing by it.
                                let aligned = |base: u32, offset: u32| {
                                    required == 1
                                        || private && base == 0x2000
                                        || !private
                                            && shift.is_some_and(|shift| {
                                                (i128::from(base + offset) - i128::from(shift))
                                                    % i128::from(required)
                                                    == 0
                                            })
                                };
                                let expected = empty
                                    || match kind {
                                        AlignmentKind::Read
                                        | AlignmentKind::Scan
                                        | AlignmentKind::Load => aligned(0x1000, source),
                                        AlignmentKind::Write | AlignmentKind::Fill => {
                                            aligned(0x2000, destination)
                                        }
                                        AlignmentKind::Copy | AlignmentKind::Move => {
                                            aligned(0x1000, source) && aligned(0x2000, destination)
                                        }
                                        AlignmentKind::Compare => {
                                            aligned(0x1000, source) && aligned(0x1040, other)
                                        }
                                    };
                                let (mut ssa, loaded) =
                                    alignment_loop(kind, width, reverse, empty, private, case);
                                let image = AlignmentImage(case.shift);
                                let context = AnalysisContext::new(MACHINE).with_image(&image);
                                let facts = StaticExecution::new(&ssa, &context)
                                    .with_host(&FrameHost)
                                    .run();
                                assert!(
                                    facts.is_complete(),
                                    "{kind:?} {case:?}: {:?}",
                                    facts.completion()
                                );
                                if let Some(loaded) = loaded {
                                    assert_eq!(
                                        facts.constant(loaded),
                                        Some(&alignment_pattern(width).1)
                                    );
                                }
                                let before = format!("{ssa:?}");
                                let writes = facts.image_writes();
                                let changed =
                                    residualize_loops(&mut ssa, &facts).unwrap_or_else(abort);
                                assert_eq!(
                                    changed,
                                    usize::from(expected),
                                    "{kind:?} width={width} {case:?} private={private} reverse={reverse} empty={empty}"
                                );
                                if expected {
                                    let after = StaticExecution::new(&ssa, &context)
                                        .with_host(&FrameHost)
                                        .run();
                                    assert!(after.is_complete());
                                    assert_eq!(after.image_writes(), writes);
                                } else {
                                    assert_eq!(format!("{ssa:?}"), before);
                                    assert!(ssa.facts().image_reads.is_empty());
                                }
                                ssa.validate().unwrap_or_else(abort);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Qualifies one access in source order, counting both sides of block operations.
    fn qualify_access(
        ssa: &mut SsaFunction<FactfulTarget>,
        selected: usize,
        qualified: AccessQualifiers,
    ) -> usize {
        let mut count = 0;
        let mut visit = |access: &mut AccessQualifiers| {
            if count == selected {
                *access = qualified;
            }
            count += 1;
        };
        for (_, _, instruction) in ssa.iter_instructions_mut() {
            match instruction.op_mut() {
                SsaOp::LoadIndirect { access, .. }
                | SsaOp::StoreIndirect { access, .. }
                | SsaOp::InitBlk { access, .. }
                | SsaOp::ScanBlk { access, .. }
                | SsaOp::LoadBlk { access, .. } => visit(access),
                SsaOp::CopyBlk {
                    src_access,
                    dest_access,
                    ..
                }
                | SsaOp::MoveBlk {
                    src_access,
                    dest_access,
                    ..
                } => {
                    visit(src_access);
                    visit(dest_access);
                }
                SsaOp::CompareBlk(data) => {
                    visit(&mut data.left_access);
                    visit(&mut data.right_access);
                }
                _ => {}
            }
        }
        count
    }

    /// An unsupported direction stays unknown even when addresses and bytes are known.
    fn unknown_direction(ssa: &mut SsaFunction<FactfulTarget>) {
        let unknown = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Bool,
        );
        for (_, _, instruction) in ssa.iter_instructions_mut() {
            match instruction.op_mut() {
                SsaOp::InitBlk { backward, .. }
                | SsaOp::CopyBlk { backward, .. }
                | SsaOp::ScanBlk { backward, .. }
                | SsaOp::LoadBlk { backward, .. } => *backward = Some(unknown),
                SsaOp::CompareBlk(data) => data.backward = Some(unknown),
                _ => {}
            }
        }
    }

    /// Checks refusal without permitting partial edits or leaked source-read metadata.
    fn assert_loop_refused(mut ssa: SsaFunction<FactfulTarget>) {
        ssa.recompute_uses();
        ssa.validate().unwrap_or_else(|error| panic!("{error:?}"));
        let before = format!("{ssa:?}");
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(
            residualize_loops(&mut ssa, &facts).unwrap_or_else(|error| panic!("{error:?}")),
            0
        );
        assert_eq!(format!("{ssa:?}"), before);
        assert!(ssa.facts().image_reads.is_empty());
    }

    /// Byte equality does not prove that an access with different permissions cannot fault.
    #[test]
    fn residual_loops_preserve_unproved_access_permissions() {
        for kind in [
            ReadFixture::Scalar { entered: false },
            ReadFixture::Fill,
            ReadFixture::Copy { reverse: false },
            ReadFixture::Move,
            ReadFixture::Compare { reverse: false },
            ReadFixture::Scan { reverse: false },
            ReadFixture::Load { reverse: false },
        ] {
            let mut ordinary = reading_loop(kind);
            let count = qualify_access(&mut ordinary, usize::MAX, AccessQualifiers::FLAT);
            let context = AnalysisContext::new(MACHINE).with_image(&Image);
            let facts = StaticExecution::new(&ordinary, &context)
                .with_host(&FrameHost)
                .run();
            assert_eq!(
                residualize_loops(&mut ordinary, &facts)
                    .unwrap_or_else(|error| panic!("{error:?}")),
                1,
                "{kind:?}"
            );
            for selected in 0..count {
                for permission in [
                    AccessPermission::Unprivileged,
                    AccessPermission::Execute,
                    AccessPermission::ShadowStack,
                    AccessPermission::UnprivilegedShadowStack,
                ] {
                    let mut ssa = reading_loop(kind);
                    qualify_access(&mut ssa, selected, AccessQualifiers::new(None, permission));
                    assert_loop_refused(ssa);
                }
            }
        }
    }

    /// Even an unused result cannot justify deleting an unsupported potentially faulting read.
    #[test]
    fn residual_loops_preserve_unsupported_block_reads() {
        for kind in [
            ReadFixture::Compare { reverse: false },
            ReadFixture::Scan { reverse: false },
            ReadFixture::Load { reverse: false },
        ] {
            for direction in [false, true] {
                let mut ssa = reading_loop(kind);
                // Keep the read results unused, so no unknown store masks its effect proof.
                for block in ssa.blocks_mut() {
                    block.instructions_mut().retain(|instruction| {
                        !matches!(instruction.op(), SsaOp::StoreIndirect { .. })
                    });
                }
                if direction {
                    unknown_direction(&mut ssa);
                } else {
                    qualify_access(
                        &mut ssa,
                        0,
                        AccessQualifiers::new(NonZeroU16::new(7), AccessPermission::Current),
                    );
                }
                assert_loop_refused(ssa);
            }
        }
    }

    /// An empty transfer needs no address-space, permission or direction proof and preserves inputs.
    #[test]
    fn residual_empty_blocks_require_no_memory_access_proof() {
        for volatile in [false, true] {
            for kind in [
                ReadFixture::Fill,
                ReadFixture::Copy { reverse: false },
                ReadFixture::Move,
                ReadFixture::Compare { reverse: false },
                ReadFixture::Scan { reverse: false },
                ReadFixture::Load { reverse: false },
            ] {
                let mut ssa = reading_loop(kind);
                let count = qualify_access(&mut ssa, usize::MAX, AccessQualifiers::FLAT);
                for selected in 0..count {
                    qualify_access(
                        &mut ssa,
                        selected,
                        AccessQualifiers {
                            volatile,
                            ..AccessQualifiers::new(
                                NonZeroU16::new(7),
                                AccessPermission::Unprivileged,
                            )
                        },
                    );
                }
                unknown_direction(&mut ssa);
                let mut expected = Vec::new();
                for (_, _, instruction) in ssa.iter_instructions_mut() {
                    match instruction.op_mut() {
                        SsaOp::Const {
                            value: ConstValue::U64(value),
                            ..
                        } if *value == 2 => *value = 0,
                        SsaOp::CompareBlk(data) => expected.extend([
                            (data.remaining, ConstValue::U64(0)),
                            (data.left_next, ConstValue::U64(0x1000)),
                            (data.right_next, ConstValue::U64(0x1001)),
                            (data.flags, ConstValue::U64(0)),
                        ]),
                        SsaOp::ScanBlk {
                            remaining,
                            next,
                            flags,
                            ..
                        } => expected.extend([
                            (*remaining, ConstValue::U64(0)),
                            (*next, ConstValue::U64(0x1000)),
                            (*flags, ConstValue::U64(0)),
                        ]),
                        SsaOp::LoadBlk { value, next, .. } => expected.extend([
                            (*value, ConstValue::U8(0x0f)),
                            (*next, ConstValue::U64(0x1000)),
                        ]),
                        _ => {}
                    }
                }
                // This fixture's scalar store also carries a qualifier; the empty block stands alone.
                for block in ssa.blocks_mut() {
                    block.instructions_mut().retain(|instruction| {
                        !matches!(instruction.op(), SsaOp::StoreIndirect { .. })
                    });
                }
                ssa.recompute_uses();
                ssa.validate().unwrap_or_else(|error| panic!("{error:?}"));
                let context = AnalysisContext::new(MACHINE).with_image(&Image);
                let facts = StaticExecution::new(&ssa, &context)
                    .with_host(&FrameHost)
                    .run();
                assert!(facts.is_complete());
                for (value, expected) in expected {
                    assert_eq!(facts.constant(value), Some(&expected), "{kind:?}");
                }
                assert_eq!(
                    residualize_loops(&mut ssa, &facts).unwrap_or_else(|error| panic!("{error:?}")),
                    usize::from(!volatile),
                    "{kind:?}, volatile={volatile}"
                );
                assert!(ssa.facts().image_reads.is_empty());
                ssa.validate().unwrap_or_else(|error| panic!("{error:?}"));
            }
        }
    }

    /// A write into immutable image storage faults the path that makes it: the
    /// run completes with the fault as its finding, and since no path gets
    /// past the write, it writes no image byte, makes no call and rewrites
    /// nothing; the values computed before the write stay observed.
    #[test]
    fn read_only_image_writes_fault_their_path() {
        for kind in [
            ReadFixture::Scalar { entered: false },
            ReadFixture::Fill,
            ReadFixture::Copy { reverse: false },
            ReadFixture::Copy { reverse: true },
            ReadFixture::Move,
            ReadFixture::Load { reverse: false },
        ] {
            let mut ssa = reading_loop(kind);
            for (_, _, instruction) in ssa.iter_instructions_mut() {
                if let SsaOp::Const {
                    value: ConstValue::U64(value),
                    ..
                } = instruction.op_mut()
                {
                    if *value == 0x2000 {
                        *value = 0x1000;
                    }
                    if *value == 0x2001 {
                        *value = 0x1001;
                    }
                }
            }
            let before = format!("{ssa:?}");
            let context = AnalysisContext::new(MACHINE).with_image(&Image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert!(facts.is_complete(), "{kind:?}: {:?}", facts.completion());
            assert!(
                facts
                    .findings()
                    .iter()
                    .any(|finding| finding.kind == FindingKind::ReadOnlyWrite),
                "{kind:?}"
            );
            assert!(facts.image_writes().is_empty());
            assert!(facts.calls().is_empty());
            assert_eq!(
                residualize_loops(&mut ssa, &facts).unwrap_or_else(|error| panic!("{error:?}")),
                0
            );
            assert_eq!(format!("{ssa:?}"), before);
        }
    }

    /// A repeated load's last element supplies its result; earlier elements do not become call targets.
    #[test]
    fn repeated_loads_attribute_entered_value_to_final_element() {
        for reverse in [false, true] {
            let mut ssa = reading_loop(ReadFixture::EnteredLoad { reverse });
            let context = AnalysisContext::new(MACHINE).with_image(&Image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert!(facts.is_complete());
            assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 1);
            assert_eq!(
                ssa.facts().entered_reads,
                [if reverse { 0x1000 } else { 0x1001 }]
            );
            assert_eq!(ssa.facts().image_reads, [(0x1000, 1), (0x1001, 1)]);
            ssa.validate().unwrap_or_else(abort);
        }
    }

    /// A changing loop load keeps its distinct source cells after the loop is removed.
    #[test]
    fn residual_loops_retain_each_iteration_read() {
        let (mut ssa, _) = decoder_for::<FactfulTarget>();
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 1);
        assert_eq!(ssa.facts().image_reads, [(0x1000, 1), (0x1001, 1)]);
    }

    /// Builds nested or disjoint loop reads, optionally with the first loop
    /// entered from two blocks, which no one residual can be placed for.
    fn arranged_reads(nested: bool, two_entries: bool) -> SsaFunction<FactfulTarget> {
        arranged_reads_for::<FactfulTarget>(nested, two_entries)
    }

    /// Builds read-only loops for a host with independent rewrite capabilities.
    fn arranged_reads_for<T: Target<Type = MockType>>(
        nested: bool,
        two_entries: bool,
    ) -> SsaFunction<T> {
        let mut builder = SsaFunctionBuilder::<T>::new(0, 0);
        let (first, second, zero) = builder
            .in_block(0, |b| {
                let first =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let second =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1001))?;
                let zero = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                if two_entries {
                    let yes = b.const_value(SsaDefSpec::tmp(MockType::Bool), ConstValue::True)?;
                    b.branch(yes, 1, 5)?;
                } else {
                    b.jump(1)?;
                }
                Ok((first, second, zero))
            })
            .unwrap_or_else(abort);
        if nested {
            builder.in_block(1, |b| b.jump(2)).unwrap_or_else(abort);
            builder
                .in_block(2, |b| {
                    b.load_indirect(SsaDefSpec::tmp(MockType::U8), first, MockType::U8)?;
                    b.branch_cmp(zero, zero, CmpKind::Lt, false, 2, 3)
                })
                .unwrap_or_else(abort);
            builder
                .in_block(3, |b| b.branch_cmp(zero, zero, CmpKind::Lt, false, 1, 4))
                .unwrap_or_else(abort);
        } else {
            builder
                .in_block(1, |b| {
                    b.load_indirect(SsaDefSpec::tmp(MockType::U8), first, MockType::U8)?;
                    b.branch_cmp(zero, zero, CmpKind::Lt, false, 1, 2)
                })
                .unwrap_or_else(abort);
            builder.in_block(2, |b| b.jump(3)).unwrap_or_else(abort);
            builder
                .in_block(3, |b| {
                    b.load_indirect(SsaDefSpec::tmp(MockType::U8), second, MockType::U8)?;
                    b.branch_cmp(zero, zero, CmpKind::Lt, false, 3, 4)
                })
                .unwrap_or_else(abort);
        }
        builder.in_block(4, |b| b.ret(None)).unwrap_or_else(abort);
        if two_entries {
            builder.in_block(5, |b| b.jump(1)).unwrap_or_else(abort);
        }
        builder.finish().unwrap_or_else(abort)
    }

    /// An eligible outer residual carries each nested read once; disjoint residuals retain both.
    #[test]
    fn residual_loops_publish_nonoverlapping_read_witnesses() {
        for nested in [false, true] {
            let mut ssa = arranged_reads(nested, false);
            let context = AnalysisContext::new(MACHINE).with_image(&Image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert!(facts.is_complete());
            assert_eq!(facts.decoded_loops().len(), 2);
            assert_eq!(
                residualize_loops(&mut ssa, &facts).unwrap_or_else(abort),
                if nested { 1 } else { 2 }
            );
            assert_eq!(
                ssa.facts().image_reads,
                if nested {
                    vec![(0x1000, 1)]
                } else {
                    vec![(0x1000, 1), (0x1001, 1)]
                }
            );
            let before = ssa.facts().image_reads.clone();
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            fold_constants(&mut ssa, &facts).unwrap_or_else(abort);
            assert_eq!(ssa.facts().image_reads, before);
            ssa.validate().unwrap_or_else(abort);
        }
    }

    /// A refused loop contributes no metadata while a disjoint eligible loop still commits.
    #[test]
    fn residual_loops_withhold_reads_for_refused_plans() {
        let mut ssa = arranged_reads(false, true);
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert_eq!(facts.decoded_loops(), [1, 3]);
        assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 1);
        assert_eq!(ssa.facts().image_reads, [(0x1001, 1)]);
        assert!(
            ssa.block(0)
                .is_some_and(|block| block.successors().contains(&1))
        );
        ssa.validate().unwrap_or_else(abort);
    }

    /// A stale entry edge rolls back inserted byte writes and withholds every read witness.
    #[test]
    fn residual_loops_rollback_writes_and_read_metadata_together() {
        let (mut ssa, _) = decoder_for::<FactfulTarget>();
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        let forest = LoopAnalyzer::new(&ssa).analyze();
        let cfg = SsaCfg::from_ssa(&ssa);
        let mut plan = Planner {
            ssa: &ssa,
            facts: &facts,
            control: None,
        }
        .one_loop(&cfg, &compute_dominators(&cfg, NodeId::new(0)), &forest, 1)
        .unwrap_or_else(|| abort(()));
        drop(cfg);
        assert_eq!(
            plan.writes
                .iter()
                .flat_map(|writes| writes.runs.iter())
                .flat_map(|write| write.bytes.iter().copied())
                .collect::<Vec<_>>(),
            [0x0f ^ 0x5a, 0xf0 ^ 0x5a]
        );
        let entered = OnceCell::new();
        let reads: Vec<_> = facts
            .loop_reads(1)
            .map(|(key, value)| {
                residual_read(&ssa, key, value, &entered).unwrap_or_else(|| abort(()))
            })
            .collect();
        assert_eq!(reads.len(), 2);
        plan.header = ssa.block_count();
        let before = format!("{ssa:?}");
        assert!(
            commit_static_facts(
                &mut ssa,
                Vec::new(),
                vec![plan],
                reads,
                None,
                SsaRollbackPolicy::OnFailure,
            )
            .is_err()
        );
        assert_eq!(format!("{ssa:?}"), before);
        assert!(ssa.facts().image_reads.is_empty());
        ssa.validate().unwrap_or_else(abort);
    }

    /// An absent intermediate element cannot justify skipping a repeated load.
    #[test]
    fn repeated_loads_need_the_complete_read_extent() {
        struct Hole;
        impl ImageData for Hole {
            fn endianness(&self) -> Endianness {
                Endianness::Little
            }
            fn region(&self, at: u64) -> Option<ImageRegion> {
                Image.region(at)
            }
            fn bytes(&self, at: u64, len: usize) -> Option<&[u8]> {
                if at == 0x1000 {
                    None
                } else {
                    Image.bytes(at, len)
                }
            }
            fn number_offset(&self) -> Option<i64> {
                Some(0)
            }
        }
        let mut ssa = reading_loop(ReadFixture::Load { reverse: false });
        let context = AnalysisContext::new(MACHINE).with_image(&Hole);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete());
        assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 0);
        assert!(ssa.facts().image_reads.is_empty());
    }

    /// A decoded loop is replaced by the bytes it wrote, its header is no
    /// longer reached from the entry, and the result still verifies.
    #[test]
    fn a_decoded_loop_becomes_its_bytes() {
        let (mut ssa, stack) = decoder();
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        let replaced = residualize_loops(&mut ssa, &facts).unwrap_or_else(abort);
        assert_eq!(replaced, 1);
        ssa.validate().unwrap_or_else(abort);
        // The preheader now jumps past the loop, straight to the exit.
        let preheader = ssa.block(0).unwrap_or_else(|| abort(()));
        assert!(preheader.successors().contains(&2));
        assert!(!preheader.successors().contains(&1));
        // Its bytes are `0x0f, 0xf0` XOR `0x5a`, stored at `sp - 8`.
        let stored: Vec<u8> = preheader
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.op() {
                SsaOp::StoreIndirect { value, .. } => ssa.get_var_constant(*value)?.as_u64(),
                _ => None,
            })
            .map(|byte| byte as u8)
            .collect();
        assert_eq!(stored, vec![0x0f ^ 0x5a, 0xf0 ^ 0x5a]);
        let _ = stack;
    }

    /// **A multi-result operation in a loop's preheader is replaced beside the
    /// loop's residual.** The loop's bytes are stored into the preheader the
    /// operation lies in, by positions planned on the same snapshot; the
    /// operation's second result follows its first once those stores are
    /// placed, and the body verifies.
    #[test]
    fn a_multi_result_operation_is_replaced_beside_a_loop_residual() {
        let halves = Cell::new(None);
        let (mut ssa, _) = decoder_after::<MockTarget>(false, |b| {
            let left = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(200))?;
            let right = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
            halves.set(Some(b.emit_two_defs(
                SsaDefSpec::tmp(MockType::U8),
                SsaDefSpec::tmp(MockType::U8),
                |low, high| SsaOp::WideMul {
                    low,
                    high,
                    left,
                    right,
                    unsigned: true,
                },
            )?));
            Ok(())
        });
        let (low, high) = halves.get().unwrap_or_else(|| abort(()));
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert!(apply_static_facts(&mut ssa, &facts).unwrap_or_else(abort) >= 2);
        ssa.validate().unwrap_or_else(abort);
        let preheader = ssa.block(0).unwrap_or_else(|| abort(()));
        assert!(!preheader.successors().contains(&1));
        let at = |value: SsaVarId| {
            preheader
                .instructions()
                .iter()
                .position(|instruction| instruction.defs().any(|def| def == value))
        };
        assert_eq!(ssa.get_var_constant(low), Some(&ConstValue::U8(0x78)));
        assert_eq!(ssa.get_var_constant(high), Some(&ConstValue::U8(0x05)));
        assert_eq!(at(low).map(|index| index.saturating_add(1)), at(high));
    }

    /// **A loop decoding into a realigned frame becomes its bytes, stored
    /// through the realigned pointer.** The frame `and sp, -16` realigns lies
    /// at no offset of the entry stack, so the bytes are addressed from the
    /// program's own realigned value, where the loop was entered.
    #[test]
    fn a_decoded_loop_into_a_realigned_frame_becomes_its_bytes() {
        let (mut ssa, frame) = decoder_into::<MockTarget>(true);
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete(), "{:?}", facts.completion());
        assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 1);
        ssa.validate().unwrap_or_else(abort);
        let preheader = ssa.block(0).unwrap_or_else(|| abort(()));
        assert!(!preheader.successors().contains(&1));
        let stored: Vec<u8> = preheader
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.op() {
                SsaOp::StoreIndirect { value, .. } => ssa.get_var_constant(*value)?.as_u64(),
                _ => None,
            })
            .map(|byte| byte as u8)
            .collect();
        assert_eq!(stored, vec![0x0f ^ 0x5a, 0xf0 ^ 0x5a]);
        // Every residual pointer is formed from the realigned value.
        assert!(preheader.instructions().iter().any(|instruction| matches!(
            instruction.op(),
            SsaOp::PtrAdd { base, .. } if *base == frame
        )));
    }

    /// **The frame an activation reserves below its realigned base proves the
    /// loop's writes there.** With nothing stated of the entry frame, the
    /// realigned extent alone admits the decoder and the rewrite replaces it;
    /// one naming another realignment, or a range short of the buffer, proves
    /// nothing, and neither does no extent at all.
    #[test]
    fn a_realigned_extent_proves_the_frame_below_its_base() {
        for (extent, decoded) in [
            (
                Some(RealignedExtent {
                    at: 0,
                    range: -16..0,
                }),
                true,
            ),
            (
                Some(RealignedExtent {
                    at: -8,
                    range: -16..0,
                }),
                false,
            ),
            (
                Some(RealignedExtent {
                    at: 0,
                    range: -7..0,
                }),
                false,
            ),
            (None, false),
        ] {
            let (mut ssa, _) = decoder_into::<MockTarget>(true);
            let image = Image;
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&RealignedHost(extent.clone()))
                .run();
            assert!(facts.is_complete(), "{extent:?}: {:?}", facts.completion());
            assert_eq!(!facts.decoded_loops().is_empty(), decoded, "{extent:?}");
            let replaced = residualize_loops(&mut ssa, &facts).unwrap_or_else(abort);
            assert_eq!(replaced, usize::from(decoded), "{extent:?}");
        }
    }

    /// **An unoptimised decoder's frame slots are proved by its frame.** The
    /// index's eight-byte slot is read and written on every iteration, and
    /// nothing states the stack pointer's low bits, so no alignment of the
    /// slot is known; the frame the activation reserved holds it, and the IR
    /// already treats that frame as never faulting. Without a frame the
    /// slot's accesses prove nothing and the loop is left as it is.
    #[test]
    fn a_frame_slot_index_is_proved_by_its_frame() {
        for (framed, decoded) in [(true, true), (false, false)] {
            let mut ssa = slot_decoder();
            let image = Image;
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            let execution = StaticExecution::new(&ssa, &context);
            let facts = if framed {
                execution.with_host(&FrameHost).run()
            } else {
                execution.with_host(&RealignedHost(None)).run()
            };
            assert!(facts.is_complete(), "{framed}: {:?}", facts.completion());
            assert_eq!(!facts.decoded_loops().is_empty(), decoded, "{framed}");
            let replaced = residualize_loops(&mut ssa, &facts).unwrap_or_else(abort);
            assert_eq!(replaced, usize::from(decoded), "{framed}");
        }
    }

    /// **A decoded loop entered by a conditional edge becomes its bytes.**
    /// The residual cannot go into a block that branches elsewhere too, so
    /// the edge into the loop is split and the new block holds it: a loop
    /// entered from the exit test of the loop before it, and one under a
    /// guard. Where the loop before is replaced as well, its exit goes on
    /// through the split block, so both residuals run, in order.
    #[test]
    fn a_decoded_loop_entered_by_a_conditional_edge_becomes_its_bytes() {
        for (entered, replaced, expected) in [
            (
                Entered::AfterLoop,
                2,
                vec![0x0f, 0xf0, 0x0f ^ 0x5a, 0xf0 ^ 0x5a],
            ),
            (Entered::Guarded, 1, vec![0x0f ^ 0x5a, 0xf0 ^ 0x5a]),
        ] {
            let mut ssa = entered_decoder(entered);
            let image = Image;
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert!(facts.is_complete(), "{entered:?}: {:?}", facts.completion());
            assert_eq!(facts.decoded_loops().len(), replaced, "{entered:?}");
            assert_eq!(
                residualize_loops(&mut ssa, &facts).unwrap_or_else(abort),
                replaced,
                "{entered:?}"
            );
            ssa.validate().unwrap_or_else(abort);
            // The bytes the reachable blocks store, breadth first from the
            // entry, each a constant: no load of the image or the frame is
            // left, so neither loop is.
            let mut stored = Vec::new();
            let mut queue = VecDeque::from([0]);
            let mut seen = BTreeSet::from([0]);
            while let Some(block) = queue.pop_front() {
                let current = ssa.block(block).unwrap_or_else(|| abort(()));
                for instruction in current.instructions() {
                    match instruction.op() {
                        SsaOp::StoreIndirect { value, .. } => stored.push(
                            ssa.get_var_constant(*value)
                                .and_then(ConstValue::as_u64)
                                .unwrap_or_else(|| abort(())),
                        ),
                        SsaOp::LoadIndirect { .. } => abort(()),
                        _ => {}
                    }
                }
                for next in current.successors() {
                    if seen.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
            assert_eq!(stored, expected, "{entered:?}");
        }
    }

    /// The preheader of a loop replaced by its residual records the loop's
    /// host blocks as passed through: the loop ran, and left only that.
    #[test]
    fn a_residualized_loop_is_passed_through_by_its_preheader() {
        let (mut ssa, _) = decoder();
        for (index, host) in [(0, 0x100), (1, 0x110), (2, 0x120)] {
            if let Some(block) = ssa.block_mut(index) {
                block.provenance_mut().carry(host);
            }
        }
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert_eq!(residualize_loops(&mut ssa, &facts).unwrap_or_else(abort), 1);
        let preheader = ssa.block(0).unwrap_or_else(|| abort(()));
        assert_eq!(preheader.provenance().carried(), &[0x100]);
        assert_eq!(preheader.provenance().passed(), &[0x110]);
    }

    /// A load of read-only data and the arithmetic over it fold to what they
    /// compute; a call keeps its effect.
    #[test]
    fn what_is_constant_on_every_run_folds() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let (loaded, mixed, called) = builder
            .in_block(0, |b| {
                let at = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let loaded = b.load_indirect(SsaDefSpec::tmp(MockType::U16), at, MockType::U16)?;
                let mask =
                    b.const_value(SsaDefSpec::tmp(MockType::U16), ConstValue::U16(0x00ff))?;
                let mixed = b.and(SsaDefSpec::tmp(MockType::U16), loaded, mask)?;
                let called = b.call(
                    SsaDefSpec::tmp(MockType::U16),
                    Callee::Direct(9),
                    vec![mixed],
                )?;
                b.ret(Some(called))?;
                Ok((loaded, mixed, called))
            })
            .unwrap_or_else(abort);
        let mut ssa = builder.finish().unwrap_or_else(abort);
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        let folded = fold_constants(&mut ssa, &facts).unwrap_or_else(abort);
        assert_eq!(folded, 2);
        assert_eq!(ssa.get_var_constant(loaded), Some(&ConstValue::U16(0xf00f)));
        assert_eq!(ssa.get_var_constant(mixed), Some(&ConstValue::U16(0x000f)));
        assert!(ssa.get_var_constant(called).is_none());
    }

    /// Successful visits do not license deleting a later overflow at the same definition.
    #[test]
    fn checked_replacement_requires_nontrapping_evidence_from_every_visit() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let (zero, one, two, max) = builder
            .in_block(0, |b| {
                let zero = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                let one = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
                let two = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
                let max = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(255))?;
                b.jump(1)?;
                Ok((zero, one, two, max))
            })
            .unwrap_or_else(abort);
        let index = builder
            .in_block(1, |b| b.empty_phi(SsaDefSpec::tmp(MockType::U8)))
            .unwrap_or_else(abort);
        let (checked, next) = builder
            .in_block(1, |b| {
                let checked = b.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| SsaOp::AddOvf {
                    dest,
                    left: max,
                    right: index,
                    unsigned: true,
                    flags: None,
                })?;
                let next = b.add(SsaDefSpec::tmp(MockType::U8), index, one)?;
                b.branch_cmp(next, two, CmpKind::Lt, true, 1, 2)?;
                Ok((checked, next))
            })
            .unwrap_or_else(abort);
        builder
            .add_phi_operand(1, index, 0, zero)
            .unwrap_or_else(abort);
        builder
            .add_phi_operand(1, index, 1, next)
            .unwrap_or_else(abort);
        builder
            .in_block(2, |b| {
                b.call_void(crate::ir::Callee::Direct(99), vec![checked])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let mut ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete());
        assert_eq!(facts.constant(checked), Some(&ConstValue::U8(255)));
        assert_eq!(facts.replacement_proof(checked), None);
        assert_eq!(facts.usage().traps, 1);
        assert!(facts.calls().is_empty());
        fold_constants(&mut ssa, &facts).unwrap_or_else(abort);
        assert!(matches!(
            ssa.recorded_definition(checked).map(|def| def.op()),
            Some(SsaOp::AddOvf { .. })
        ));
        let rewritten = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(rewritten.is_complete());
        assert_eq!(rewritten.usage().traps, 1);
        assert!(rewritten.calls().is_empty());
    }

    /// An independently safe checked operation remains eligible for replacement.
    #[test]
    fn safe_checked_arithmetic_can_be_replaced() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let checked = builder
            .in_block(0, |b| {
                let one = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
                let checked = b.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| SsaOp::AddOvf {
                    dest,
                    left: one,
                    right: one,
                    unsigned: true,
                    flags: None,
                })?;
                b.ret(Some(checked))?;
                Ok(checked)
            })
            .unwrap_or_else(abort);
        let mut ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert_eq!(
            facts.replacement_proof(checked),
            Some(ReplacementProof::Completed)
        );
        assert_eq!(fold_constants(&mut ssa, &facts).unwrap_or_else(abort), 1);
        assert_eq!(ssa.get_var_constant(checked), Some(&ConstValue::U8(2)));
    }

    /// The division family under test: a quotient, a remainder, or both
    /// results of one double-width division.
    #[derive(Debug, Clone, Copy)]
    enum Division {
        Quotient,
        Remainder,
        Wide,
    }

    impl Division {
        /// Emits `dividend / divisor` in this form, returning the value a
        /// replacement would define.
        fn emit(
            self,
            b: &mut SsaBlockBuilder<'_, MockTarget>,
            dividend: SsaVarId,
            divisor: SsaVarId,
            zero: SsaVarId,
        ) -> Result<SsaVarId> {
            match self {
                Self::Quotient => b.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| SsaOp::Div {
                    dest,
                    left: dividend,
                    right: divisor,
                    unsigned: true,
                    flags: None,
                }),
                Self::Remainder => b.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| SsaOp::Rem {
                    dest,
                    left: dividend,
                    right: divisor,
                    unsigned: true,
                    flags: None,
                }),
                Self::Wide => {
                    let (quotient, _) = b.emit_two_defs(
                        SsaDefSpec::tmp(MockType::U8),
                        SsaDefSpec::tmp(MockType::U8),
                        |quotient, remainder| SsaOp::WideDiv {
                            quotient,
                            remainder,
                            high: zero,
                            low: dividend,
                            divisor,
                            unsigned: true,
                        },
                    )?;
                    Ok(quotient)
                }
            }
        }
    }

    /// A division replaced by its constant must have completed on every
    /// visit: a later visit dividing by zero keeps it, on every form.
    #[test]
    fn division_replacement_requires_a_completed_division_on_every_visit() {
        for division in [Division::Quotient, Division::Remainder, Division::Wide] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            let (zero, one, two, six) = builder
                .in_block(0, |b| {
                    let zero = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                    let one = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
                    let two = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
                    let six = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(6))?;
                    b.jump(1)?;
                    Ok((zero, one, two, six))
                })
                .unwrap_or_else(abort);
            let index = builder
                .in_block(1, |b| b.empty_phi(SsaDefSpec::tmp(MockType::U8)))
                .unwrap_or_else(abort);
            let (divided, next) = builder
                .in_block(1, |b| {
                    // One on the first visit, zero on the second.
                    let divisor = b.sub(SsaDefSpec::tmp(MockType::U8), one, index)?;
                    let divided = division.emit(b, six, divisor, zero)?;
                    let next = b.add(SsaDefSpec::tmp(MockType::U8), index, one)?;
                    b.branch_cmp(next, two, CmpKind::Lt, true, 1, 2)?;
                    Ok((divided, next))
                })
                .unwrap_or_else(abort);
            builder
                .add_phi_operand(1, index, 0, zero)
                .unwrap_or_else(abort);
            builder
                .add_phi_operand(1, index, 1, next)
                .unwrap_or_else(abort);
            builder
                .in_block(2, |b| {
                    b.call_void(crate::ir::Callee::Direct(99), vec![divided])?;
                    b.ret(None)
                })
                .unwrap_or_else(abort);
            let mut ssa = builder.finish().unwrap_or_else(abort);
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert!(facts.is_complete());
            assert!(facts.constant(divided).is_some());
            assert_eq!(facts.replacement_proof(divided), None);
            assert_eq!(facts.usage().traps, 1);
            fold_constants(&mut ssa, &facts).unwrap_or_else(abort);
            assert!(matches!(
                ssa.recorded_definition(divided).map(|def| def.op()),
                Some(SsaOp::Div { .. } | SsaOp::Rem { .. } | SsaOp::WideDiv { .. })
            ));
        }
    }

    /// A division that completes on every visit is proven, by the machine's
    /// own outcome for a zero divisor where it states one instead of a trap,
    /// and replaced by its constants: a double-width one by one per result.
    #[test]
    fn a_completed_division_can_be_replaced() {
        for (rules, divisor, quotient, remainder) in [
            (DivisionRules::CHECKED, 4, 1, 2),
            (DivisionRules::ARM, 0, 0, 6),
            (DivisionRules::RISCV, 0, 255, 6),
        ] {
            for division in [Division::Quotient, Division::Remainder, Division::Wide] {
                let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
                let divided = builder
                    .in_block(0, |b| {
                        let zero =
                            b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                        let six =
                            b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(6))?;
                        let by =
                            b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(divisor))?;
                        let divided = division.emit(b, six, by, zero)?;
                        b.ret(Some(divided))?;
                        Ok(divided)
                    })
                    .unwrap_or_else(abort);
                let mut ssa = builder.finish().unwrap_or_else(abort);
                let context = AnalysisContext::new(MACHINE.with_division_rules(rules));
                let facts = StaticExecution::new(&ssa, &context)
                    .with_host(&FrameHost)
                    .run();
                let expected = match division {
                    Division::Remainder => remainder,
                    Division::Quotient | Division::Wide => quotient,
                };
                if matches!(division, Division::Wide) && divisor == 0 {
                    // The double-width quotient keeps the x86 contract: a
                    // zero divisor traps on every machine.
                    assert_eq!(facts.replacement_proof(divided), None);
                    continue;
                }
                assert_eq!(facts.constant(divided), Some(&ConstValue::U8(expected)));
                assert_eq!(
                    facts.replacement_proof(divided),
                    Some(ReplacementProof::Completed)
                );
                let folded = fold_constants(&mut ssa, &facts).unwrap_or_else(abort);
                assert_eq!(folded, 1);
                assert!(
                    ssa.iter_instructions().all(|(_, _, instruction)| matches!(
                        instruction.op(),
                        SsaOp::Const { .. } | SsaOp::Return { .. }
                    )),
                    "{division:?}"
                );
            }
        }
    }

    /// **An operation of several results is replaced by one constant per
    /// result**, in its place and after it, where every result is known and
    /// proven: a double-width product's halves, a double-width division's
    /// quotient and remainder. A result nothing reads needs no constant: a
    /// division whose flags are not known (as a machine that leaves them
    /// undefined has them) is replaced by its quotient where nothing reads
    /// the flags, and stays whole where something does.
    #[test]
    fn a_multi_result_operation_is_replaced_by_one_constant_per_result() {
        enum Form {
            Product,
            Division,
            Flagged { read: bool },
        }
        for form in [
            Form::Product,
            Form::Division,
            Form::Flagged { read: false },
            Form::Flagged { read: true },
        ] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            let (first, second) = builder
                .in_block(0, |b| {
                    let zero = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                    let left = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(200))?;
                    let right = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(7))?;
                    let results = match form {
                        Form::Product => b.emit_two_defs(
                            SsaDefSpec::tmp(MockType::U8),
                            SsaDefSpec::tmp(MockType::U8),
                            |low, high| SsaOp::WideMul {
                                low,
                                high,
                                left,
                                right,
                                unsigned: true,
                            },
                        )?,
                        Form::Division => b.emit_two_defs(
                            SsaDefSpec::tmp(MockType::U8),
                            SsaDefSpec::tmp(MockType::U8),
                            |quotient, remainder| SsaOp::WideDiv {
                                quotient,
                                remainder,
                                high: zero,
                                low: left,
                                divisor: right,
                                unsigned: true,
                            },
                        )?,
                        Form::Flagged { .. } => b.emit_two_defs(
                            SsaDefSpec::tmp(MockType::U8),
                            SsaDefSpec::tmp(MockType::U32),
                            |dest, flags| SsaOp::Div {
                                dest,
                                left,
                                right,
                                unsigned: true,
                                flags: Some(flags),
                            },
                        )?,
                    };
                    let sum = b.add(SsaDefSpec::tmp(MockType::U8), results.0, results.0)?;
                    b.ret(Some(if matches!(form, Form::Flagged { read: true }) {
                        results.1
                    } else {
                        sum
                    }))?;
                    Ok(results)
                })
                .unwrap_or_else(abort);
            let mut ssa = builder.finish().unwrap_or_else(abort);
            let context = AnalysisContext::new(MACHINE);
            let mut facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            if matches!(form, Form::Flagged { .. }) {
                // Flags a machine leaves undefined are known on no run.
                facts.forget_values(|value| value == second);
            }
            let expected = match form {
                // 200 * 7 = 1400 = 0x0578.
                Form::Product => Some((0x78, Some(0x05))),
                // 200 / 7 = 28 remainder 4.
                Form::Division => Some((28, Some(4))),
                Form::Flagged { read: false } => Some((28, None)),
                Form::Flagged { read: true } => None,
            };
            fold_constants(&mut ssa, &facts).unwrap_or_else(abort);
            let defined = |value: SsaVarId| {
                ssa.iter_instructions()
                    .find(|(_, _, instruction)| instruction.defs().any(|def| def == value))
                    .map(|(_, _, instruction)| instruction.op().clone())
            };
            match expected {
                Some((low, None)) => {
                    assert_eq!(
                        defined(first),
                        Some(SsaOp::Const {
                            dest: first,
                            value: ConstValue::U8(low),
                        })
                    );
                    // The unread flags go with the division; the edit
                    // renumbers what follows them.
                    assert!(
                        ssa.iter_instructions().all(|(_, _, instruction)| !matches!(
                            instruction.op(),
                            SsaOp::Div { .. }
                        ))
                    );
                }
                Some((low, Some(high))) => {
                    for (value, constant) in [(first, low), (second, high)] {
                        assert_eq!(
                            defined(value),
                            Some(SsaOp::Const {
                                dest: value,
                                value: ConstValue::U8(constant),
                            })
                        );
                    }
                    // The first result keeps the operation's place and the
                    // second follows it.
                    let positions: Vec<_> = [first, second]
                        .into_iter()
                        .filter_map(|value| {
                            ssa.iter_instructions()
                                .find(|(_, _, instruction)| {
                                    instruction.defs().any(|def| def == value)
                                })
                                .map(|(block, index, _)| (block, index))
                        })
                        .collect();
                    assert_eq!(positions, vec![(0, 3), (0, 4)]);
                }
                None => assert!(matches!(defined(first), Some(SsaOp::Div { .. }))),
            }
            assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        }
    }

    /// States no frame: every stack access is outside storage the run knows
    /// is live.
    struct Unattested;

    impl<T: Target> ExecutionHost<T> for Unattested {}

    /// A frame load is replaced only where the frame it reads is one the run
    /// knows is live: inside the host's attested extent, not outside it.
    #[test]
    fn frame_loads_are_replaced_only_inside_a_live_frame() {
        for attested in [false, true] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            let stack = {
                let ssa = builder.function_mut();
                let stack = ssa.create_variable(
                    VariableOrigin::EntryLiveIn,
                    0,
                    DefSite::entry(),
                    MockType::Ptr,
                );
                if let Some(variable) = ssa.variable_mut(stack) {
                    variable.set_role(VariableRole::EntryStackPointer);
                }
                stack
            };
            let loaded = builder
                .in_block(0, |b| {
                    let at = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                        dest,
                        base: stack,
                        index: None,
                        stride: 1,
                        offset: -8,
                        result_type: MockType::Ptr,
                    })?;
                    let byte =
                        b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x41))?;
                    b.store_indirect(at, byte, MockType::U8)?;
                    let loaded =
                        b.load_indirect(SsaDefSpec::tmp(MockType::U8), at, MockType::U8)?;
                    b.ret(Some(loaded))?;
                    Ok(loaded)
                })
                .unwrap_or_else(abort);
            let mut ssa = builder.finish().unwrap_or_else(abort);
            let context = AnalysisContext::new(MACHINE);
            let host: &dyn ExecutionHost<MockTarget> =
                if attested { &FrameHost } else { &Unattested };
            let facts = StaticExecution::new(&ssa, &context).with_host(host).run();
            assert!(facts.is_complete(), "{:?}", facts.completion());
            assert_eq!(facts.constant(loaded), Some(&ConstValue::U8(0x41)));
            assert_eq!(
                facts.replacement_proof(loaded),
                attested.then_some(ReplacementProof::Accessible)
            );
            assert_eq!(
                fold_constants(&mut ssa, &facts).unwrap_or_else(abort),
                usize::from(attested)
            );
        }
    }

    /// A broadcast and a relaxed atomic load of read-only image bytes are
    /// replaced and keep the read; an acquire load orders what follows and a
    /// reserving load opens a reservation, so both stay.
    #[test]
    fn broadcast_and_relaxed_atomic_image_loads_are_replaced_with_their_reads() {
        for form in 0..4 {
            let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
            let loaded = builder
                .in_block(0, |b| {
                    let at =
                        b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                    let loaded = if form == 0 {
                        b.emit_def(SsaDefSpec::tmp(MockType::V4I16), |dest| {
                            SsaOp::VectorBroadcastLoad {
                                dest,
                                addr: at,
                                vector_type: MockType::V4I16,
                                access: AccessQualifiers::default(),
                            }
                        })?
                    } else {
                        b.emit_def(SsaDefSpec::tmp(MockType::U16), |dest| SsaOp::AtomicLoad {
                            dest,
                            addr: at,
                            value_type: MockType::U16,
                            ordering: if form == 2 {
                                AtomicOrdering::Acquire
                            } else {
                                AtomicOrdering::Relaxed
                            },
                            width: AtomicAccessWidth::Bits16,
                            reserves: form == 3,
                            access: AccessQualifiers::default(),
                        })?
                    };
                    b.ret(Some(loaded))?;
                    Ok(loaded)
                })
                .unwrap_or_else(abort);
            let mut ssa = builder.finish().unwrap_or_else(abort);
            let machine = Machine {
                byte_order: Some(Endianness::Little),
                ..MACHINE
            };
            let context = AnalysisContext::new(machine).with_image(&Image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            let replaced = form < 2;
            if form < 3 {
                assert!(facts.is_complete(), "{form}: {:?}", facts.completion());
                assert!(facts.constant(loaded).is_some(), "{form}");
            }
            assert_eq!(
                facts.replacement_proof(loaded),
                replaced.then_some(ReplacementProof::ReadOnlyImage {
                    address: 0x1000,
                    len: 2,
                }),
                "{form}"
            );
            assert_eq!(
                fold_constants(&mut ssa, &facts).unwrap_or_else(abort),
                usize::from(replaced),
                "{form}"
            );
            let reads: &[(u64, u64)] = if replaced { &[(0x1000, 2)] } else { &[] };
            assert_eq!(ssa.facts().image_reads, reads, "{form}");
        }
    }

    /// Residual image loads keep whether their value is entered before the defining load is replaced.
    #[test]
    fn folded_image_reads_retain_control_entry_provenance() {
        for entered in [false, true] {
            let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
            let loaded = builder
                .in_block(0, |b| {
                    let at =
                        b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                    let loaded =
                        b.load_indirect(SsaDefSpec::tmp(MockType::U16), at, MockType::U16)?;
                    if entered {
                        b.call_void(
                            crate::ir::Callee::Indirect {
                                fptr: loaded,
                                signature: 0,
                            },
                            vec![],
                        )?;
                    }
                    b.ret(Some(loaded))?;
                    Ok(loaded)
                })
                .unwrap_or_else(abort);
            let mut ssa = builder.finish().unwrap_or_else(abort);
            let context = AnalysisContext::new(MACHINE).with_image(&Image);
            let facts = StaticExecution::new(&ssa, &context)
                .with_host(&FrameHost)
                .run();
            assert_eq!(facts.constant(loaded), Some(&ConstValue::U16(0xf00f)));
            assert_eq!(fold_constants(&mut ssa, &facts).unwrap_or_else(abort), 1);
            assert_eq!(ssa.facts().image_reads, [(0x1000, 2)]);
            assert_eq!(
                ssa.facts().entered_reads,
                if entered { vec![0x1000] } else { vec![] }
            );
        }
    }

    /// A folded image read retains host provenance; an unproven access keeps its load.
    #[test]
    fn image_load_replacements_need_access_proofs_and_keep_provenance() {
        for permission in [
            crate::ir::ops::AccessPermission::Current,
            crate::ir::ops::AccessPermission::Unprivileged,
        ] {
            for prefix in [false, true] {
                let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
                let loaded = builder
                    .in_block(0, |b| {
                        let at =
                            b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                        let loaded = b.emit_def(SsaDefSpec::tmp(MockType::U16), |dest| {
                            SsaOp::LoadIndirect {
                                dest,
                                addr: at,
                                value_type: MockType::U16,
                                access: AccessQualifiers {
                                    permission,
                                    volatile: prefix,
                                    ..AccessQualifiers::default()
                                },
                            }
                        })?;
                        b.ret(Some(loaded))?;
                        Ok(loaded)
                    })
                    .unwrap_or_else(abort);
                let mut ssa = builder.finish().unwrap_or_else(abort);
                let context = AnalysisContext::new(MACHINE).with_image(&Image);
                let facts = StaticExecution::new(&ssa, &context)
                    .with_host(&FrameHost)
                    .run();
                assert_eq!(facts.constant(loaded), Some(&ConstValue::U16(0xf00f)));
                let allowed = !prefix && permission == crate::ir::ops::AccessPermission::Current;
                assert_eq!(
                    fold_constants(&mut ssa, &facts).unwrap_or_else(abort),
                    usize::from(allowed)
                );
                assert_eq!(
                    ssa.facts().image_reads,
                    if allowed { vec![(0x1000, 2)] } else { vec![] }
                );
            }
        }
    }
    /// A counted loop plus a known or runtime-dependent suffix, with optional orphan entry ids.
    fn identity_body<T: Target<Type = MockType>>(
        dynamic: bool,
        orphan: bool,
        return_loop_value: bool,
    ) -> Result<(SsaFunction<T>, SsaVarId)> {
        let mut builder = SsaFunctionBuilder::<T>::new(if dynamic || orphan { 2 } else { 0 }, 0);
        if orphan {
            builder.function_mut().create_variable(
                VariableOrigin::Argument(1),
                0,
                DefSite::entry(),
                MockType::I32,
            );
        }
        let incoming = if dynamic {
            Some(builder.in_block(0, |b| b.load_arg(SsaDefSpec::argument(0, MockType::I32), 0))?)
        } else {
            None
        };
        let (zero, one, count, source) = builder.in_block(0, |b| {
            let zero = b.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
            let one = b.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(1))?;
            let count = b.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(2))?;
            let source = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
            b.jump(1)?;
            Ok((zero, one, count, source))
        })?;
        let index = builder.in_block(1, |b| b.empty_phi(SsaDefSpec::tmp(MockType::I32)))?;
        let next = builder.in_block(1, |b| {
            b.load_indirect(SsaDefSpec::tmp(MockType::U8), source, MockType::U8)?;
            let next = b.add(SsaDefSpec::tmp(MockType::I32), index, one)?;
            b.branch_cmp(next, count, CmpKind::Lt, false, 1, 2)?;
            Ok(next)
        })?;
        builder.add_phi_operand(1, index, 0, zero)?;
        builder.add_phi_operand(1, index, 1, next)?;
        let sum = builder.in_block(2, |b| {
            let left = b.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(20))?;
            let right = b.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(22))?;
            let sum = b.add(
                SsaDefSpec::tmp(MockType::I32),
                incoming.unwrap_or(left),
                right,
            )?;
            b.ret(Some(if return_loop_value { next } else { sum }))?;
            Ok(sum)
        })?;
        Ok((builder.finish()?, sum))
    }

    /// Original ids remain publication evidence while both transformations commit once.
    #[test]
    fn combined_residuals_preserve_constant_and_dynamic_definition_identities() -> Result<()> {
        for orphan in [false, true] {
            for dynamic in [false, true] {
                let (mut ssa, sum) = identity_body::<MockTarget>(dynamic, orphan, false)?;
                let context = AnalysisContext::new(MACHINE).with_image(&Image);
                let facts = StaticExecution::new(&ssa, &context).run();
                assert!(facts.is_complete());
                assert_eq!(facts.decoded_loops(), [1]);
                let original = facts
                    .constants()
                    .into_iter()
                    .map(|(id, value)| (id, value.clone()))
                    .collect::<Vec<_>>();
                assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
                assert!(apply_static_facts(&mut ssa, &facts)? > 0);
                assert!(LoopAnalyzer::new(&ssa).analyze().is_empty());
                assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
                assert_eq!(
                    facts
                        .constants()
                        .into_iter()
                        .map(|(id, value)| (id, value.clone()))
                        .collect::<Vec<_>>(),
                    original
                );
                let after = StaticExecution::new(&ssa, &context).run();
                assert!(after.is_complete());
                assert_eq!(
                    after
                        .returned()
                        .and_then(|value| value.constant())
                        .and_then(ConstValue::as_i64),
                    if dynamic { None } else { Some(42) }
                );
                assert_eq!(
                    facts
                        .constants()
                        .into_iter()
                        .find(|(id, _)| *id == sum)
                        .map(|(_, value)| value.as_i64()),
                    if dynamic { None } else { Some(Some(42)) }
                );
                assert_eq!(
                    ssa.blocks()
                        .iter()
                        .flat_map(|block| block.instructions())
                        .any(|instruction| matches!(instruction.op(), SsaOp::Add { .. })),
                    dynamic
                );
            }
        }
        Ok(())
    }

    /// Full verification rejects a completed CFG edit before any read witnesses escape.
    #[test]
    fn residual_verifier_rejection_restores_ir_and_read_metadata() -> Result<()> {
        let (mut ssa, _) = identity_body::<FactfulTarget>(false, false, true)?;
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        let forest = LoopAnalyzer::new(&ssa).analyze();
        let cfg = SsaCfg::from_ssa(&ssa);
        let mut plan = Planner {
            ssa: &ssa,
            facts: &facts,
            control: None,
        }
        .one_loop(&cfg, &compute_dominators(&cfg, NodeId::new(0)), &forest, 1)
        .ok_or_else(|| Error::new("loop plan"))?;
        drop(cfg);
        assert!(!plan.live_outs.is_empty());
        plan.live_outs.clear();
        let entered = OnceCell::new();
        let reads = facts
            .loop_reads(1)
            .map(|(key, value)| {
                residual_read(&ssa, key, value, &entered).ok_or_else(|| Error::new("read witness"))
            })
            .collect::<Result<Vec<_>>>()?;
        assert!(!reads.is_empty());
        let mut unchecked = ssa.clone();
        unchecked.edit(
            SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| apply_residual(editor, &plan, &BTreeMap::new(), None, &mut Vec::new()),
        )?;
        assert!(
            !SsaVerifier::new(&unchecked)
                .verify(VerifyLevel::Full)
                .is_empty(),
            "edit succeeds; verifier must reject its missing live-out"
        );
        let before = format!("{ssa:?}");
        assert!(
            commit_static_facts(
                &mut ssa,
                Vec::new(),
                vec![plan],
                reads,
                None,
                SsaRollbackPolicy::OnFailure,
            )
            .is_err()
        );
        assert_eq!(format!("{ssa:?}"), before);
        assert!(ssa.facts().image_reads.is_empty());
        assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        Ok(())
    }

    /// Two disjoint read loops both have live-outs and retain independent image witnesses.
    fn bounded_disjoint_reads() -> Result<SsaFunction<FactfulTarget>> {
        let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
        let (first, second, zero) = builder.in_block(0, |block| {
            let first =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
            let second =
                block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1001))?;
            let zero = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
            block.jump(1)?;
            Ok((first, second, zero))
        })?;
        let left = builder.in_block(1, |block| {
            let value = block.load_indirect(SsaDefSpec::tmp(MockType::U8), first, MockType::U8)?;
            block.branch_cmp(zero, zero, CmpKind::Lt, true, 1, 2)?;
            Ok(value)
        })?;
        let left_phi = builder.in_block(2, |block| {
            let joined = block.empty_phi(SsaDefSpec::tmp(MockType::U8))?;
            block.jump(3)?;
            Ok(joined)
        })?;
        builder.add_phi_operand(2, left_phi, 1, left)?;
        let right = builder.in_block(3, |block| {
            let value = block.load_indirect(SsaDefSpec::tmp(MockType::U8), second, MockType::U8)?;
            block.branch_cmp(zero, zero, CmpKind::Lt, true, 3, 4)?;
            Ok(value)
        })?;
        let right_phi = builder.in_block(4, |block| {
            let joined = block.empty_phi(SsaDefSpec::tmp(MockType::U8))?;
            let sum = block.add(SsaDefSpec::tmp(MockType::U8), left_phi, joined)?;
            block.ret(Some(sum))?;
            Ok(joined)
        })?;
        builder.add_phi_operand(4, right_phi, 3, right)?;
        let function = builder.finish()?;
        assert!(
            SsaVerifier::new(&function)
                .verify(VerifyLevel::Full)
                .is_empty()
        );
        Ok(function)
    }

    /// A whole-request limit cannot accept one disjoint plan and silently omit another.
    #[test]
    fn bounded_disjoint_plans_share_limits_and_publish_metadata_atomically() -> Result<()> {
        let ssa = bounded_disjoint_reads()?;
        let original = format!("{ssa:?}");
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert_eq!(facts.decoded_loops(), vec![1, 3]);
        let initial = ResidualSize::measure(&ssa, &WorkBudget::DEFAULT, None)
            .map_err(|reason| Error::new(format!("{reason:?}")))?;
        let limits = WorkBudget::DEFAULT.with_limit(
            Budget::ResidualOperations,
            initial
                .operations
                .checked_add(1)
                .ok_or_else(|| Error::new("fixture count"))?,
        );
        assert!(matches!(
            rewrite_static_facts(&ssa, &facts, &limits, &|| false)?,
            ResidualRewrite::Refused(ResidualRefusal::Exhausted(Exhaustion {
                budget: Budget::ResidualOperations,
                ..
            }))
        ));
        assert_eq!(format!("{ssa:?}"), original);
        assert!(ssa.facts().image_reads.is_empty());
        let limits = limits.with_limit(
            Budget::ResidualOperations,
            initial
                .operations
                .checked_add(2)
                .ok_or_else(|| Error::new("fixture count"))?,
        );
        let phi_limit = limits.with_limit(
            Budget::ResidualPhiOperands,
            initial
                .phi_operands
                .checked_add(1)
                .ok_or_else(|| Error::new("phi count"))?,
        );
        assert!(matches!(
            rewrite_static_facts(&ssa, &facts, &phi_limit, &|| false)?,
            ResidualRewrite::Refused(ResidualRefusal::Exhausted(Exhaustion {
                budget: Budget::ResidualPhiOperands,
                ..
            }))
        ));
        let limits = limits.with_limit(
            Budget::ResidualPhiOperands,
            initial
                .phi_operands
                .checked_add(2)
                .ok_or_else(|| Error::new("phi count"))?,
        );
        let polls = Cell::new(0usize);
        let count = || {
            polls.set(polls.get().saturating_add(1));
            false
        };
        let ResidualRewrite::Rewritten {
            function,
            replacements,
            ..
        } = rewrite_static_facts(&ssa, &facts, &limits, &count)?
        else {
            return Err(Error::new("complete bounded edit"));
        };
        assert_eq!(
            replacements, 3,
            "two loops and the scalar sum share one edit"
        );
        assert_eq!(function.facts().image_reads, vec![(0x1000, 1), (0x1001, 1)]);
        let after = StaticExecution::new(&function, &context).run();
        assert_eq!(
            after.returned().and_then(|value| value.constant()),
            Some(&ConstValue::U8(255))
        );
        let total = polls.get();
        polls.set(0);
        let cancelled = || {
            polls.set(polls.get().saturating_add(1));
            polls.get() == total
        };
        assert!(matches!(
            rewrite_static_facts(&ssa, &facts, &limits, &cancelled)?,
            ResidualRewrite::Refused(ResidualRefusal::Cancelled)
        ));
        assert_eq!(format!("{ssa:?}"), original);
        assert!(ssa.facts().image_reads.is_empty());
        assert!(facts.is_complete());
        Ok(())
    }

    /// **A constant standing for a loop's result is of that result's type.**
    /// Each loop of the fixture leaves with the byte it loaded, a `uint8_t`
    /// the host's constant query has no answer for; the constant the residual
    /// replaces it by is a `uint8_t` too, not a value of no known type.
    #[test]
    fn a_live_out_constant_has_the_type_of_the_value_it_replaces() -> Result<()> {
        let ssa = bounded_disjoint_reads()?;
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        let ResidualRewrite::Rewritten { function, .. } =
            rewrite_static_facts(&ssa, &facts, &WorkBudget::DEFAULT, &|| false)?
        else {
            return Err(Error::new("complete edit"));
        };
        // The fixture's own zero, each loop's byte and their folded sum.
        let types: Vec<_> = function
            .blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .filter_map(|instruction| match instruction.op() {
                SsaOp::Const {
                    dest,
                    value: ConstValue::U8(_),
                } => function.variable(*dest).map(|var| *var.var_type()),
                _ => None,
            })
            .collect();
        assert!(
            types.len() >= 3,
            "the residual states the loops' bytes: {types:?}"
        );
        assert!(types.iter().all(|ty| *ty == MockType::U8), "{types:?}");
        Ok(())
    }

    /// Checked peak arithmetic refuses overflow without allocating an expansion.
    #[test]
    fn bounded_peak_count_overflow_is_explicit() -> Result<()> {
        let ssa = bounded_disjoint_reads()?;
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context).run();
        let planner = Planner {
            ssa: &ssa,
            facts: &facts,
            control: None,
        };
        let plans = planner.loops();
        assert_eq!(plans.plans.len(), 2);
        let initial = ResidualSize {
            operations: usize::MAX,
            ..ResidualSize::default()
        };
        assert_eq!(
            planner.peak(initial, &plans.plans),
            Err(ResidualRefusal::SizeOverflow)
        );
        Ok(())
    }

    /// Census cancellation covers block and phi visits before a working copy exists.
    #[test]
    fn bounded_census_cancellation_and_phi_limit_stop_traversal() -> Result<()> {
        let ssa = bounded_disjoint_reads()?;
        let original = format!("{ssa:?}");
        let polls = Cell::new(0usize);
        let count = || {
            polls.set(polls.get().saturating_add(1));
            false
        };
        let control = ResidualControl {
            cancelled: &count,
            refused: Cell::new(false),
        };
        let limits = WorkBudget::DEFAULT;
        let size = ResidualSize::measure(&ssa, &limits, Some(&|| !control.active()))
            .map_err(|reason| Error::new(format!("{reason:?}")))?;
        let total = polls.get();
        assert_eq!(total, 8, "one boundary, five blocks and two exit phis");
        assert_eq!(size.phi_operands, 2);
        for stop in 1..=total {
            polls.set(0);
            let cancelled = || {
                polls.set(polls.get().saturating_add(1));
                polls.get() == stop
            };
            let control = ResidualControl {
                cancelled: &cancelled,
                refused: Cell::new(false),
            };
            assert_eq!(
                ResidualSize::measure(&ssa, &limits, Some(&|| !control.active())),
                Err(ResidualRefusal::Cancelled)
            );
            assert_eq!(polls.get(), stop);
            assert!(!control.active(), "one-shot cancellation remains sticky");
            assert_eq!(polls.get(), stop);
        }
        polls.set(0);
        let control = ResidualControl {
            cancelled: &count,
            refused: Cell::new(false),
        };
        assert!(matches!(
            ResidualSize::measure(
                &ssa,
                &limits.with_limit(Budget::ResidualPhiOperands, 0),
                Some(&|| !control.active())
            ),
            Err(ResidualRefusal::Exhausted(Exhaustion {
                budget: Budget::ResidualPhiOperands,
                ..
            }))
        ));
        assert_eq!(
            polls.get(),
            5,
            "stop after the first exit phi, before block 3"
        );
        assert_eq!(format!("{ssa:?}"), original);
        assert!(ssa.facts().image_reads.is_empty());
        Ok(())
    }

    /// Peak planning polls the admitted loop and its borrowed byte runs before expansion.
    #[test]
    fn bounded_peak_cancellation_visits_plans_and_byte_runs() -> Result<()> {
        let (ssa, _) = decoder_for::<FactfulTarget>();
        let original = format!("{ssa:?}");
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let facts = StaticExecution::new(&ssa, &context)
            .with_host(&FrameHost)
            .run();
        assert!(facts.is_complete());
        let planner = Planner {
            ssa: &ssa,
            facts: &facts,
            control: None,
        };
        let plans = planner.loops();
        assert_eq!(plans.plans.len(), 1);
        let runs = plans
            .plans
            .iter()
            .filter_map(|plan| plan.writes.as_ref())
            .map(|writes| writes.runs.len())
            .sum::<usize>();
        assert!(runs > 0);
        let initial = ResidualSize::measure(&ssa, &WorkBudget::DEFAULT, None)
            .map_err(|reason| Error::new(format!("{reason:?}")))?;
        let polls = Cell::new(0usize);
        let count = || {
            polls.set(polls.get().saturating_add(1));
            false
        };
        let control = ResidualControl {
            cancelled: &count,
            refused: Cell::new(false),
        };
        assert!(
            Planner {
                ssa: &ssa,
                facts: &facts,
                control: Some(&control),
            }
            .peak(initial, &plans.plans)
            .is_ok()
        );
        let total = polls.get();
        assert_eq!(total, runs.saturating_add(1));
        for stop in 1..=total {
            polls.set(0);
            let cancelled = || {
                polls.set(polls.get().saturating_add(1));
                polls.get() == stop
            };
            let control = ResidualControl {
                cancelled: &cancelled,
                refused: Cell::new(false),
            };
            assert_eq!(
                Planner {
                    ssa: &ssa,
                    facts: &facts,
                    control: Some(&control),
                }
                .peak(initial, &plans.plans),
                Err(ResidualRefusal::Cancelled)
            );
            assert_eq!(polls.get(), stop);
        }
        assert_eq!(format!("{ssa:?}"), original);
        assert!(ssa.facts().image_reads.is_empty());
        assert!(facts.is_complete());
        Ok(())
    }
}
