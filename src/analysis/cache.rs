//! [`FunctionAnalyses`] - the derived analyses of one SSA function, each
//! computed at most once.
//!
//! # Why this exists
//!
//! Several analyses in [`crate::analysis`] are pure functions of a function's
//! SSA: run them twice on the same IR and they return the same thing. The
//! pipeline does exactly that - a driver that re-recovers a function's types
//! because its *seeds* moved re-derives the points-to relation and the alias
//! keys as well, even though the IR it derived them from cannot have changed.
//!
//! # Validity
//!
//! A derived analysis is valid exactly as long as the IR it was derived from is
//! unchanged, and nothing about a `(key, analysis)` pair records that. So the
//! memo **borrows** the function it describes and hands that same borrow back
//! through [`FunctionAnalyses::ir`]: while a handle is alive no `&mut
//! SsaFunction` can exist, so a cached analysis cannot outlive the IR it
//! describes and a consumer cannot pair one function's relation with another's
//! IR. Misuse does not compile rather than being caught by review.
//!
//! The consequence is that a handle cannot span a mutation. Code that rewrites
//! the IR between two analyses builds a second handle afterwards, and that is
//! the honest outcome: those two analyses genuinely describe different
//! programs.
//!
//! # Laziness
//!
//! Nothing is computed until asked for. This is load-bearing, not an
//! optimization: callers exist that return early without ever needing an
//! analysis, and an eager handle would do work the current code skips.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use crate::{
    analysis::{
        address::{AddressWalk, AliasKey},
        budget::Exhaustion,
        cfg::SsaCfg,
        consts::Machine,
        context::AnalysisContext,
        dataflow::SparseDenseSchedule,
        defuse::DefUseIndex,
        memory::MemorySsa,
        pointsto::{CallEscape, PointsTo, analyze_function},
    },
    ir::function::{AddressTakenSlots, InstrLocations, SsaFunction},
    passes::ranges::{self, ValueRanges},
    target::Target,
};

/// The derived analyses of one SSA function, computed on demand and reused.
///
/// Each analysis is a separate slot, so a consumer that needs only one pays
/// only for that one. Slots are [`OnceLock`]s rather than a lock around a map:
/// accessors take `&self`, the type stays `Sync`, and once a slot is filled
/// reads are uncontended.
///
/// # Examples
///
/// ```
/// use analyssa::{
///     analysis::{
///         AnalysisContext,
///         cache::{Analysis, FunctionAnalyses},
///     },
///     testing,
/// };
///
/// let ir = testing::const_i32_return(7);
/// let analyses = FunctionAnalyses::new(&ir, &AnalysisContext::new(testing::MACHINE));
/// assert!(!analyses.is_computed(Analysis::PointsTo), "nothing is solved up front");
///
/// let relation = analyses.points_to(); // solves here
/// let again = analyses.points_to(); // same relation, no second solve
/// assert!(std::ptr::eq(relation, again));
/// assert!(analyses.is_computed(Analysis::PointsTo));
/// assert!(!analyses.is_computed(Analysis::DefUse), "other slots stay empty");
/// ```
#[derive(Debug)]
pub struct FunctionAnalyses<'ir, T: Target> {
    /// The function every slot below describes.
    ir: &'ir SsaFunction<T>,
    /// What the analyses run under: the machine, the work budget, what the
    /// called functions do, and the interrupt a long analysis asks.
    context: AnalysisContext<'ir>,
    /// Inclusion-based points-to relation ([`analyze_function`]).
    points_to: OnceLock<Arc<PointsTo>>,
    /// Which cells a call can write, answered from the points-to relation
    /// ([`CallEscape::of`]).
    call_escape: OnceLock<CallEscape>,
    /// Cell-identity projection of the function's addresses
    /// ([`AddressWalk::alias_keys`](crate::analysis::address::AddressWalk::alias_keys)).
    alias_keys: OnceLock<BTreeMap<u32, AliasKey>>,
    /// Definitions and uses of every value ([`DefUseIndex::build`]).
    def_use: OnceLock<DefUseIndex<T>>,
    /// Proved value ranges, or the budget the fixpoint ran out of
    /// ([`ranges::analyze`]).
    ranges: OnceLock<Result<ValueRanges, Exhaustion>>,
    /// Where each instruction identity currently sits
    /// ([`SsaFunction::instruction_locations`]).
    instr_locations: OnceLock<InstrLocations>,
    /// Which argument and local slots escape
    /// ([`SsaFunction::address_taken_slots`]).
    address_taken: OnceLock<AddressTakenSlots>,
    /// What a sparse/dense solve schedules from the function alone
    /// ([`SparseDenseSchedule::of`]).
    sparse_dense: OnceLock<SparseDenseSchedule>,
    /// The function's memory, read through the points-to relation above
    /// ([`MemorySsa::build_with_cfg`]).
    memory: OnceLock<MemorySsa<T>>,
}

impl<'ir, T: Target> FunctionAnalyses<'ir, T> {
    /// Builds an empty set of analyses over `ir`, run under `context`: on its
    /// machine, within its budget, with its called functions' models, and
    /// asking its interrupt.
    ///
    /// Nothing is computed here; each accessor computes its own slot on first
    /// use.
    #[must_use]
    pub const fn new(ir: &'ir SsaFunction<T>, context: &AnalysisContext<'ir>) -> Self {
        Self {
            ir,
            context: *context,
            points_to: OnceLock::new(),
            call_escape: OnceLock::new(),
            alias_keys: OnceLock::new(),
            def_use: OnceLock::new(),
            ranges: OnceLock::new(),
            instr_locations: OnceLock::new(),
            address_taken: OnceLock::new(),
            sparse_dense: OnceLock::new(),
            memory: OnceLock::new(),
        }
    }

    /// Returns the function these analyses describe.
    ///
    /// Consumers should read the IR through this rather than carrying their own
    /// reference alongside the handle - that is what makes a mismatched pair
    /// unrepresentable.
    #[must_use]
    pub const fn ir(&self) -> &'ir SsaFunction<T> {
        self.ir
    }

    /// Returns what the analyses run under ([`AnalysisContext`]): an analysis
    /// a consumer runs over [`Self::ir`] runs under the same.
    #[must_use]
    pub const fn context(&self) -> &AnalysisContext<'ir> {
        &self.context
    }

    /// Returns the machine the analyses run on.
    #[must_use]
    pub fn machine(&self) -> Machine {
        self.context.machine()
    }

    /// Returns the function's points-to relation, solving on first use.
    ///
    /// See [`analyze_function`] for what the relation models.
    #[must_use]
    pub fn points_to(&self) -> &PointsTo {
        self.shared_points_to()
    }

    /// The points-to relation as the slot shares it, solving on first use.
    fn shared_points_to(&self) -> &Arc<PointsTo> {
        self.points_to
            .get_or_init(|| analyze_function(self.ir, self.context.budget(), self.context.models()))
    }

    /// Returns which cells the function's calls can write, deriving the oracle
    /// from the points-to relation on first use.
    ///
    /// See [`CallEscape::of`]. A function whose types are recovered many times
    /// asks the same oracle in every solve.
    #[must_use]
    pub fn call_escape(&self) -> &CallEscape {
        self.call_escape
            .get_or_init(|| CallEscape::of(Arc::clone(self.shared_points_to()), self.ir))
    }

    /// Returns the function's alias keys, deriving them on first use.
    ///
    /// See [`AddressWalk::alias_keys`](crate::analysis::address::AddressWalk::alias_keys) for the cell-identity projection.
    #[must_use]
    pub fn alias_keys(&self) -> &BTreeMap<u32, AliasKey> {
        self.alias_keys
            .get_or_init(|| AddressWalk::new(self.ir, self.machine().ptr_size).alias_keys())
    }

    /// Returns the function's definitions and uses, indexing on first use.
    ///
    /// See [`DefUseIndex`]; built without the defining operations.
    #[must_use]
    pub fn def_use(&self) -> &DefUseIndex<T> {
        self.def_use.get_or_init(|| DefUseIndex::build(self.ir))
    }

    /// Returns the function's proved value ranges, solving on first use.
    ///
    /// The passes of a pipeline read theirs from their environment's
    /// [`RangeFacts`](crate::passes::ranges::RangeFacts), which outlives a
    /// rewrite; this is the same fixpoint for a reader holding the function
    /// unchanged.
    ///
    /// # Errors
    ///
    /// The [`Exhaustion`] [`ranges::analyze`] reports when the fixpoint did not
    /// converge within the budget, which means "no facts", not "no
    /// constraints".
    pub fn ranges(&self) -> Result<&ValueRanges, Exhaustion> {
        self.ranges
            .get_or_init(|| ranges::analyze(self.ir, self.context.budget(), &BTreeMap::new()))
            .as_ref()
            .map_err(|exhaustion| *exhaustion)
    }

    /// Returns where every identified instruction currently sits, indexing on
    /// first use.
    ///
    /// One walk answers a whole host table's worth of lookups; see
    /// [`InstrLocations`].
    #[must_use]
    pub fn instr_locations(&self) -> &InstrLocations {
        self.instr_locations
            .get_or_init(|| self.ir.instruction_locations())
    }

    /// Returns which argument and local slots escape, scanning on first use.
    #[must_use]
    pub fn address_taken(&self) -> &AddressTakenSlots {
        self.address_taken
            .get_or_init(|| self.ir.address_taken_slots())
    }

    /// Returns the function's sparse/dense solve schedule, deriving it on first
    /// use.
    ///
    /// See [`SparseDenseSchedule`]; what [`solve_sparse_dense`] takes so a
    /// function solved again under new seeds is not scheduled again.
    ///
    /// [`solve_sparse_dense`]: crate::analysis::dataflow::solve_sparse_dense
    #[must_use]
    pub fn sparse_dense_schedule(&self) -> &SparseDenseSchedule {
        self.sparse_dense
            .get_or_init(|| SparseDenseSchedule::of(self.ir))
    }

    /// Returns the function's memory SSA, building it on first use.
    ///
    /// Built over the points-to relation these analyses hold
    /// ([`Self::points_to`]), solved here only where a footprint can set two
    /// locations apart, so every reader of the function's memory and of its
    /// relation shares one of each.
    #[must_use]
    pub fn memory_ssa(&self) -> &MemorySsa<T> {
        self.memory.get_or_init(|| {
            MemorySsa::build_with_cfg(&SsaCfg::from_ssa(self.ir), &self.context, None, || {
                self.points_to()
            })
        })
    }

    /// Returns whether an analysis has been computed, for tests and for
    /// cache accounting.
    #[must_use]
    pub fn is_computed(&self, analysis: Analysis) -> bool {
        match analysis {
            Analysis::PointsTo => self.points_to.get().is_some(),
            Analysis::CallEscape => self.call_escape.get().is_some(),
            Analysis::AliasKeys => self.alias_keys.get().is_some(),
            Analysis::DefUse => self.def_use.get().is_some(),
            Analysis::Ranges => self.ranges.get().is_some(),
            Analysis::InstrLocations => self.instr_locations.get().is_some(),
            Analysis::AddressTaken => self.address_taken.get().is_some(),
            Analysis::SparseDenseSchedule => self.sparse_dense.get().is_some(),
            Analysis::MemorySsa => self.memory.get().is_some(),
        }
    }
}

/// Names one of the analyses [`FunctionAnalyses`] holds.
///
/// The "analysis id" half of the `(function, analysis)` key: the cache key
/// selects the function, this selects which of its analyses is meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Analysis {
    /// The inclusion-based points-to relation.
    PointsTo,
    /// Which cells a call can write.
    CallEscape,
    /// The cell-identity alias keys.
    AliasKeys,
    /// The definitions and uses of every value.
    DefUse,
    /// The proved value ranges.
    Ranges,
    /// Where each instruction identity sits.
    InstrLocations,
    /// Which argument and local slots escape.
    AddressTaken,
    /// What a sparse/dense solve schedules from the function alone.
    SparseDenseSchedule,
    /// The function's memory SSA.
    MemorySsa,
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::{
        analysis::{
            address::AddressBase,
            budget::{Budget, WorkBudget},
        },
        ir::{ops::SsaOp, variable::SsaVarId},
        testing::{MACHINE, MockTarget, const_i32_return, memory_effect_fixture},
    };

    /// **The cached points-to runs within the host's budget.** A context
    /// allowing no points-to step leaves a relation saturated and exhausted,
    /// where the default budget solves the same function.
    #[test]
    fn the_points_to_relation_runs_within_the_contexts_budget() {
        let ir = memory_effect_fixture();
        let starved = WorkBudget::DEFAULT.with_limit(Budget::PointsToStepsPerConstraint, 0);
        let context = AnalysisContext::new(MACHINE).with_budget(starved);
        let exhaustion = FunctionAnalyses::new(&ir, &context)
            .points_to()
            .exhaustion()
            .map(|exhaustion| exhaustion.budget);
        assert_eq!(exhaustion, Some(Budget::PointsToStepsPerConstraint));

        let unbounded = AnalysisContext::new(MACHINE);
        assert!(
            FunctionAnalyses::new(&ir, &unbounded)
                .points_to()
                .exhaustion()
                .is_none()
        );
    }

    /// Pins that a slot solves on first access and hands back the same
    /// relation (same allocation) on every later call, never re-solving.
    #[test]
    fn an_analysis_is_computed_once_and_reused() {
        let function = memory_effect_fixture();
        let analyses =
            FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));

        assert!(!analyses.is_computed(Analysis::PointsTo));
        let first = analyses.points_to();
        assert!(analyses.is_computed(Analysis::PointsTo));
        let second = analyses.points_to();

        // Same allocation, not merely an equal value: the second call did not
        // re-solve.
        assert!(ptr::eq(first, second));
    }

    /// **A function's call oracle is derived once, from the relation the slot
    /// keeps.** Asking it solves the points-to relation if nothing has, every
    /// later request is the same oracle, and it answers as one built afresh
    /// from that relation does.
    #[test]
    fn the_call_oracle_is_derived_once_from_the_kept_relation() {
        let function = memory_effect_fixture();
        let analyses =
            FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));

        assert!(!analyses.is_computed(Analysis::CallEscape));
        let first = analyses.call_escape();
        assert!(analyses.is_computed(Analysis::CallEscape));
        assert!(analyses.is_computed(Analysis::PointsTo));
        assert!(ptr::eq(first, analyses.call_escape()));

        let fresh = CallEscape::of(Arc::new(analyses.points_to().clone()), &function);
        for value in 0..function.variable_count() {
            let cell = AliasKey {
                base: AddressBase::Value(SsaVarId::from_index(value)),
                index: None,
                stride_bytes: 0,
                offset_bits: 0,
            };
            assert_eq!(first.clobbers(&cell), fresh.clobbers(&cell), "{cell:?}");
        }
    }

    /// **The instruction locations are indexed once, where the function
    /// places them.** The first request indexes every identified
    /// instruction, later ones reuse that index, and it answers as the
    /// function's own walk does.
    #[test]
    fn the_instruction_locations_are_indexed_once() {
        let function = memory_effect_fixture();
        let analyses =
            FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));
        assert!(!analyses.is_computed(Analysis::InstrLocations));
        let first = analyses.instr_locations();
        assert!(analyses.is_computed(Analysis::InstrLocations));
        assert!(ptr::eq(first, analyses.instr_locations()));
        let walked = function.instruction_locations();
        for block in function.blocks() {
            for instruction in block.instructions() {
                if let Some(id) = instruction.id() {
                    assert_eq!(first.get(id), walked.get(id));
                }
            }
        }
    }

    /// **A function's memory is built once, over the relation the slot
    /// keeps.** The first request builds it, later ones are the same memory,
    /// and it is the memory a build of its own solving the relation afresh
    /// makes.
    #[test]
    fn the_memory_is_built_once_over_the_kept_relation() {
        let function = memory_effect_fixture();
        let context = AnalysisContext::new(MACHINE);
        let analyses = FunctionAnalyses::<MockTarget>::new(&function, &context);
        assert!(!analyses.is_computed(Analysis::MemorySsa));
        let first = analyses.memory_ssa();
        assert!(analyses.is_computed(Analysis::MemorySsa));
        assert!(ptr::eq(first, analyses.memory_ssa()));
        assert!(!first.operations().is_empty(), "the fixture touches memory");
        assert_eq!(
            format!("{first:?}"),
            format!("{:?}", MemorySsa::build(&function, &context))
        );
    }

    /// Pins that the solve schedule is derived on first use and handed back
    /// unchanged after.
    #[test]
    fn the_solve_schedule_is_derived_once() {
        let function = memory_effect_fixture();
        let analyses =
            FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));

        assert!(!analyses.is_computed(Analysis::SparseDenseSchedule));
        let first = analyses.sparse_dense_schedule();
        assert!(analyses.is_computed(Analysis::SparseDenseSchedule));
        assert!(ptr::eq(first, analyses.sparse_dense_schedule()));
    }

    /// Pins that requesting one analysis leaves the other slots uncomputed.
    #[test]
    fn each_analysis_is_independent() {
        let function = memory_effect_fixture();
        let analyses =
            FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));

        let _ = analyses.points_to();

        assert!(analyses.is_computed(Analysis::PointsTo));
        assert!(
            !analyses.is_computed(Analysis::AliasKeys),
            "asking for one analysis must not compute the others"
        );
    }

    /// Pins that [`FunctionAnalyses::ir`] returns the exact function the handle
    /// borrows, so consumers read the IR through the handle rather than a
    /// parallel reference.
    #[test]
    fn the_handle_yields_the_function_it_describes() {
        let function = memory_effect_fixture();
        let analyses =
            FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));

        assert!(ptr::eq(analyses.ir(), &function));
    }

    /// The def-use slot indexes on first access, is reused afterwards, and a
    /// handle built after the IR changes describes the new IR.
    ///
    /// The borrow is the invalidation: a handle cannot outlive a mutation, so
    /// the only index a caller can read after one is a fresh handle's.
    #[test]
    fn the_def_use_index_is_lazy_and_follows_the_ir() {
        let mut function = const_i32_return(42);
        let returned = match function
            .block(0)
            .and_then(|block| block.control_terminator())
        {
            Some(SsaOp::Return {
                value: Some(value), ..
            }) => *value,
            other => panic!("the fixture returns a value: {other:?}"),
        };
        {
            let analyses =
                FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));
            assert!(!analyses.is_computed(Analysis::DefUse));
            let first = analyses.def_use();
            assert!(analyses.is_computed(Analysis::DefUse));
            assert!(
                !analyses.is_computed(Analysis::PointsTo),
                "asking for the index computes nothing else"
            );
            assert!(ptr::eq(first, analyses.def_use()), "indexed once");
            assert_eq!(first.use_count(returned), 1, "the return reads it");
        }

        // Stop returning the value: nothing reads it any more.
        let block = function.block_mut(0).expect("the entry block");
        let last = block.instructions().len().saturating_sub(1);
        block.instructions_mut()[last].set_op(SsaOp::returns(None));

        let analyses =
            FunctionAnalyses::<MockTarget>::new(&function, &AnalysisContext::new(MACHINE));
        assert!(
            !analyses.is_computed(Analysis::DefUse),
            "a new handle starts empty"
        );
        assert!(analyses.def_use().is_unused(returned), "{function}");
    }
}
