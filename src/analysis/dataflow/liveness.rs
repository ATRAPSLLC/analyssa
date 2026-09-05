//! Live variable analysis using the dataflow framework.
//!
//! A variable is *live* at a program point if there exists a path from that
//! point to a use of the variable without passing through a definition of the
//! variable. In SSA form, since each variable is defined exactly once, this
//! simplifies to: a variable is live if it will be used on some future path.
//!
//! # Uses
//!
//! - **Dead code elimination**: If a definition's result is never live, it's dead
//! - **Register allocation**: Variables live simultaneously need different registers
//! - **Debugging**: Determine which variables can be inspected at a breakpoint
//!
//! # Algorithm
//!
//! This is a backward data flow analysis using the generic framework:
//!
//! ## Dataflow Equations
//!
//! | Set | Definition |
//! |-----|------------|
//! | `USE[B]` | Variables used in B before any definition |
//! | `DEF[B]` | Variables defined in B (phi nodes + instruction defs) |
//! | `OUT[B]` | `∪ { IN(s) for s in succ(B) }` |
//! | `IN[B]` | `USE[B] ∪ (OUT[B] \\ DEF[B])` |
//!
//! ## PHI Operand Handling
//!
//! PHI operand uses are placed at the END of the predecessor block (ECMA-335 /
//! standard SSA semantics: phi copies execute at the predecessor's outgoing edge).
//! This ensures backward dataflow correctly propagates liveness from the predecessor
//! back through all intermediate blocks to the definition.
//!
//! ## Lattice
//!
//! The `LivenessResult` lattice uses `MeetSemiLattice` with:
//! - **Meet = union**: A variable is live if it's live on ANY successor path
//!   (may analysis — over-approximate liveness)
//! - **Boundary**: At function exit, no variables are live
//! - **Initial**: All blocks start with empty live sets

use crate::{
    analysis::dataflow::{
        framework::{DataFlowAnalysis, Direction},
        lattice::MeetSemiLattice,
    },
    bitset::BitSet,
    ir::{block::SsaBlock, function::SsaFunction, variable::SsaVarId},
    target::Target,
};

/// Live variable analysis.
///
/// Computes which variables are live at each program point.
/// A variable is live if its value may be used on some path from
/// that point forward.
///
/// # Example
///
/// ```rust
/// use analyssa::{
///     analysis::{
///         dataflow::{DataFlowSolver, LiveVariables},
///         exceptions::EhCfg,
///     },
///     ir::SsaVarId,
///     testing,
/// };
///
/// // Block 1 defines the loop counter phi; block 2 returns it.
/// let ssa = testing::loop_counter_fixture();
/// let graph = EhCfg::from_ssa(&ssa);
///
/// let analysis = LiveVariables::new(&ssa);
/// let solver = DataFlowSolver::new(analysis);
/// let results = solver.solve(&ssa, &graph);
///
/// // The counter is live on entry to block 2, because block 2 returns it.
/// let counter = SsaVarId::from_index(1);
/// let live_in = results.in_state(2).unwrap();
/// assert!(live_in.variables().any(|var| var == counter));
///
/// // Nothing is live after the return terminator.
/// assert_eq!(results.out_state(2).unwrap().variables().count(), 0);
/// ```
pub struct LiveVariables {
    /// Number of variables in the function.
    num_vars: usize,
    /// USE sets for each block (variables used before definition).
    use_sets: Vec<BitSet>,
    /// DEF sets for each block (variables defined).
    def_sets: Vec<BitSet>,
    /// Values consumed by a successor's phi on each block's outgoing edges.
    ///
    /// `phi_out[b]` holds every variable that some successor of `b` reads as a
    /// phi operand on the edge from `b` — *including* ones `b` itself defines.
    /// See [`LiveVariables::live_out`] for why this cannot be recovered from the
    /// solver's `out_state`.
    phi_out: Vec<BitSet>,
}

impl LiveVariables {
    /// Creates a new live variables analysis for the given SSA function.
    #[must_use]
    pub fn new<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let num_vars = ssa.variable_count();
        let num_blocks = ssa.block_count();

        let mut use_sets = Vec::with_capacity(num_blocks);
        let mut def_sets = Vec::with_capacity(num_blocks);

        // Phase 1: Initialize use/def sets without PHI operands
        for block in ssa.blocks() {
            let mut uses = BitSet::new(num_vars);
            let mut defs = BitSet::new(num_vars);

            // Process phi nodes: they define variables
            for phi in block.phi_nodes() {
                if let Some(def_idx) = ssa.var_index(phi.result()) {
                    defs.insert(def_idx);
                }
            }

            // Process instructions in forward order
            for instr in block.instructions() {
                // Uses first (before def, since this is the "USE before DEF" set)
                instr.for_each_use(|use_var| {
                    if let Some(var_idx) = ssa.var_index(use_var)
                        && !defs.contains(var_idx)
                    {
                        uses.insert(var_idx);
                    }
                });

                // Then definition
                for def in instr.defs() {
                    if let Some(def_idx) = ssa.var_index(def) {
                        defs.insert(def_idx);
                    }
                }
            }

            use_sets.push(uses);
            def_sets.push(defs);
        }

        // Phase 2: Add PHI operand uses to their PREDECESSOR blocks.
        // A PHI operand `v<-B_pred` means variable v is used at the END
        // of B_pred (ECMA-335 / SSA semantics: phi copies happen at the
        // predecessor's outgoing edge). Placing the use in the predecessor
        // ensures backward dataflow propagates liveness from the predecessor
        // back through all intermediate blocks to the definition.
        let mut phi_out = vec![BitSet::new(num_vars); num_blocks];
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                for op in phi.operands() {
                    let pred = op.predecessor();
                    if let Some(var_idx) = ssa.var_index(op.value()) {
                        // Recorded unconditionally, unlike the USE insert below:
                        // the value crosses the edge whether or not `pred`
                        // defines it, and a value `pred` defines is precisely the
                        // case the USE set must exclude but live-out must not.
                        if let Some(slot) = phi_out.get_mut(pred) {
                            slot.insert_checked(var_idx);
                        }
                        let already_def = def_sets.get(pred).is_some_and(|s| s.contains(var_idx));
                        if !already_def && let Some(slot) = use_sets.get_mut(pred) {
                            slot.insert(var_idx);
                        }
                    }
                }
            }
        }

        Self {
            num_vars,
            use_sets,
            def_sets,
            phi_out,
        }
    }

    /// Returns the values live on `block`'s outgoing edges.
    ///
    /// **Use this rather than the solver's raw `out_state`.** `out_state` is the
    /// meet of the successors' IN sets, and a value consumed *only* by a
    /// successor's phi never appears there: phi-operand uses are relocated into
    /// the predecessor's USE set, which feeds `IN[pred]`, not `OUT[pred]`. The
    /// value is genuinely live across the edge — the phi copy reads it — so raw
    /// `out_state` under-approximates liveness, which is the unsafe direction
    /// for a may-analysis and would let a register allocator reuse a live
    /// register.
    ///
    /// This computes `OUT[B] = out_state(B) ∪ PhiUses(B)`, where `PhiUses(B)` is
    /// every value some successor reads as a phi operand on an edge from `B`.
    #[must_use]
    pub fn live_out(&self, block: usize, out_state: &LivenessResult) -> LivenessResult {
        let mut live = out_state.live.clone();
        if let Some(phi_uses) = self.phi_out.get(block) {
            live.union_with(phi_uses);
        }
        LivenessResult { live }
    }

    /// Returns the values a successor's phi reads on `block`'s outgoing edges.
    #[must_use]
    pub fn phi_out_set(&self, block: usize) -> Option<&BitSet> {
        self.phi_out.get(block)
    }

    /// Returns the number of variables being tracked.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Returns the USE set for a block.
    #[must_use]
    pub fn use_set(&self, block: usize) -> Option<&BitSet> {
        self.use_sets.get(block)
    }

    /// Returns the DEF set for a block.
    #[must_use]
    pub fn def_set(&self, block: usize) -> Option<&BitSet> {
        self.def_sets.get(block)
    }
}

impl<T: Target> DataFlowAnalysis<T> for LiveVariables {
    type Lattice = LivenessResult;
    const DIRECTION: Direction = Direction::Backward;

    fn boundary(&self, _ssa: &SsaFunction<T>) -> Self::Lattice {
        // At function exit, no variables are live
        // (unless we're tracking return values, which we could add)
        LivenessResult {
            live: BitSet::new(self.num_vars),
        }
    }

    fn initial(&self, _ssa: &SsaFunction<T>) -> Self::Lattice {
        // Initially, no variables are live
        LivenessResult {
            live: BitSet::new(self.num_vars),
        }
    }

    fn transfer(
        &self,
        block_id: usize,
        _block: &SsaBlock<T>,
        output: &Self::Lattice,
        _ssa: &SsaFunction<T>,
    ) -> Self::Lattice {
        // For backward analysis: IN = USE ∪ (OUT - DEF)
        let mut result = output.live.clone();

        // Remove definitions (OUT - DEF)
        if let Some(d) = self.def_sets.get(block_id) {
            result.difference_with(d);
        }

        // Add uses (USE ∪ ...)
        if let Some(u) = self.use_sets.get(block_id) {
            result.union_with(u);
        }

        LivenessResult { live: result }
    }
}

/// Result of live variable analysis for a single program point.
#[derive(Debug, Clone, PartialEq)]
pub struct LivenessResult {
    /// Bit vector of live variables (indexed by `SsaVarId`).
    live: BitSet,
}

impl LivenessResult {
    /// Creates a new empty result.
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        Self {
            live: BitSet::new(num_vars),
        }
    }

    /// Returns `true` if the given variable is live at this point.
    #[must_use]
    pub fn is_live(&self, var: SsaVarId) -> bool {
        let idx = var.index();
        idx < self.live.len() && self.live.contains(idx)
    }

    /// Returns an iterator over all live variables.
    pub fn variables(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.live.iter().map(SsaVarId::from_index)
    }

    /// Returns the number of live variables.
    #[must_use]
    pub fn count(&self) -> usize {
        self.live.count()
    }

    /// Returns `true` if no variables are live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Marks a variable as live.
    pub fn add(&mut self, var: SsaVarId) {
        let idx = var.index();
        if idx < self.live.len() {
            self.live.insert(idx);
        }
    }

    /// Marks a variable as not live.
    pub fn remove(&mut self, var: SsaVarId) {
        let idx = var.index();
        if idx < self.live.len() {
            self.live.remove(idx);
        }
    }

    /// Returns the underlying bit set.
    #[must_use]
    pub const fn as_bitset(&self) -> &BitSet {
        &self.live
    }
}

impl MeetSemiLattice for LivenessResult {
    /// Meet is union (a variable is live if it's live on ANY successor path).
    fn meet(&self, other: &Self) -> Self {
        let mut result = self.live.clone();
        result.union_with(&other.live);
        Self { live: result }
    }

    /// Union in place — no temporary set per predecessor.
    fn meet_into(&mut self, other: &Self) {
        self.live.union_with(&other.live);
    }

    fn is_bottom(&self) -> bool {
        // Bottom is when all variables are live (full set).
        self.live.is_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            SsaBlock, SsaInstruction,
            ops::SsaOp,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// A value consumed *only* by a successor's phi is live across the edge, but
    /// never appears in the solver's `out_state`: phi-operand uses are relocated
    /// into the predecessor's USE set, which feeds `IN[pred]`, not `OUT[pred]`.
    /// Under-approximating live-out is the unsafe direction — a register
    /// allocator reading it would reuse a register that is still live.
    #[test]
    fn live_out_includes_values_a_successor_phi_consumes() {
        // B0 defines `v` and jumps to B1, whose phi is `v`'s only consumer.
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let value = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let merged =
            ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(1), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(value, 0));
        b1.add_phi(phi);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(merged),
        }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let analysis = LiveVariables::new(&ssa);
        let var_idx = ssa.var_index(value).expect("value is registered");

        assert!(
            analysis
                .phi_out_set(0)
                .is_some_and(|set| set.contains(var_idx)),
            "the phi operand crosses the B0->B1 edge"
        );

        // The raw successor-meet does not see it: B1's IN excludes `value`
        // because the phi *defines* rather than uses it there.
        let bare = LivenessResult {
            live: BitSet::new(analysis.num_variables()),
        };
        assert!(
            !bare.live.contains(var_idx),
            "precondition: the bare out_state does not contain it"
        );

        // `live_out` adds it back.
        let live_out = analysis.live_out(0, &bare);
        assert!(
            live_out.is_live(value),
            "live_out must report a value the successor's phi consumes"
        );
    }

    #[test]
    fn test_liveness_result() {
        let mut result = LivenessResult::new(10);
        assert!(result.is_empty());

        result.add(SsaVarId::from_index(0));
        result.add(SsaVarId::from_index(5));

        assert!(!result.is_empty());
        assert_eq!(result.count(), 2);
        assert!(result.is_live(SsaVarId::from_index(0)));
        assert!(result.is_live(SsaVarId::from_index(5)));
        assert!(!result.is_live(SsaVarId::from_index(1)));

        result.remove(SsaVarId::from_index(0));
        assert!(!result.is_live(SsaVarId::from_index(0)));
        assert_eq!(result.count(), 1);
    }

    #[test]
    fn test_liveness_meet() {
        let mut a = LivenessResult::new(10);
        let mut b = LivenessResult::new(10);

        a.add(SsaVarId::from_index(0));
        a.add(SsaVarId::from_index(1));
        b.add(SsaVarId::from_index(1));
        b.add(SsaVarId::from_index(2));

        let result = a.meet(&b);
        assert!(result.is_live(SsaVarId::from_index(0)));
        assert!(result.is_live(SsaVarId::from_index(1)));
        assert!(result.is_live(SsaVarId::from_index(2)));
        assert_eq!(result.count(), 3);
    }

    #[test]
    fn test_liveness_iterator() {
        let mut result = LivenessResult::new(100);
        result.add(SsaVarId::from_index(5));
        result.add(SsaVarId::from_index(42));
        result.add(SsaVarId::from_index(99));

        let vars: Vec<_> = result.variables().collect();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&SsaVarId::from_index(5)));
        assert!(vars.contains(&SsaVarId::from_index(42)));
        assert!(vars.contains(&SsaVarId::from_index(99)));
    }
}
