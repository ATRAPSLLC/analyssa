//! Sparse Conditional Constant Propagation (SCCP).
//!
//! SCCP is a powerful constant propagation algorithm that combines sparse
//! analysis (working on SSA def-use chains directly) with conditional
//! propagation (using constant branch conditions to prune unreachable paths).
//!
//! # Algorithm (Wegman & Zadeck 1991)
//!
//! SCCP maintains two lattice structures:
//!
//! 1. **Value lattice** ([`ScalarValue`]): Three-level lattice per variable
//!    - `Top`: No information yet (value might be anything)
//!    - `Constant(c)`: Known compile-time constant
//!    - `Bottom`: Not constant (multiple possible values or varying)
//!
//! 2. **CFG reachability**: Tracks which edges `(from, to)` are executable.
//!    Only executable edges contribute to phi evaluation.
//!
//! ## Worklists
//!
//! The algorithm uses dual worklists:
//! - **SSA worklist**: Variables whose lattice value changed → triggers
//!   re-evaluation of all uses of that variable
//! - **CFG worklist**: Edges that became executable → triggers block discovery
//!   and phi re-evaluation
//!
//! ## Edge-Based Phi Evaluation (Key Insight)
//!
//! Phi nodes are evaluated based on which **edges** are executable, not which
//! blocks are reachable. Consider:
//!
//! ```text
//!        B0
//!       /  \
//!      B1  B2
//!       \  /
//!        B3: y = phi(x from B1, x from B2)
//! ```
//!
//! If the branch in B0 evaluates to a constant `true`, only edge B0→B1 is
//! executable. Even though B3 is reachable via B1→B3, the phi should only
//! consider the operand from B1. This gives `y = 5` (if B1 assigns 5) instead
//! of `y = Bottom` (which would result from merging B1's 5 with B2's unknown).
//!
//! ## Back Edge Handling
//!
//! For loop back edges, phi operands are treated as `Bottom` (unknown/varying).
//! This prevents incorrect constant propagation where a phi appears constant
//! using the first-iteration value but actually varies across iterations
//! (e.g., a Fibonacci counter). Without this, SCCP would incorrectly conclude
//! that `b = phi(1, temp)` is always `1` when `temp = 1` in the first iteration.
//!
//! ## Argument Initialization
//!
//! Argument variables (version 0, defined at function entry) start as `Bottom`
//! rather than `Top`. This is critical because arguments are external inputs
//! that could be anything, and without this distinction, branch conditions
//! depending on arguments would stay at `Top` forever (since no instruction
//! defines them), preventing the branch from ever adding edges.
//!
//! # Differences from Generic Solver
//!
//! Unlike the generic framework solver, SCCP doesn't use block-level transfer
//! functions. Instead, it processes individual SSA instructions and phi nodes
//! directly, which is more efficient for sparse analyses and enables the
//! edge-based precision.
//!
//! # Reference
//!
//! Wegman & Zadeck, "Constant Propagation with Conditional Branches",
//! ACM TOPLAS 1991.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    mem,
};

use crate::{
    analysis::{
        consts::{Destination, Machine, OperandLayout},
        dataflow::lattice::MeetSemiLattice,
        exceptions::FunctionRoots,
    },
    bitset::BitSet,
    graph::{NodeId, RootedGraph, Successors, algorithms::compute_dominators},
    ir::{
        block::SsaBlock, function::SsaFunction, instruction::SsaInstruction, ops::SsaOp,
        phi::PhiNode, value::ConstValue, variable::SsaVarId,
    },
    target::Target,
};

/// Sparse Conditional Constant Propagation analysis.
///
/// This analysis computes which SSA variables have constant values,
/// taking into account that some branches may never be taken.
///
/// # Example
///
/// ```rust
/// use analyssa::{
///     analysis::{
///         SsaCfg,
///         dataflow::{ConstantPropagation, ScalarValue},
///     },
///     ir::SsaVarId,
///     testing,
/// };
///
/// // `const_i32_return(42)` is `v0 = 42; return v0`.
/// let ssa = testing::const_i32_return(42);
/// let graph = SsaCfg::from_ssa(&ssa);
///
/// let mut sccp = ConstantPropagation::new(testing::MACHINE);
/// let results = sccp.analyze(&ssa, &graph);
///
/// // Check if a variable is constant.
/// let var_id = SsaVarId::from_index(0);
/// assert!(matches!(
///     results.get_value(var_id),
///     Some(ScalarValue::Constant(_))
/// ));
/// ```
pub struct ConstantPropagation<T: Target> {
    /// Current value for each SSA variable.
    values: BTreeMap<SsaVarId, ScalarValue<T>>,
    /// Executable CFG edges.
    executable_edges: BTreeSet<(usize, usize)>,
    /// Blocks that have been marked executable.
    executable_blocks: BitSet,
    /// SSA worklist: variables whose values have changed.
    ssa_worklist: VecDeque<SsaVarId>,
    /// CFG worklist: edges that have become executable.
    cfg_worklist: VecDeque<(usize, usize)>,
    /// Loop back edges: `(u, v)` where `v` dominates `u`. Their phi operands are
    /// treated as unknown, because they carry a value from a previous iteration.
    ///
    /// Determined structurally from the dominator tree, not from the order edges
    /// happen to become executable - the latter also matches the second incoming
    /// edge of every ordinary join, which would collapse all merge precision.
    back_edges: BTreeSet<(usize, usize)>,
    /// The analyzed function's exception entry points, read once per run.
    ///
    /// Every executable block inside a protected region makes the region's
    /// handlers executable, whatever its terminator: an instruction anywhere in
    /// the block can throw. Taken from the function rather than the graph so
    /// the answer does not depend on which CFG type the caller passes.
    roots: Option<FunctionRoots>,
    /// The machine the analysed function computes on.
    machine: Machine,
}

impl<T: Target> ConstantPropagation<T> {
    /// Creates a new constant propagation analysis.
    ///
    /// # Arguments
    ///
    /// * `machine` - The machine the analysed function computes on.
    #[must_use]
    pub fn new(machine: Machine) -> Self {
        Self {
            values: BTreeMap::new(),
            executable_edges: BTreeSet::new(),
            executable_blocks: BitSet::new(0),
            ssa_worklist: VecDeque::new(),
            cfg_worklist: VecDeque::new(),
            back_edges: BTreeSet::new(),
            roots: None,
            machine,
        }
    }

    /// Runs the SCCP algorithm on the given SSA function.
    ///
    /// Initializes the lattice and worklists, drains them to a fixpoint, then
    /// moves the accumulated per-variable values and executable-block set out into
    /// a [`SccpResult`], leaving `self` reusable. The CFG is taken by generic
    /// reference rather than a concrete type so the same analysis runs over a
    /// block-level `ControlFlowGraph`, over [`SsaCfg`](crate::analysis::SsaCfg)
    /// and over its narrow terminator view; only
    /// [`RootedGraph`] (for the entry) and [`Successors`] (for outgoing edges) are
    /// required.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function whose variables and instructions are evaluated.
    /// * `cfg` - The control flow graph supplying the entry and successor edges.
    ///
    /// # Returns
    ///
    /// The lattice value for every variable and the set of executable blocks.
    pub fn analyze<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G) -> SccpResult<T>
    where
        G: RootedGraph + Successors,
    {
        self.initialize(ssa, cfg);
        self.propagate(ssa, cfg);

        SccpResult {
            values: mem::take(&mut self.values),
            executable_blocks: mem::take(&mut self.executable_blocks),
        }
    }

    /// Initializes the lattice and seeds the worklists from the entry block.
    ///
    /// Clears all prior state, then gives each variable its starting lattice
    /// value: an argument's initial definition (version 0, no defining
    /// instruction) starts at `Bottom` because it is an external input that could
    /// be anything, while every other variable starts at `Top` and is refined as
    /// its defining instruction is evaluated. Without the `Bottom` seed a branch
    /// condition derived from an argument would sit at `Top` forever - nothing
    /// defines it - and the branch could never mark an edge executable. The entry
    /// block is then marked executable and, crucially, its own definitions are
    /// evaluated *before* its outgoing edges are chosen, so a constant branch
    /// placed first (a common opaque-predicate shape) prunes its dead arm rather
    /// than seeding both successors.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function supplying the variables and entry block.
    /// * `cfg` - The control flow graph supplying the entry node and successors.
    fn initialize<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        self.values.clear();
        self.executable_edges.clear();
        self.executable_blocks = BitSet::new(ssa.block_count());
        self.ssa_worklist.clear();
        self.cfg_worklist.clear();
        self.back_edges.clear();
        self.roots = Some(FunctionRoots::of(ssa)).filter(FunctionRoots::has_flow_roots);

        // Initialize variable values:
        // - Argument variables (version 0, defined at entry) start as Bottom (unknown input)
        // - All other variables start as Top (no information yet)
        //
        // This distinction is critical: arguments are external inputs that could be anything,
        // while other variables are defined by instructions that SCCP will evaluate.
        // Without this, branch conditions depending on arguments stay at Top forever
        // (since no instruction defines them), causing the branch to never add edges.
        for var in ssa.variables() {
            let initial_value = if var.origin().is_argument()
                && var.version() == 0
                && var.def_site().index().is_none()
            {
                // This is the initial definition of an argument - it's an unknown input
                ScalarValue::Bottom
            } else {
                // Regular variable - will be evaluated by instructions
                ScalarValue::Top
            };
            self.values.insert(var.id(), initial_value);
        }

        // Mark entry block as executable
        let entry = cfg.entry().index();
        self.mark_block_executable(entry);

        // Process the entry block's definitions *before* deciding which of its
        // edges are executable, then route the decision through the same
        // `propagate_outgoing_edges` every other block uses. Seeding all
        // successors unconditionally would mark both arms of a constant branch
        // in the entry block executable, so SCCP could never prune one - and
        // that is precisely the shape an opaque predicate takes when the lifter
        // puts it first.
        if let Some(block) = ssa.block(entry) {
            self.process_block_definitions(block, ssa);
            self.propagate_outgoing_edges(entry, block, cfg);
        } else {
            for succ in cfg.successors(cfg.entry()) {
                self.cfg_worklist.push_back((entry, succ.index()));
            }
        }
    }

    /// Drives both worklists to a fixpoint.
    ///
    /// Computes the dominator tree up front so back edges can be identified
    /// structurally - `(u, v)` is a back edge iff `v` dominates `u`. This must be
    /// structural, not order-based: classifying a back edge as "the target was
    /// already executable when the edge arrived" also matches the second and every
    /// later incoming edge of an ordinary join, and since [`Self::evaluate_phi`]
    /// substitutes `Bottom` for back-edge operands, that mistake would collapse
    /// every merge with two live predecessors. The loop then drains the CFG
    /// worklist first (discovering new blocks and phi operands) and processes one
    /// SSA change per turn, terminating only when both worklists are empty.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function whose instructions and phis are evaluated.
    /// * `cfg` - The control flow graph, used for dominators and successor edges.
    fn propagate<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        // Back edges are a *structural* property of the CFG: `(u, v)` is one iff
        // `v` dominates `u`. Deciding it by arrival order instead - "the target
        // was already executable when this edge arrived" - is true of the second
        // and every later incoming edge of *any* join block, not just of loops.
        // Since `evaluate_phi` substitutes Bottom for back-edge operands, that
        // collapsed every merge with two executable predecessors, which is the
        // opposite of what SCCP's phi handling exists to do.
        let dominators = compute_dominators(cfg, cfg.entry());

        // Process until both worklists are empty
        loop {
            // Process CFG worklist first (to discover new blocks)
            while let Some((from, to)) = self.cfg_worklist.pop_front() {
                if self.executable_edges.insert((from, to)) {
                    // PHI operands from back edges represent values that change
                    // across loop iterations and should be treated as unknown.
                    if dominators.dominates(NodeId::new(to), NodeId::new(from)) {
                        self.back_edges.insert((from, to));
                    }
                    // This edge became executable
                    self.process_edge(from, to, ssa, cfg);
                }
            }

            // Process SSA worklist
            if let Some(var) = self.ssa_worklist.pop_front() {
                self.process_variable_uses(var, ssa, cfg);
            } else {
                // Both worklists empty
                break;
            }
        }
    }

    /// Processes a newly executable CFG edge.
    ///
    /// When an edge `(from, to)` becomes executable:
    /// 1. If this is the first edge reaching `to`, mark the block executable and
    ///    process all its definitions
    /// 2. Re-evaluate all phi nodes in `to` since they may now have a new operand
    ///    from the `from` block
    /// 3. If first visit, propagate outgoing edges based on the terminator
    ///
    /// # Arguments
    ///
    /// * `from` - Source block of the newly executable edge.
    /// * `to` - Target block of the edge.
    /// * `ssa` - The SSA function supplying the target block's phis and definitions.
    /// * `cfg` - The control flow graph, used when propagating `to`'s outgoing edges.
    fn process_edge<G>(&mut self, from: usize, to: usize, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let first_visit = !self.is_block_executable(to);

        if first_visit {
            self.mark_block_executable(to);

            // Process all definitions in the block
            if let Some(block) = ssa.block(to) {
                self.process_block_definitions(block, ssa);
            }
        }

        // Re-evaluate phi nodes in the target block.
        // The new edge (from, to) may contribute a new operand value.
        if let Some(block) = ssa.block(to) {
            for phi in block.phi_nodes() {
                // Only re-evaluate if this phi has an operand from the `from` block
                if phi.operand_from(from).is_some() {
                    let new_value = self.evaluate_phi(phi, to);
                    self.update_value(phi.result(), &new_value);
                }
            }
        }

        // If first visit, propagate outgoing edges based on terminator
        if first_visit && let Some(block) = ssa.block(to) {
            self.propagate_outgoing_edges(to, block, cfg);
        }
    }

    /// Processes all definitions in a block (non-phi instructions).
    ///
    /// This evaluates each instruction and updates the value lattice for any
    /// variables defined by the instruction.
    fn process_block_definitions(&mut self, block: &SsaBlock<T>, ssa: &SsaFunction<T>) {
        for instr in block.instructions() {
            self.update_instruction_defs(instr, ssa);
        }
    }

    /// Re-evaluates every use of a variable whose lattice value just changed.
    ///
    /// This is the sparse half of SCCP: rather than re-running whole blocks, it
    /// walks `var`'s def-use chain directly. Uses in blocks not yet executable are
    /// skipped, since their values are not yet meaningful. A phi-operand use
    /// re-evaluates the phi; an ordinary use re-evaluates the defining
    /// instruction, and if that instruction is a terminator its outgoing edges are
    /// re-propagated - the change may decide a branch condition that was
    /// undecided, or add the edges a varying condition needs.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable whose value changed.
    /// * `ssa` - The SSA function supplying the variable's use sites.
    /// * `cfg` - The control flow graph, used when a terminator use re-propagates edges.
    fn process_variable_uses<G>(&mut self, var: SsaVarId, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        // Find all uses of this variable
        if let Some(ssa_var) = ssa.variable(var) {
            for use_site in ssa_var.uses() {
                let block_id = use_site.block();

                // Skip if block is not executable
                if !self.is_block_executable(block_id) {
                    continue;
                }

                if use_site.is_phi_operand() {
                    // Re-evaluate the phi node
                    if let Some(block) = ssa.block(block_id)
                        && let Some(phi) = block.phi(use_site.index())
                    {
                        let new_value = self.evaluate_phi(phi, block_id);
                        self.update_value(phi.result(), &new_value);
                    }
                } else {
                    // Re-evaluate the instruction
                    if let Some(block) = ssa.block(block_id)
                        && let Some(instr) = block.instruction(use_site.index())
                    {
                        self.update_instruction_defs(instr, ssa);

                        // Check if this is a branch instruction
                        if instr.is_terminator() {
                            self.propagate_outgoing_edges(block_id, block, cfg);
                        }
                    }
                }
            }
        }
    }

    /// Marks a block's outgoing edges executable according to its terminator.
    ///
    /// This is where conditional propagation prunes dead paths. A `Branch` whose
    /// condition is a known `Constant` adds only the taken edge; a `Top` condition
    /// adds nothing yet (the direction may still be resolved); a `Bottom` condition
    /// conservatively adds both. A `Switch` on a constant selects the matching arm
    /// (or the default), while an unknown or varying selector adds every arm - the
    /// safe choice for a dynamically computed selector, which is the norm under
    /// control flow obfuscation. A constant that has no truth value (for a
    /// `Branch`) or is not an integer (for a `Switch`) decides nothing and adds
    /// every edge. Returns and other no-successor terminators add nothing;
    /// anything unrecognized falls back to every CFG successor. Whatever the
    /// terminator, a block inside a protected region also adds an edge to each
    /// handler a throw from it can reach.
    ///
    /// # Arguments
    ///
    /// * `block_id` - The block whose outgoing edges are being decided.
    /// * `block` - That block, providing its control terminator.
    /// * `cfg` - The control flow graph, used for the fallback successor set.
    fn propagate_outgoing_edges<G>(&mut self, block_id: usize, block: &SsaBlock<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        // Find the terminator instruction
        match block.control_terminator() {
            Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) => {
                // Conditional branch - check if condition is constant
                match self.get_value(*condition) {
                    ScalarValue::Constant(c) => match c.as_bool() {
                        Some(true) => self.add_cfg_edge(block_id, *true_target),
                        Some(false) => self.add_cfg_edge(block_id, *false_target),
                        // A constant with no truth value (a float, a string)
                        // decides nothing.
                        None => {
                            self.add_cfg_edge(block_id, *true_target);
                            self.add_cfg_edge(block_id, *false_target);
                        }
                    },
                    ScalarValue::Top => {
                        // Unknown - don't add edges yet
                    }
                    ScalarValue::Bottom => {
                        // Could go either way - add both edges
                        self.add_cfg_edge(block_id, *true_target);
                        self.add_cfg_edge(block_id, *false_target);
                    }
                }
            }
            Some(SsaOp::Switch {
                value,
                targets,
                default,
            }) => {
                // Switch statement
                match self.get_value(*value) {
                    ScalarValue::Constant(c) => {
                        // Any integer selects: a non-negative one indexes the
                        // table, and one past it or negative (a huge index,
                        // read unsigned) takes the default. A non-integer
                        // selector decides nothing.
                        if let Some(index) = c.as_u64() {
                            let target = usize::try_from(index)
                                .ok()
                                .and_then(|idx| targets.get(idx))
                                .unwrap_or(default);
                            self.add_cfg_edge(block_id, *target);
                        } else if c.as_i64().is_some() {
                            self.add_cfg_edge(block_id, *default);
                        } else {
                            for &target in targets {
                                self.add_cfg_edge(block_id, target);
                            }
                            self.add_cfg_edge(block_id, *default);
                        }
                    }
                    ScalarValue::Top | ScalarValue::Bottom => {
                        // Unknown or could be anything - conservatively add all edges.
                        // This is critical for control flow obfuscation where the switch
                        // value is computed dynamically and cannot be statically determined.
                        for &target in targets {
                            self.add_cfg_edge(block_id, target);
                        }
                        self.add_cfg_edge(block_id, *default);
                    }
                }
            }
            Some(SsaOp::Jump { target }) => {
                // Unconditional jump
                self.add_cfg_edge(block_id, *target);
            }
            Some(
                SsaOp::Return { .. }
                | SsaOp::Throw { .. }
                | SsaOp::Rethrow
                | SsaOp::EndFilter { .. }
                | SsaOp::InterruptReturn
                | SsaOp::Unreachable,
            ) => {
                // No successors
            }
            // `endfinally` resumes where the `leave` that ran the finally was
            // going; the graph states those targets (none, for a finally only
            // unwinding reaches). Taken as having no successors, it left all
            // code after a `try`/`finally` unexecutable.
            Some(SsaOp::EndFinally) => {
                for succ in cfg.successors(NodeId::new(block_id)) {
                    self.add_cfg_edge(block_id, succ.index());
                }
            }
            _ => {
                // Fall through or unknown terminator - add all CFG successors
                let node = NodeId::new(block_id);
                for succ in cfg.successors(node) {
                    self.add_cfg_edge(block_id, succ.index());
                }
            }
        }

        // A throw from anywhere in a protected block reaches its handlers,
        // whichever way its terminator goes.
        // (`add_cfg_edge` inlined: it would borrow all of `self` while `roots`
        // is borrowed.)
        if let Some(roots) = &self.roots {
            for &handler in roots.exceptional_successors(block_id) {
                if !self.executable_edges.contains(&(block_id, handler)) {
                    self.cfg_worklist.push_back((block_id, handler));
                }
            }
        }
    }

    /// Adds a CFG edge to the worklist if not already executable.
    fn add_cfg_edge(&mut self, from: usize, to: usize) {
        if !self.executable_edges.contains(&(from, to)) {
            self.cfg_worklist.push_back((from, to));
        }
    }

    /// Returns `true` if `block` is currently marked executable.
    ///
    /// Block indices flow in from terminator targets, which the IR permits to
    /// be out of range - a terminator may reference a block that was never
    /// recovered (common in stripped/obfuscated binaries), and the
    /// [`crate::analysis::verifier`] explicitly tolerates such dangling
    /// successors. The `executable_blocks` bitset is sized to exactly
    /// `block_count`, so an out-of-range target is by definition unreachable:
    /// report it `false` instead of indexing past the bitset and panicking.
    fn is_block_executable(&self, block: usize) -> bool {
        self.executable_blocks.contains_checked(block)
    }

    /// Marks `block` executable, ignoring out-of-range indices (see
    /// [`Self::is_block_executable`]).
    fn mark_block_executable(&mut self, block: usize) {
        self.executable_blocks.insert_checked(block);
    }

    /// Evaluates a phi node to get its current value.
    ///
    /// This is the key to SCCP's precision: we only consider operands from
    /// **executable edges**, not just reachable blocks. This allows us to
    /// propagate constants through conditional branches more precisely.
    ///
    /// For example, if we have:
    /// ```text
    /// B0: if (true) goto B1 else goto B2
    /// B1: x = 5; goto B3
    /// B2: x = 10; goto B3
    /// B3: y = phi(x from B1, x from B2)
    /// ```
    /// Even though B3 is reachable, only the edge B1→B3 is executable (because
    /// the branch condition is constant true). So y = 5, not bottom.
    ///
    /// # Arguments
    ///
    /// * `phi` - The phi node to evaluate
    /// * `block_id` - The block containing this phi node (needed to check edge executability)
    fn evaluate_phi(&self, phi: &PhiNode, block_id: usize) -> ScalarValue<T> {
        let mut result = ScalarValue::Top;
        let mut has_executable_operand = false;

        for operand in phi.operands() {
            let pred = operand.predecessor();

            // The key SCCP insight: only consider this operand if the specific
            // edge (pred -> block_id) is executable, not just if pred is reachable.
            if !self.executable_edges.contains(&(pred, block_id)) {
                continue;
            }

            has_executable_operand = true;

            // For back edges (loop edges), treat the operand value as Bottom.
            // Back edge values represent loop-carried dependencies that change
            // across iterations. Using the first-iteration value would incorrectly
            // mark the PHI as constant when it's actually varying.
            //
            // Example: Fibonacci loop where b = phi(1, temp)
            // - First iteration: temp = 0 + 1 = 1, so b = phi(1, 1) looks constant
            // - But iteration 2: temp = 1 + 1 = 2, so b should be 2
            // Without this check, SCCP would incorrectly conclude b is always 1.
            let op_value = if self.back_edges.contains(&(pred, block_id)) {
                ScalarValue::Bottom
            } else {
                self.get_value(operand.value())
            };
            result = result.meet(&op_value);

            // Early exit if already bottom
            if result.is_bottom() {
                break;
            }
        }

        // If no operands were from executable edges, return Top (no information yet)
        if !has_executable_operand {
            return ScalarValue::Top;
        }

        result
    }

    /// Evaluates an SSA instruction to get its result value.
    ///
    /// This performs abstract interpretation of the instruction, computing
    /// what value the result would have given the current lattice values
    /// of the operands. Delegates to [`Machine::evaluate_with_operand_layouts`]
    /// for arithmetic dispatch, each operand read as its type declares it,
    /// while handling lattice Top/Bottom propagation locally. A `None`
    /// from the shared evaluator is disambiguated by whether any operand was `Top`:
    /// an unresolved operand keeps the result at `Top` (still unknown), whereas a
    /// genuinely non-constant operand drives it to `Bottom`.
    ///
    /// # Arguments
    ///
    /// * `op` - The instruction operation to abstractly evaluate.
    /// * `dest` - What `op`'s destination declares, if known.
    /// * `ssa` - The function declaring the operands' types.
    ///
    /// # Returns
    ///
    /// The lattice value of the instruction's result.
    fn evaluate_instruction(
        &self,
        op: &SsaOp<T>,
        dest: Destination,
        ssa: &SsaFunction<T>,
    ) -> ScalarValue<T> {
        // Copy propagates the source's lattice value directly.
        if let SsaOp::Copy { src, .. } = op {
            return self.get_value(*src);
        }

        // Delegate arithmetic to the shared constant evaluator.
        // Track whether any operand was Top (unknown) vs Bottom (varying)
        // so the lattice result is correct.
        let mut saw_top = false;
        // Each operand is read as its type declares it, as the constant
        // evaluator reads it: a vector's elements at their declared width.
        let result = self.machine.evaluate_with_operand_layouts(
            op,
            dest,
            |var| match self.get_value(var) {
                ScalarValue::Constant(c) => Some(c),
                ScalarValue::Top => {
                    saw_top = true;
                    None
                }
                ScalarValue::Bottom => None,
            },
            |var| OperandLayout::of(ssa, var),
        );

        match result {
            Some(c) => ScalarValue::Constant(c),
            // If the shared evaluator returned None but an operand was Top,
            // the result is still unknown (Top), not varying (Bottom).
            None if saw_top => ScalarValue::Top,
            None => ScalarValue::Bottom,
        }
    }

    /// Evaluates an instruction and records the lattice value of each result.
    ///
    /// An operation whose results fold one by one (the double-width
    /// arithmetic, the exact float functions of several results, a
    /// computation's data and status: [`SsaOp::folds_results`]) gives
    /// each result that folds its own value, its operands read as their types
    /// declare them ([`Machine::evaluate_results_with_operand_layouts`]), and
    /// each other result the value of an unknown. Any other instruction's primary
    /// destination receives the evaluated value, and its further defs (a flags
    /// output) are forced to `Bottom`, since SCCP models the primary result
    /// alone for it.
    ///
    /// # Arguments
    ///
    /// * `instr` - The instruction whose defined variables are updated.
    /// * `ssa` - The function declaring the destination's type.
    fn update_instruction_defs(&mut self, instr: &SsaInstruction<T>, ssa: &SsaFunction<T>) {
        if instr.op().folds_results() {
            let mut saw_top = false;
            let mut saw_bottom = false;
            let results = self
                .machine
                .evaluate_results_with_operand_layouts(
                    instr.op(),
                    |var| match self.get_value(var) {
                        ScalarValue::Constant(c) => Some(c),
                        ScalarValue::Top => {
                            saw_top = true;
                            None
                        }
                        ScalarValue::Bottom => {
                            saw_bottom = true;
                            None
                        }
                    },
                    |var| OperandLayout::of(ssa, var),
                )
                .unwrap_or_default();
            // A varying operand makes a result that does not fold vary; an
            // operand not yet reached leaves it unknown; constants the
            // arithmetic has no value for (a zero divisor) vary.
            let unknown = if saw_top && !saw_bottom {
                ScalarValue::Top
            } else {
                ScalarValue::Bottom
            };
            for def in instr.defs() {
                match results.iter().find(|(result, _)| *result == def) {
                    Some((_, value)) => {
                        self.update_value(def, &ScalarValue::Constant(value.clone()))
                    }
                    None => self.update_value(def, &unknown),
                }
            }
            return;
        }
        let primary = instr.op().dest();
        let value =
            self.evaluate_instruction(instr.op(), Destination::of_operation(ssa, instr.op()), ssa);
        for def in instr.defs() {
            if Some(def) == primary {
                self.update_value(def, &value);
            } else {
                self.update_value(def, &ScalarValue::Bottom);
            }
        }
    }

    /// Gets the current value of a variable.
    fn get_value(&self, var: SsaVarId) -> ScalarValue<T> {
        self.values.get(&var).cloned().unwrap_or_default()
    }

    /// Meets a new value into a variable, re-queuing its uses if it moved.
    ///
    /// The stored value is replaced by `old.meet(new)` rather than by `new`
    /// outright, so a variable can only ever descend the lattice
    /// (`Top -> Constant -> Bottom`). That monotonicity is what guarantees the
    /// fixpoint terminates. The variable is pushed onto the SSA worklist only when
    /// the meet actually changed it, avoiding redundant re-evaluation.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to update.
    /// * `new_value` - The newly computed value to meet into the current one.
    fn update_value(&mut self, var: SsaVarId, new_value: &ScalarValue<T>) {
        let old_value = self.values.get(&var).cloned().unwrap_or_default();

        // Apply meet to move down the lattice (values can only decrease)
        let final_value = old_value.meet(new_value);

        if final_value != old_value {
            self.values.insert(var, final_value);
            self.ssa_worklist.push_back(var);
        }
    }
}

/// Scalar value in the SCCP lattice.
///
/// This forms a simple three-level lattice:
/// - Top: No information (might be any value)
/// - Constant: Known compile-time constant
/// - Bottom: Not a constant (multiple possible values)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalarValue<T: Target> {
    /// No information yet (top of lattice).
    #[default]
    Top,
    /// Known constant value.
    Constant(ConstValue<T>),
    /// Multiple possible values (bottom of lattice).
    Bottom,
}

impl<T: Target> ScalarValue<T> {
    /// Returns `true` if this is the top element.
    #[must_use]
    pub const fn is_top(&self) -> bool {
        matches!(self, Self::Top)
    }

    /// Returns `true` if this is the bottom element.
    #[must_use]
    pub const fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }

    /// Returns `true` if this is a known constant.
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns the constant value if this is a constant.
    #[must_use]
    pub const fn as_constant(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }
}

impl<T: Target> MeetSemiLattice for ScalarValue<T> {
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            // Top meets anything yields the other
            (Self::Top, x) | (x, Self::Top) => x.clone(),

            // Same constants stay constant
            (Self::Constant(a), Self::Constant(b)) if a == b => Self::Constant(a.clone()),

            // Different constants or anything with bottom yields bottom
            _ => Self::Bottom,
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }
}

/// Results of SCCP analysis.
#[derive(Debug, Clone)]
pub struct SccpResult<T: Target> {
    /// Value for each SSA variable.
    values: BTreeMap<SsaVarId, ScalarValue<T>>,
    /// Blocks determined to be executable.
    executable_blocks: BitSet,
}

impl<T: Target> SccpResult<T> {
    /// Creates an empty SCCP result.
    ///
    /// This is useful for testing or when no analysis has been performed.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            values: BTreeMap::new(),
            executable_blocks: BitSet::new(0),
        }
    }

    /// Gets the value of an SSA variable.
    #[must_use]
    pub fn get_value(&self, var: SsaVarId) -> Option<&ScalarValue<T>> {
        self.values.get(&var)
    }

    /// Returns `true` if a variable is known to be constant.
    #[must_use]
    pub fn is_constant(&self, var: SsaVarId) -> bool {
        self.values
            .get(&var)
            .is_some_and(|v| matches!(v, ScalarValue::Constant(_)))
    }

    /// Returns the constant value of a variable if known.
    #[must_use]
    pub fn constant_value(&self, var: SsaVarId) -> Option<&ConstValue<T>> {
        self.values.get(&var).and_then(|v| match v {
            ScalarValue::Constant(c) => Some(c),
            _ => None,
        })
    }

    /// Returns `true` if a block is executable (reachable).
    ///
    /// An out-of-range `block` index (past the analyzed `block_count`) is
    /// reported `false` rather than panicking - terminator targets may be
    /// dangling and callers should treat unknown blocks as unreachable.
    #[must_use]
    pub fn is_block_executable(&self, block: usize) -> bool {
        block < self.executable_blocks.len() && self.executable_blocks.contains(block)
    }

    /// Returns an iterator over all constant variables.
    pub fn constants(&self) -> impl Iterator<Item = (SsaVarId, &ConstValue<T>)> {
        self.values.iter().filter_map(|(var, val)| match val {
            ScalarValue::Constant(c) => Some((*var, c)),
            _ => None,
        })
    }

    /// Returns an iterator over all executable blocks.
    pub fn executable_blocks(&self) -> impl Iterator<Item = usize> + '_ {
        self.executable_blocks.iter()
    }

    /// Returns the number of variables found to be constant.
    #[must_use]
    pub fn constant_count(&self) -> usize {
        self.values
            .values()
            .filter(|v| matches!(v, ScalarValue::Constant(_)))
            .count()
    }

    /// Returns the number of executable blocks.
    #[must_use]
    pub fn executable_block_count(&self) -> usize {
        self.executable_blocks.count()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        analysis::{cfg::SsaCfg, dataflow::lattice::MeetSemiLattice},
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            function::SsaFunction,
            instruction::SsaInstruction,
            lanes::VectorConst,
            ops::{
                Computation, Kernel, LaneControl, SsaOp, VectorBinaryKind, VectorElement,
                VectorPredicateLayout, VectorRounding, VectorStatus,
            },
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        target::VectorTailPolicy,
        testing::{MACHINE, MockTarget, MockType},
    };

    type Sv = ScalarValue<MockTarget>;

    /// **Each result of the double-width arithmetic has its own value.** A
    /// product of constants is two constants; a division by a zero constant
    /// has no value and both its results vary.
    #[test]
    fn each_wide_result_has_its_own_value() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        for index in 0..8u16 {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, usize::from(index)),
                MockType::I32,
            );
        }
        let var = SsaVarId::from_index;
        let mut b0 = SsaBlock::new(0);
        for op in [
            SsaOp::Const {
                dest: var(0),
                value: ConstValue::I32(-3),
            },
            SsaOp::Const {
                dest: var(1),
                value: ConstValue::I32(5),
            },
            SsaOp::WideMul {
                low: var(2),
                high: var(3),
                left: var(0),
                right: var(1),
                unsigned: false,
            },
            SsaOp::Const {
                dest: var(4),
                value: ConstValue::I32(0),
            },
            SsaOp::WideDiv {
                quotient: var(5),
                remainder: var(6),
                high: var(0),
                low: var(1),
                divisor: var(4),
                unsigned: false,
            },
            SsaOp::returns(Some(var(2))),
        ] {
            b0.add_instruction(SsaInstruction::synthetic(op));
        }
        ssa.add_block(b0);
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let mut sccp: ConstantPropagation<MockTarget> = ConstantPropagation::new(MACHINE);
        let result = sccp.analyze(&ssa, &cfg);
        assert_eq!(result.constant_value(var(2)), Some(&ConstValue::I32(-15)));
        assert_eq!(result.constant_value(var(3)), Some(&ConstValue::I32(-1)));
        assert_eq!(result.get_value(var(5)), Some(&Sv::Bottom));
        assert_eq!(result.get_value(var(6)), Some(&Sv::Bottom));
    }

    /// **A computation over lanes gives its status a value where its data
    /// has none.** `vfadd.vv` of two doubles at a length of one leaves its
    /// agnostic tail element unspecified, so its data is no constant; the one
    /// element executed, `1 + 2^-60`, is inexact, so its status is 1.
    #[test]
    fn a_lane_status_folds_where_the_data_does_not() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let types = [
            MockType::U64,
            MockType::U64,
            MockType::V2F64,
            MockType::V2F64,
            MockType::V2F64,
            MockType::V2F64,
            MockType::U8,
        ];
        for (index, var_type) in (0u16..).zip(types) {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, usize::from(index)),
                var_type,
            );
        }
        let var = SsaVarId::from_index;
        let doubles = |values: [f64; 2]| {
            ConstValue::Vector(
                VectorConst::from_lanes(64, values.map(|value| u128::from(value.to_bits())))
                    .unwrap(),
            )
        };
        let mut b0 = SsaBlock::new(0);
        for op in [
            SsaOp::Const {
                dest: var(0),
                value: ConstValue::U64(0),
            },
            SsaOp::Const {
                dest: var(1),
                value: ConstValue::U64(1),
            },
            SsaOp::Const {
                dest: var(3),
                value: doubles([1.0, 1.0]),
            },
            SsaOp::Const {
                dest: var(4),
                value: doubles([2.0f64.powi(-60), 1.0]),
            },
            SsaOp::compute(
                Computation {
                    kernel: Kernel::Binary(VectorBinaryKind::Add),
                    element: VectorElement::float(64),
                    predicate: VectorPredicateLayout::Packed,
                    rounding: VectorRounding::None,
                    status: VectorStatus::FloatingExceptions,
                    canonical_nan: true,
                    lanes: Some(LaneControl {
                        mask_tail: false,
                        tail: VectorTailPolicy::Agnostic,
                        inactive: None,
                        no_passthrough: false,
                    }),
                },
                vec![var(5), var(6)],
                vec![var(0), var(1), var(2), var(3), var(4)],
            ),
            SsaOp::returns(Some(var(6))),
        ] {
            b0.add_instruction(SsaInstruction::synthetic(op));
        }
        ssa.add_block(b0);
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let mut sccp: ConstantPropagation<MockTarget> = ConstantPropagation::new(MACHINE);
        let result = sccp.analyze(&ssa, &cfg);
        assert_eq!(result.constant_value(var(6)), Some(&ConstValue::U8(1)));
        assert_eq!(result.constant_value(var(5)), None);
    }

    /// `B0: jump B1` / `B1 (try): leave B3` / `B2 (finally): endfinally` /
    /// `B3: y = 5; ret y`.
    ///
    /// A `leave` runs the `finally` and `endfinally` resumes at the leave's
    /// target: code after a `try`/`finally` runs. Read as a terminator with no
    /// successors, `endfinally` left every block after it unexecutable, and a
    /// value merging with that code was folded to the constant from before it.
    #[test]
    fn code_after_a_finally_is_executable() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let y = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(3, 0),
            MockType::I32,
        );
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Leave { target: 3 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::EndFinally));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(5),
        }));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(y))));
        ssa.add_block(b3);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, 2),
            handler_range: BlockRange::new(2, 3),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let result = ConstantPropagation::<MockTarget>::new(MACHINE).analyze(&ssa, &cfg);

        assert!(result.is_block_executable(3), "the leave's target runs");
        assert_eq!(result.constant_value(y), Some(&ConstValue::I32(5)));
    }
    type Cv = ConstValue<MockTarget>;

    /// A merge whose arms produce the *same* constant must fold to that
    /// constant. This is the whole point of SCCP's phi handling.
    ///
    /// `B0: branch -> B1, B2` / `B1: x = 7` / `B2: y = 7` / `B3: p = phi(x, y)`.
    /// Classifying an edge as a back edge by "the target was already executable"
    /// is true of the *second* incoming edge of any join, so `p`'s second operand
    /// gets substituted with Bottom and the merge collapses.
    #[test]
    fn a_diamond_whose_arms_agree_folds_to_that_constant() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let mk = |ssa: &mut SsaFunction<MockTarget>, o: u16, b: usize, i: usize| {
            ssa.create_variable(
                VariableOrigin::Local(o),
                0,
                DefSite::instruction(b, i),
                MockType::I32,
            )
        };
        let cond = mk(&mut ssa, 0, 0, 0);
        let x = mk(&mut ssa, 1, 1, 0);
        let y = mk(&mut ssa, 2, 2, 0);
        let p = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(3), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        for (block_id, var) in [(1usize, x), (2usize, y)] {
            let mut block = SsaBlock::new(block_id);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest: var,
                value: ConstValue::I32(7),
            }));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(block);
        }

        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(p, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(x, 1));
        phi.add_operand(PhiOperand::new(y, 2));
        b3.add_phi(phi);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(p))));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let result = ConstantPropagation::<MockTarget>::new(MACHINE).analyze(&ssa, &cfg);

        assert_eq!(
            result.constant_value(p),
            Some(&ConstValue::I32(7)),
            "both arms of the diamond produce 7, so the merge is 7"
        );
    }

    /// The counterpart the back-edge substitution was written for: a genuine
    /// loop-carried value must *not* be folded to its first-iteration value.
    ///
    /// `B0: c = 1; jump B1` / `B1: p = phi(c@B0, t@B1); t = p + c; jump B1`.
    #[test]
    fn a_loop_carried_phi_is_not_constant() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let c = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let p = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(1), MockType::I32);
        let t = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: c,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(p, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(c, 0));
        phi.add_operand(PhiOperand::new(t, 1));
        b1.add_phi(phi);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: t,
            left: p,
            right: c,
            flags: None,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let result = ConstantPropagation::<MockTarget>::new(MACHINE).analyze(&ssa, &cfg);

        assert_eq!(
            result.constant_value(p),
            None,
            "p takes 1 then 2 then 3 ... and must not be folded to its first value"
        );
    }

    /// Pins the SCCP value lattice: `Top` is the identity, equal constants
    /// survive, and unequal constants or any `Bottom` fall to `Bottom`.
    #[test]
    fn test_scalar_value_meet() {
        // Top meets anything yields the other
        assert_eq!(
            Sv::Top.meet(&Sv::Constant(Cv::I32(5))),
            Sv::Constant(Cv::I32(5))
        );

        // Same constants stay constant
        assert_eq!(
            Sv::Constant(Cv::I32(5)).meet(&Sv::Constant(Cv::I32(5))),
            Sv::Constant(Cv::I32(5))
        );

        // Different constants become bottom
        assert_eq!(
            Sv::Constant(Cv::I32(5)).meet(&Sv::Constant(Cv::I32(10))),
            Sv::Bottom
        );

        // Bottom meets anything yields bottom
        assert_eq!(Sv::Bottom.meet(&Sv::Constant(Cv::I32(5))), Sv::Bottom);
    }

    /// The `is_top`/`is_constant`/`is_bottom`/`as_constant` accessors report the
    /// correct lattice level for each `ScalarValue` variant.
    #[test]
    fn test_scalar_value_accessors() {
        let top: Sv = Sv::Top;
        let const_val: Sv = Sv::Constant(Cv::I32(42));
        let bottom: Sv = Sv::Bottom;

        assert!(top.is_top());
        assert!(!top.is_constant());
        assert!(!top.is_bottom());

        assert!(!const_val.is_top());
        assert!(const_val.is_constant());
        assert!(!const_val.is_bottom());
        assert_eq!(const_val.as_constant(), Some(&ConstValue::I32(42)));

        assert!(!bottom.is_top());
        assert!(!bottom.is_constant());
        assert!(bottom.is_bottom());
    }

    /// The `SccpResult` query surface reports constants, constant values,
    /// executable blocks, and counts, and treats out-of-range blocks as unreachable.
    #[test]
    fn test_sccp_result() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);

        let mut values: BTreeMap<SsaVarId, Sv> = BTreeMap::new();
        values.insert(v0, Sv::Constant(Cv::I32(42)));
        values.insert(v1, Sv::Bottom);
        values.insert(v2, Sv::Top);

        let mut executable_blocks = BitSet::new(3);
        executable_blocks.insert(0);
        executable_blocks.insert(1);

        let result = SccpResult {
            values,
            executable_blocks,
        };

        assert!(result.is_constant(v0));
        assert!(!result.is_constant(v1));
        assert!(!result.is_constant(v2));

        assert_eq!(result.constant_value(v0), Some(&ConstValue::I32(42)));
        assert_eq!(result.constant_value(v1), None);

        assert!(result.is_block_executable(0));
        assert!(result.is_block_executable(1));
        assert!(!result.is_block_executable(2));
        // Out-of-range index is reported unreachable, never a panic.
        assert!(!result.is_block_executable(9_999));

        assert_eq!(result.constant_count(), 1);
        assert_eq!(result.executable_block_count(), 2);
    }

    /// A terminator whose target block was never recovered is treated as
    /// unreachable rather than panicking on an out-of-range bitset index.
    #[test]
    fn out_of_range_branch_targets_do_not_panic() {
        // A terminator may reference a block that was never recovered. SCCP's
        // `executable_blocks` bitset is sized to `block_count`, so an
        // out-of-range target index must be treated as unreachable rather than
        // reaching the asserting `BitSet::contains`. The condition is an
        // unconstrained argument (`Bottom`), so both edges - including the
        // dangling one - are explored.
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            // Block 99 does not exist - only blocks 0 and 1 are present.
            false_target: 99,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let cfg = SsaCfg::from_ssa(&ssa);
        let mut sccp: ConstantPropagation<MockTarget> = ConstantPropagation::new(MACHINE);
        let result = sccp.analyze(&ssa, &cfg);

        // Real blocks reachable, the dangling target never marked executable.
        assert!(result.is_block_executable(0));
        assert!(result.is_block_executable(1));
        assert!(!result.is_block_executable(99));
    }
}
