//! SSA verifier for validating SSA invariants.
//!
//! Provides comprehensive verification of SSA form at three levels to catch
//! invariant violations introduced by transformations, preventing silent
//! corruption that would manifest as broken codegen or incorrect deobfuscation.
//!
//! # Verification Levels
//!
//! ## Quick (O(n))
//!
//! Checks the fundamental SSA invariants:
//! - **Single definition**: Every variable is defined at most once (scan all
//!   phi nodes and instructions, tracking destination variables in a HashMap).
//! - **Block structure**: Every block with successors has a terminator.
//!   Terminators are the last instruction in their block.
//! - **No intra-block cycles**: A variable is not used before its definition
//!   within the same block (use-before-def cycle detection using def_indices map).
//! - **No placeholders**: No `SsaVarId::PLACEHOLDER` (usize::MAX) remains in
//!   finalized SSA.
//! - **No self-referential instructions**: An instruction's destination does
//!   not appear in its own operands.
//!
//! ## Standard (O(n * m))
//!
//! Adds:
//! - **Def-use chain integrity**: Every variable used in an instruction or phi
//!   operand has a definition somewhere (either in the variables vec or a block).
//! - **Phi operand coverage**: Each phi node has exactly one operand per CFG
//!   predecessor, no operands from non-predecessor blocks, and no phis in the
//!   entry block (which has no predecessors).
//! - **Variable registration**: Every variable used in a block is registered in
//!   the function's variables vec, and vice-versa (no orphan or unregistered vars).
//!
//! ## Full (O(n^2) worst case)
//!
//! Adds:
//! - **Dominance verification**: Every use of a variable must be dominated by its
//!   definition (standard SSA requirement). For phi operands, the use is considered
//!   to be at the end of the predecessor block, following standard SSA semantics.
//!   Only verified for reachable blocks.
//!
//! # Error Types
//!
//! Various [`VerifierError`] variants describe each invariant violation with
//! precise locations (block, instruction, variable) for debugging.
//!
//! # Complexity
//!
//! | Level | Time | Description |
//! |-------|------|-------------|
//! | Quick | O(n) | Single pass over blocks and instructions |
//! | Standard | O(n*m) | n = blocks, m = avg predecessors per block |
//! | Full | O(b^2) | Dominator computation + O(b * u) queries |

use crate::{
    BitSet,
    analysis::cfg::SsaCfg,
    graph::{
        NodeId, RootedGraph,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        function::SsaFunction,
        ops::{
            AtomicAccessWidth, AtomicOrdering, ControlEffect, MemoryAccessSemantics,
            MemoryEffectLocation, NativeClobber, NativeStateAccessKind, SsaEffectKind, SsaEffects,
            SsaOp,
        },
        variable::{DefSite, SsaVarId, SsaVariable, VariableOrigin},
        varstore::VarMap,
    },
    target::{Target, VectorDescriptor, VectorMaskDescriptor, VectorMaskShape},
};

/// Definition site for verifier error reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierDefSite {
    /// Index of the block containing the definition.
    pub block: usize,
    /// Location class of the definition within the block.
    pub kind: DefKind,
}

/// What kind of definition produced a variable within a block.
///
/// Distinguishes between phi node definitions (from SSA control flow merges)
/// and instruction definitions (from regular operations). Used by the verifier
/// to provide precise error locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefKind {
    /// Variable is defined by a phi node at the given phi index within the block.
    Phi(usize),
    /// Variable is defined by an instruction at the given instruction index within the block.
    Instruction(usize),
}

/// Errors detected by the SSA verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierError {
    /// A variable is used but never defined.
    UndefinedUse {
        /// Index of the block containing the use.
        block: usize,
        /// Index of the instruction containing the use.
        instr_idx: usize,
        /// Variable read before any verifier-visible definition.
        var: SsaVarId,
    },
    /// A phi node has more than one operand naming the same predecessor edge.
    ///
    /// One edge carries one value, so duplicates have no defined meaning and
    /// consumers disagree about them: [`PhiNode::operand_from`] returns the first
    /// and discards the rest, while SCCP meets all of them and yields Bottom.
    ///
    /// [`PhiNode::operand_from`]: crate::ir::phi::PhiNode::operand_from
    DuplicatePhiOperand {
        /// Index of the block containing the phi node.
        block: usize,
        /// Index of the phi node in the block's phi list.
        phi_idx: usize,
        /// Predecessor named by more than one operand.
        pred: usize,
    },
    /// A phi node is missing an operand for a CFG predecessor.
    MissingPhiOperand {
        /// Index of the block containing the phi node.
        block: usize,
        /// Index of the phi node in the block's phi list.
        phi_idx: usize,
        /// Predecessor block that lacks a corresponding phi operand.
        missing_pred: usize,
    },
    /// A phi node has an operand for a non-predecessor block.
    ExtraPhiOperand {
        /// Index of the block containing the phi node.
        block: usize,
        /// Index of the phi node in the block's phi list.
        phi_idx: usize,
        /// Block referenced by the phi operand that is not a CFG predecessor.
        extra_pred: usize,
    },
    /// A variable is defined more than once.
    DuplicateDefinition {
        /// Variable that has more than one definition site.
        var: SsaVarId,
        /// First definition site found by the verifier.
        def1: VerifierDefSite,
        /// Later conflicting definition site found by the verifier.
        def2: VerifierDefSite,
    },
    /// A variable exists in the variables vec but has no definition in any block.
    OrphanVariable {
        /// Registered variable that has no block-local definition.
        var: SsaVarId,
    },
    /// A variable appears in an instruction but is not in the variables vec.
    UnregisteredVariable {
        /// Variable referenced by IR but absent from the function's registry.
        var: SsaVarId,
    },
    /// A block has successors but no terminator instruction.
    MissingTerminator {
        /// Index of the unterminated block.
        block: usize,
    },
    /// A phi node appears in the entry block (block 0), which has no predecessors
    /// in a well-formed CFG.
    PhiInEntryBlock {
        /// Entry block index that contains the invalid phi.
        block: usize,
        /// Index of the phi node in the entry block's phi list.
        phi_idx: usize,
    },
    /// A variable is used in a block not dominated by its definition block.
    DominanceViolation {
        /// Variable whose definition does not dominate its use.
        var: SsaVarId,
        /// Index of the block containing the variable definition.
        def_block: usize,
        /// Index of the block containing the variable use.
        use_block: usize,
    },
    /// A terminator instruction is not the last instruction in its block.
    TerminatorNotLast {
        /// Index of the block with a misplaced terminator.
        block: usize,
        /// Index of the terminator instruction.
        instr_idx: usize,
        /// Total number of instructions in the block.
        instr_count: usize,
    },
    /// An instruction uses a variable defined later in the same block (cycle).
    IntraBlockCycle {
        /// Index of the block containing both instructions.
        block: usize,
        /// Index of the instruction that reads too early.
        use_instr: usize,
        /// Index of the later instruction that defines the variable.
        def_instr: usize,
        /// Variable read before its in-block definition.
        var: SsaVarId,
    },
    /// The variables table is not densely indexed by variable id.
    ///
    /// [`SsaFunction::variable`] is a raw index into the table, so the whole IR
    /// depends on `variables[i].id().index() == i`. When a transform drops a row
    /// without renumbering, every id above the hole silently resolves to a
    /// *different* variable's def site, type, and use list — corruption that no
    /// other check detects, because the resulting IR is structurally well-formed.
    VariableTableNotDense {
        /// Position in the variables table.
        slot: usize,
        /// Id actually stored at that position.
        found: SsaVarId,
    },
    /// A placeholder variable ID (usize::MAX) remains in finalized SSA.
    PlaceholderVariable {
        /// Index of the block containing the placeholder.
        block: usize,
        /// Human-readable location within the block, such as a phi or instruction operand.
        location: String,
    },
    /// An instruction's destination appears in its own operands (self-referential).
    SelfReferentialInstruction {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the self-referential instruction.
        instr_idx: usize,
        /// Destination variable that also appears as an operand.
        var: SsaVarId,
    },
    /// A vector operation has incompatible or unsupported vector shapes.
    InvalidVectorOperation {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid vector instruction.
        instr_idx: usize,
        /// Human-readable description of the shape violation.
        reason: String,
    },
    /// A native atomic operation has illegal ordering, width, or outputs.
    InvalidAtomicOperation {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid atomic instruction.
        instr_idx: usize,
        /// Human-readable description of the atomic violation.
        reason: String,
    },
    /// A wide arithmetic operation has incompatible operand or result widths.
    InvalidWideArithmetic {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid wide arithmetic instruction.
        instr_idx: usize,
        /// Human-readable description of the width violation.
        reason: String,
    },
    /// A native opaque operation or effect summary is malformed.
    InvalidNativeOperation {
        /// Index of the block containing the instruction.
        block: usize,
        /// Index of the invalid native instruction.
        instr_idx: usize,
        /// Human-readable description of the native/effect violation.
        reason: String,
    },
}

impl std::fmt::Display for VerifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndefinedUse {
                block,
                instr_idx,
                var,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} uses undefined variable {var:?}"
            ),
            Self::MissingPhiOperand {
                block,
                phi_idx,
                missing_pred,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} missing operand for predecessor {missing_pred}"
            ),
            Self::ExtraPhiOperand {
                block,
                phi_idx,
                extra_pred,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} has operand for non-predecessor {extra_pred}"
            ),
            Self::DuplicateDefinition { var, def1, def2 } => write!(
                f,
                "Variable {var:?} defined twice: at block {} ({:?}) and block {} ({:?})",
                def1.block, def1.kind, def2.block, def2.kind
            ),
            Self::OrphanVariable { var } => {
                write!(
                    f,
                    "Variable {var:?} in variables vec but not defined in any block"
                )
            }
            Self::UnregisteredVariable { var } => write!(
                f,
                "Variable {var:?} used in instruction but not in variables vec"
            ),
            Self::MissingTerminator { block } => {
                write!(f, "Block {block}: has successors but no terminator")
            }
            Self::PhiInEntryBlock { block, phi_idx } => {
                write!(f, "Block {block}: phi {phi_idx} in entry block")
            }
            Self::DominanceViolation {
                var,
                def_block,
                use_block,
            } => write!(
                f,
                "Variable {var:?}: def in block {def_block} does not dominate use in block {use_block}"
            ),
            Self::TerminatorNotLast {
                block,
                instr_idx,
                instr_count,
            } => write!(
                f,
                "Block {block}: terminator at position {instr_idx}/{instr_count} is not last"
            ),
            Self::IntraBlockCycle {
                block,
                use_instr,
                def_instr,
                var,
            } => write!(
                f,
                "Block {block}: instruction {use_instr} uses {var:?} defined at instruction {def_instr}"
            ),
            Self::DuplicatePhiOperand {
                block,
                phi_idx,
                pred,
            } => write!(
                f,
                "Block {block}: phi {phi_idx} has multiple operands for predecessor {pred}"
            ),
            Self::VariableTableNotDense { slot, found } => write!(
                f,
                "Variable table slot {slot} holds {found:?}; `variable(id)` indexes this table \
                 directly, so every id at or above {slot} now resolves to the wrong variable"
            ),
            Self::PlaceholderVariable { block, location } => write!(
                f,
                "Block {block}: placeholder variable ID (usize::MAX) at {location}"
            ),
            Self::SelfReferentialInstruction {
                block,
                instr_idx,
                var,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has self-referential use of {var:?}"
            ),
            Self::InvalidVectorOperation {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid vector op: {reason}"
            ),
            Self::InvalidAtomicOperation {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid atomic op: {reason}"
            ),
            Self::InvalidWideArithmetic {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid wide arithmetic op: {reason}"
            ),
            Self::InvalidNativeOperation {
                block,
                instr_idx,
                reason,
            } => write!(
                f,
                "Block {block}: instruction {instr_idx} has invalid native op: {reason}"
            ),
        }
    }
}

impl std::error::Error for VerifierError {}

/// Returns a conservative ordering strength rank for compare-exchange checks.
const fn ordering_rank(ordering: AtomicOrdering) -> u8 {
    match ordering {
        AtomicOrdering::Relaxed => 0,
        AtomicOrdering::Acquire => 1,
        AtomicOrdering::Release => 1,
        AtomicOrdering::AcqRel => 2,
        AtomicOrdering::SeqCst => 3,
    }
}

/// Verification depth levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerifyLevel {
    /// Single-definition + block structure checks (O(n)).
    Quick,
    /// + def-use chains + phi operand coverage (O(n*m)).
    Standard,
    /// + dominance checking (O(n^2) worst case).
    Full,
}

/// SSA verifier that validates invariants at configurable depth.
///
/// The verifier checks SSA form correctness at three levels (Quick, Standard, Full)
/// and reports all violations as `VerifierError` values. It is the safety net for
/// all SSA transformations, catching invariant violations early.
pub struct SsaVerifier<'a, T: Target> {
    /// Reference to the SSA function being verified.
    ssa: &'a SsaFunction<T>,
    /// Accumulated list of errors found during verification.
    errors: Vec<VerifierError>,
    /// Exact capacity covering every variable id the function mentions,
    /// computed once and shared by the three variable-indexed structures below.
    ///
    /// Deliberately [`SsaFunction::var_id_capacity`] and not the O(1)
    /// [`SsaFunction::var_id_bound`]: the passes may assume ids come from the
    /// allocator, but the verifier is what *establishes* that, so it has to
    /// index ids outside the allocator's range rather than skip them — skipping
    /// would make it silently fail to report the malformed IR it exists to
    /// catch. The scan is O(function), which verification already is, and doing
    /// it once here instead of per structure keeps it to a single extra walk.
    var_capacity: usize,
}

impl<'a, T: Target> SsaVerifier<'a, T> {
    /// Creates a new verifier for the given SSA function.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>) -> Self {
        Self {
            ssa,
            errors: Vec::new(),
            var_capacity: ssa.var_id_capacity(),
        }
    }

    /// Runs verification at the specified level and returns all errors found.
    pub fn verify(mut self, level: VerifyLevel) -> Vec<VerifierError> {
        self.errors.clear();

        // Quick checks (always run)
        self.check_variable_table_density();
        self.check_single_definition();
        self.check_block_structure();
        self.check_no_placeholders_or_self_refs();

        if level >= VerifyLevel::Standard {
            let cfg = SsaCfg::from_ssa(self.ssa);
            let definitions = self.collect_definitions();
            self.check_phi_operands(&cfg);
            self.check_defined_before_use(&definitions);
            self.check_registered_variables();
            self.check_vector_operations();
            self.check_atomic_operations();
            self.check_wide_arithmetic();
            self.check_native_effects();

            if level >= VerifyLevel::Full {
                let dom_tree = compute_dominators(&cfg, cfg.entry());
                self.check_dominance(&cfg, &dom_tree, &definitions);
            }
        }

        self.errors
    }

    /// Verifies the variables table is densely indexed by variable id.
    ///
    /// [`SsaFunction::variable`] and `variable_mut` are raw
    /// `variables.get(id.index())`, so `variables[i].id().index() == i` is a
    /// load-bearing invariant rather than a tidiness one. Removing a row without
    /// renumbering shifts every id above it, and the reads that go wrong —
    /// `def_site()` for dominance decisions, `var_type()` for type
    /// compatibility — are exactly the ones passes consult before rewriting.
    ///
    /// This is a Quick check because it is a single linear scan and because
    /// nothing else detects the violation: the resulting IR is well-formed in
    /// every other respect, merely describing different variables than intended.
    fn check_variable_table_density(&mut self) {
        for (slot, var) in self.ssa.variables().iter().enumerate() {
            if var.id().index() != slot {
                self.errors.push(VerifierError::VariableTableNotDense {
                    slot,
                    found: var.id(),
                });
                // One report is enough: everything after the first hole is
                // shifted too, and listing all of it buries the cause.
                break;
            }
        }
    }

    /// Verifies that every variable is defined at most once (the fundamental SSA property).
    fn check_single_definition(&mut self) {
        let mut definitions: VarMap<VerifierDefSite> = VarMap::new(self.var_capacity);

        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                let var = phi.result();
                let site = VerifierDefSite {
                    block: block_idx,
                    kind: DefKind::Phi(phi_idx),
                };
                if let Some(prev) = definitions.get(var) {
                    self.errors.push(VerifierError::DuplicateDefinition {
                        var,
                        def1: prev.clone(),
                        def2: site,
                    });
                } else {
                    definitions.insert(var, site);
                }
            }

            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                for dest in instr.op().defs() {
                    let site = VerifierDefSite {
                        block: block_idx,
                        kind: DefKind::Instruction(instr_idx),
                    };
                    if let Some(prev) = definitions.get(dest) {
                        self.errors.push(VerifierError::DuplicateDefinition {
                            var: dest,
                            def1: prev.clone(),
                            def2: site,
                        });
                    } else {
                        definitions.insert(dest, site);
                    }
                }
            }
        }
    }

    /// Checks block structural invariants:
    /// - Terminators are the last instruction
    /// - No intra-block cycles (use before def)
    ///
    /// This deliberately does *not* check "every block with successors has a
    /// terminator", and [`VerifierError::MissingTerminator`] is consequently
    /// never constructed: [`SsaBlock::successors`](crate::ir::block::SsaBlock::successors)
    /// derives the successor set from the block's *last instruction*, so a block
    /// can only have successors when that instruction is a control-flow op. The
    /// condition is unsatisfiable rather than unchecked.
    ///
    /// Detecting the related-but-different defect — a non-empty block whose last
    /// instruction is not a terminator, so control falls off the end — would be a
    /// new invariant, not a repair of this one.
    fn check_block_structure(&mut self) {
        // One `(block stamp, instruction index)` slot per variable id, reused
        // across every block rather than reallocated per block — this ran
        // 27,384 times per verify on the largest function of a real binary.
        // `usize::MAX` is a block index no function has, so an untouched slot
        // never matches the current block, which avoids an O(variables x
        // blocks) clear.
        let mut def_indices: Vec<(usize, usize)> = vec![(usize::MAX, 0); self.var_capacity];

        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            let instrs = block.instructions();
            let instr_count = instrs.len();

            // Check terminator placement
            for (instr_idx, instr) in instrs.iter().enumerate() {
                if instr.op().is_terminator() && instr_idx < instr_count.saturating_sub(1) {
                    self.errors.push(VerifierError::TerminatorNotLast {
                        block: block_idx,
                        instr_idx,
                        instr_count,
                    });
                }
            }

            // Check for intra-block use-before-def cycles.
            //
            // `def_indices` is stamped with the block index rather than cleared:
            // a slot whose stamp is not the current block is stale. Clearing a
            // capacity-sized buffer per block would be O(variables x blocks),
            // and allocating a fresh map per block — which this did — costs one
            // allocation per block, 23,744 of them on the largest function of a
            // real binary, on every verify.
            for (instr_idx, instr) in instrs.iter().enumerate() {
                for dest in instr.op().defs() {
                    if dest.is_placeholder() {
                        continue;
                    }
                    if let Some(slot) = def_indices.get_mut(dest.index()) {
                        *slot = (block_idx, instr_idx);
                    }
                }
            }

            for (instr_idx, instr) in instrs.iter().enumerate() {
                instr.op().for_each_use(|used_var| {
                    if let Some(&(stamp, def_idx)) = def_indices.get(used_var.index())
                        && !used_var.is_placeholder()
                        && stamp == block_idx
                        && def_idx >= instr_idx
                    {
                        self.errors.push(VerifierError::IntraBlockCycle {
                            block: block_idx,
                            use_instr: instr_idx,
                            def_instr: def_idx,
                            var: used_var,
                        });
                    }
                });
            }
        }
    }

    /// Collects all variable definitions into a map: var_id -> (block, def_site).
    fn collect_definitions(&self) -> VarMap<(usize, DefSite)> {
        let mut defs: VarMap<(usize, DefSite)> = VarMap::new(self.var_capacity);

        // Variables from the variables vec (includes entry-block defs for args/locals)
        for var in self.ssa.variables() {
            defs.insert(var.id(), (var.def_site().block, var.def_site()));
        }

        // Also collect from actual block contents (may differ after transforms)
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for phi in block.phi_nodes() {
                defs.or_insert(phi.result(), (block_idx, DefSite::phi(block_idx)));
            }
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                for dest in instr.op().defs() {
                    defs.or_insert(
                        dest,
                        (block_idx, DefSite::instruction(block_idx, instr_idx)),
                    );
                }
            }
        }

        defs
    }

    /// Checks that every phi node has the correct operand set:
    /// - One operand per CFG predecessor
    /// - No operands from non-predecessor blocks
    /// - No phis in the entry block (which has no predecessors)
    fn check_phi_operands(&mut self, cfg: &SsaCfg<'_, T>) {
        let block_count = self.ssa.block_count();
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            let pred_list = cfg.block_predecessors(block_idx);
            // Capacity must cover both actual predecessors and phi operand predecessors
            // (which may reference non-existent blocks in malformed SSA)
            let max_phi_pred = block
                .phi_nodes()
                .iter()
                .flat_map(|phi| phi.operands().iter().map(|op| op.predecessor()))
                .max()
                .unwrap_or(0);
            let capacity = block_count.max(max_phi_pred.saturating_add(1)).max(1);
            let mut preds = BitSet::new(capacity);
            for &p in pred_list {
                if p < capacity {
                    preds.insert(p);
                }
            }

            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                // Entry block should not have phis (no predecessors)
                if block_idx == 0 && preds.is_empty() {
                    self.errors.push(VerifierError::PhiInEntryBlock {
                        block: block_idx,
                        phi_idx,
                    });
                    continue;
                }

                // Counted, not set-tested: two operands naming one predecessor
                // collapse to a single bit, so a set comparison cannot see the
                // duplicate and the advertised "exactly one operand per
                // predecessor" invariant goes unchecked. `PhiNode::add_operand`
                // does not dedup and `PhiOperand::set_predecessor` allows a
                // retarget onto an edge that already has an operand, which is
                // what edge redirection during block merging does.
                let mut operand_preds = BitSet::new(capacity);
                let mut seen_twice = BitSet::new(capacity);
                for op in phi.operands() {
                    let pred = op.predecessor();
                    if !operand_preds.insert(pred) && seen_twice.insert(pred) {
                        self.errors.push(VerifierError::DuplicatePhiOperand {
                            block: block_idx,
                            phi_idx,
                            pred,
                        });
                    }
                }

                // Check for missing predecessors
                for pred in preds.iter() {
                    if !operand_preds.contains(pred) {
                        self.errors.push(VerifierError::MissingPhiOperand {
                            block: block_idx,
                            phi_idx,
                            missing_pred: pred,
                        });
                    }
                }

                // Check for extra (non-predecessor) operands
                for op_pred in operand_preds.iter() {
                    if !preds.contains(op_pred) {
                        self.errors.push(VerifierError::ExtraPhiOperand {
                            block: block_idx,
                            phi_idx,
                            extra_pred: op_pred,
                        });
                    }
                }
            }
        }
    }

    /// Checks that every variable used in an instruction or phi operand is defined
    /// somewhere (either in the variables vec or in a block).
    fn check_defined_before_use(&mut self, definitions: &VarMap<(usize, DefSite)>) {
        let runtime_supplied = self.runtime_supplied_vars();
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                instr.op().for_each_use(|used_var| {
                    if !definitions.contains(used_var)
                        && !runtime_supplied.contains_checked(used_var.index())
                    {
                        self.errors.push(VerifierError::UndefinedUse {
                            block: block_idx,
                            instr_idx,
                            var: used_var,
                        });
                    }
                });
            }
        }
    }

    /// Variables the runtime supplies rather than any instruction: the exception
    /// object a catch or filter handler receives on entry.
    ///
    /// The handler's first instruction is a [`SsaOp::Pop`] consuming an object
    /// the runtime pushed, so there is no definition to find and none should be
    /// demanded. [`SsaRebuilder`] already carves out exactly this shape — it
    /// preserves such a `Pop` where it drops every other one whose operand has
    /// no definition — and without the matching carve-out here the verifier
    /// rejects the IR the rebuilder is built to produce, failing every method
    /// with a handler as soon as a CFG-modifying pass triggers a rebuild.
    ///
    /// The `Pop` at a handler entry is what *identifies* the variable; once
    /// identified, it is exempt as a definition would be, since that is exactly
    /// what the runtime provides. Reading it outside the handler is a dominance
    /// violation, which [`Self::check_dominance`] reports on its own. Every
    /// other undefined read — including one in the same block that no
    /// handler-entry `Pop` consumes — stays reportable.
    ///
    /// [`SsaRebuilder`]: crate::ir::function::rebuild::SsaRebuilder
    fn runtime_supplied_vars(&self) -> BitSet {
        let mut vars = BitSet::lazy(self.var_capacity);
        let block_count = self.ssa.blocks().len();
        let mut entry_blocks = BitSet::new(block_count.max(1));
        for handler in self.ssa.exception_handlers() {
            for block in [handler.handler_start_block, handler.filter_start_block]
                .into_iter()
                .flatten()
            {
                entry_blocks.insert_checked(block);
            }
        }
        if entry_blocks.count() == 0 {
            return vars;
        }
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            if !entry_blocks.contains_checked(block_idx) {
                continue;
            }
            for instr in block.instructions() {
                if let SsaOp::Pop { value } = instr.op() {
                    vars.insert_checked(value.index());
                }
            }
        }
        vars
    }

    /// Checks that every variable used in blocks is registered in the variables vec.
    fn check_registered_variables(&mut self) {
        let variable_count = self.ssa.variable_count();
        // Capacity must cover all variable IDs that appear in blocks (may exceed variable_count)
        let max_block_var = self
            .ssa
            .blocks()
            .iter()
            .flat_map(|b| {
                let phi_ids = b.phi_nodes().iter().map(|p| p.result().index());
                let instr_ids = b
                    .instructions()
                    .iter()
                    .flat_map(|i| i.op().defs().map(|d| d.index()));
                phi_ids.chain(instr_ids)
            })
            .max()
            .unwrap_or(0);
        let max_reg_var = self
            .ssa
            .variables()
            .iter()
            .map(|v| v.id().index())
            .max()
            .unwrap_or(0);
        let capacity = max_block_var
            .saturating_add(1)
            .max(max_reg_var.saturating_add(1))
            .max(variable_count)
            .max(1);
        let mut registered = BitSet::new(capacity);
        for v in self.ssa.variables() {
            registered.insert(v.id().index());
        }

        // Check variables defined in blocks but not in variables vec
        for block in self.ssa.blocks() {
            for phi in block.phi_nodes() {
                let idx = phi.result().index();
                if idx >= capacity || !registered.contains(idx) {
                    self.errors
                        .push(VerifierError::UnregisteredVariable { var: phi.result() });
                }
            }
            for instr in block.instructions() {
                for dest in instr.op().defs() {
                    let idx = dest.index();
                    if idx >= capacity || !registered.contains(idx) {
                        self.errors
                            .push(VerifierError::UnregisteredVariable { var: dest });
                    }
                }
            }
        }

        // Check for orphan variables (in variables vec but not defined in any block)
        let mut block_defined = BitSet::new(capacity);
        for block in self.ssa.blocks() {
            for phi in block.phi_nodes() {
                let idx = phi.result().index();
                if idx < capacity {
                    block_defined.insert(idx);
                }
            }
            for instr in block.instructions() {
                for dest in instr.op().defs() {
                    let idx = dest.index();
                    if idx < capacity {
                        block_defined.insert(idx);
                    }
                }
            }
        }

        // Where each variable is first read, if anywhere. An undefined variable
        // that is also unread is dead weight for `compact_variables` to collect;
        // an undefined variable with a live *use* is a dangling read, and the two
        // deserve different errors because consumers treat them differently
        // (`SsaRebuilder::verify` filters the former as cosmetic).
        let mut first_use: Vec<Option<(usize, usize)>> = vec![None; capacity];
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                instr.op().for_each_use(|var| {
                    if let Some(slot) = first_use.get_mut(var.index())
                        && slot.is_none()
                    {
                        *slot = Some((block_idx, instr_idx));
                    }
                });
            }
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    if let Some(slot) = first_use.get_mut(operand.value().index())
                        && slot.is_none()
                    {
                        *slot = Some((block_idx, 0));
                    }
                }
            }
        }

        let runtime_supplied = self.runtime_supplied_vars();
        for var in self.ssa.variables() {
            if block_defined.contains(var.id().index())
                || runtime_supplied.contains_checked(var.id().index())
            {
                continue;
            }
            match first_use.get(var.id().index()).copied().flatten() {
                // Something reads it. Only a genuine entry definition excuses
                // that — see `is_entry_defined` for why the obvious weaker test
                // does not. Reported as `UndefinedUse` rather than
                // `OrphanVariable`: it names the reading instruction, and it is
                // not the class `SsaRebuilder::verify` discards.
                Some((block, instr_idx)) if !self.is_entry_defined(var) => {
                    self.errors.push(VerifierError::UndefinedUse {
                        block,
                        instr_idx,
                        var: var.id(),
                    });
                }
                Some(_) => {}
                // Nothing reads it: a leftover row. The weaker test is right
                // here — tightening it would report every temporary a pass
                // legitimately killed.
                None => {
                    if !(var.version() == 0 && var.def_site().instruction.is_none()) {
                        self.errors
                            .push(VerifierError::OrphanVariable { var: var.id() });
                    }
                }
            }
        }
    }

    /// Returns `true` when `var` is defined at function entry rather than by any
    /// instruction or phi — arguments, default-initialized locals, and the
    /// phi-origin placeholders the rebuilder creates for stack temp groups.
    ///
    /// This test is deliberately narrow, and the narrowness is the point. The
    /// obvious formulation — "version 0 and no defining instruction" — also
    /// matches a variable whose defining instruction was *destroyed*:
    /// [`refresh_def_sites`](SsaFunction::repair_ssa) resets such a variable to
    /// [`DefSite::entry`], leaving it indistinguishable from a real argument. A
    /// pass that nops a definition while a use survives would then produce IR
    /// that verifies clean, which is exactly how a dangling `flags` use escaped
    /// detection.
    ///
    /// Requiring the origin to name a real argument or local closes that: a
    /// temporary whose definition was removed does not.
    fn is_entry_defined(&self, var: &SsaVariable<T>) -> bool {
        if var.version() != 0 || var.def_site().instruction.is_some() {
            return false;
        }
        match var.origin() {
            // An argument origin is an explicit statement that the caller
            // supplies this value, so it needs no definition in the body. (The
            // index is not bounds-checked against `num_args`: a mismatch there is
            // a signature inconsistency, not a dangling read, and reporting it as
            // an orphan variable would be misleading.)
            VariableOrigin::Argument(_) => true,
            // A declared local is default-initialized at entry. An index beyond
            // the declared count is not a real local — it is a temporary, and a
            // temporary with no definition left is the case worth catching.
            VariableOrigin::Local(index) => usize::from(index) < self.ssa.num_locals(),
            // A phi-origin variable is defined by a phi node, full stop. The
            // rebuilder does create placeholder reaching-defs with this origin
            // for stack temp groups, and those legitimately have no definition
            // *while it is running* — but they are resolved before it verifies,
            // so exempting them here would only widen the hole. Nothing in the
            // suite depends on the exemption.
            VariableOrigin::Phi => false,
            // A machine-code front end's function-entry live-in: the caller
            // supplies it, so like an argument it needs no definition in the
            // body. This is the origin such a lifter must use — `Phi` above is
            // rejected precisely so a destroyed definition cannot masquerade as
            // an entry value, and `Argument` would assert a signature position
            // the lifter has not recovered.
            VariableOrigin::EntryLiveIn => true,
        }
    }

    /// Checks for placeholder variable IDs (usize::MAX) that should have been
    /// replaced during construction. Also checks for self-referential instructions
    /// where an instruction's destination appears in its own operands.
    fn check_no_placeholders_or_self_refs(&mut self) {
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            // Check phi nodes for placeholder IDs
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                if phi.result().is_placeholder() {
                    self.errors.push(VerifierError::PlaceholderVariable {
                        block: block_idx,
                        location: format!("phi {phi_idx} result"),
                    });
                }
                for operand in phi.operands() {
                    if operand.value().is_placeholder() {
                        self.errors.push(VerifierError::PlaceholderVariable {
                            block: block_idx,
                            location: format!(
                                "phi {phi_idx} operand from B{}",
                                operand.predecessor()
                            ),
                        });
                    }
                }
            }

            // Check instructions for placeholder IDs and self-referential uses
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let op = instr.op();
                for dest in op.defs() {
                    if dest.is_placeholder() {
                        self.errors.push(VerifierError::PlaceholderVariable {
                            block: block_idx,
                            location: format!("instruction {instr_idx} dest"),
                        });
                    }
                    // Check for self-referential instruction (dest appears in uses)
                    if op.uses_var(dest) {
                        self.errors.push(VerifierError::SelfReferentialInstruction {
                            block: block_idx,
                            instr_idx,
                            var: dest,
                        });
                    }
                }
                op.for_each_use(|used_var| {
                    if used_var.is_placeholder() {
                        self.errors.push(VerifierError::PlaceholderVariable {
                            block: block_idx,
                            location: format!("instruction {instr_idx} operand"),
                        });
                    }
                });
            }
        }
    }

    /// Checks vector operation shape consistency when operand types are known.
    fn check_vector_operations(&mut self) {
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let op = instr.op();
                match op {
                    SsaOp::VectorBinary {
                        dest, left, right, ..
                    }
                    | SsaOp::VectorCompare {
                        dest, left, right, ..
                    } => {
                        let left_shape = self.var_vector_shape(*left);
                        let right_shape = self.var_vector_shape(*right);
                        if let (Some(left_shape), Some(right_shape)) = (left_shape, right_shape)
                            && left_shape != right_shape
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "operand vector shapes differ",
                            );
                        }
                        if matches!(op, SsaOp::VectorBinary { .. }) {
                            self.check_dest_shape(block_idx, instr_idx, *dest, left_shape);
                        } else if let Some(shape) = left_shape {
                            self.check_mask_dest_shape(
                                block_idx,
                                instr_idx,
                                *dest,
                                shape.mask_descriptor(),
                            );
                        }
                    }
                    SsaOp::VectorUnary { dest, value, .. } => {
                        self.check_dest_shape(
                            block_idx,
                            instr_idx,
                            *dest,
                            self.var_vector_shape(*value),
                        );
                    }
                    SsaOp::VectorTernary {
                        dest,
                        first,
                        second,
                        third,
                        ..
                    } => {
                        let a = self.var_vector_shape(*first);
                        let b = self.var_vector_shape(*second);
                        let c = self.var_vector_shape(*third);
                        if let (Some(a), Some(b), Some(c)) = (a, b, c)
                            && (a != b || a != c)
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "ternary operand vector shapes differ",
                            );
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, a);
                    }
                    SsaOp::VectorLoad {
                        dest, vector_type, ..
                    }
                    | SsaOp::VectorBroadcastLoad {
                        dest, vector_type, ..
                    } => {
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "load vector_type is not a supported vector",
                            );
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, shape);
                    }
                    SsaOp::VectorMaskedLoad {
                        dest,
                        mask,
                        passthrough,
                        vector_type,
                        ..
                    }
                    | SsaOp::VectorGather {
                        dest,
                        mask,
                        passthrough,
                        vector_type,
                        ..
                    } => {
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "masked load vector_type is not a supported vector",
                            );
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, shape);
                        if let Some(shape) = shape {
                            self.check_mask_source_shape(
                                block_idx,
                                instr_idx,
                                *mask,
                                shape.mask_descriptor(),
                            );
                            if let Some(passthrough) = passthrough {
                                self.check_source_shape(
                                    block_idx,
                                    instr_idx,
                                    *passthrough,
                                    Some(shape),
                                );
                            }
                        }
                    }
                    SsaOp::VectorFaultingLoad {
                        dest,
                        fault,
                        mask,
                        passthrough,
                        vector_type,
                        ..
                    } => {
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "faulting load vector_type is not a supported vector",
                            );
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, shape);
                        if let Some(shape) = shape {
                            let mask_shape = shape.mask_descriptor();
                            if let Some(mask) = mask {
                                self.check_mask_source_shape(
                                    block_idx, instr_idx, *mask, mask_shape,
                                );
                            }
                            if let Some(fault) = fault {
                                self.check_mask_dest_shape(
                                    block_idx, instr_idx, *fault, mask_shape,
                                );
                            }
                            if let Some(passthrough) = passthrough {
                                self.check_source_shape(
                                    block_idx,
                                    instr_idx,
                                    *passthrough,
                                    Some(shape),
                                );
                            }
                        }
                    }
                    SsaOp::VectorSegmentLoad {
                        dests,
                        mask,
                        vector_type,
                        segments,
                        ..
                    } => {
                        if *segments == 0 {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "segment load must load at least one segment",
                            );
                        }
                        if usize::try_from(*segments).ok() != Some(dests.len()) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "segment load destination count does not match segments",
                            );
                        }
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "segment load vector_type is not a supported vector",
                            );
                        }
                        for dest in dests {
                            self.check_dest_shape(block_idx, instr_idx, *dest, shape);
                        }
                        if let (Some(mask), Some(shape)) = (mask, shape) {
                            self.check_mask_source_shape(
                                block_idx,
                                instr_idx,
                                *mask,
                                shape.mask_descriptor(),
                            );
                        }
                    }
                    SsaOp::VectorStore {
                        value, vector_type, ..
                    } => {
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "store vector_type is not a supported vector",
                            );
                        }
                        self.check_source_shape(block_idx, instr_idx, *value, shape);
                    }
                    SsaOp::VectorMaskedStore {
                        value,
                        mask,
                        vector_type,
                        ..
                    }
                    | SsaOp::VectorScatter {
                        value,
                        mask,
                        vector_type,
                        ..
                    } => {
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "masked store vector_type is not a supported vector",
                            );
                        }
                        self.check_source_shape(block_idx, instr_idx, *value, shape);
                        if let Some(shape) = shape {
                            self.check_mask_source_shape(
                                block_idx,
                                instr_idx,
                                *mask,
                                shape.mask_descriptor(),
                            );
                        }
                    }
                    SsaOp::VectorSegmentStore {
                        values,
                        mask,
                        vector_type,
                        segments,
                        ..
                    } => {
                        if *segments == 0 {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "segment store must store at least one segment",
                            );
                        }
                        if usize::try_from(*segments).ok() != Some(values.len()) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "segment store value count does not match segments",
                            );
                        }
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "segment store vector_type is not a supported vector",
                            );
                        }
                        for value in values {
                            self.check_source_shape(block_idx, instr_idx, *value, shape);
                        }
                        if let (Some(mask), Some(shape)) = (mask, shape) {
                            self.check_mask_source_shape(
                                block_idx,
                                instr_idx,
                                *mask,
                                shape.mask_descriptor(),
                            );
                        }
                    }
                    SsaOp::VectorExtract { dest, vector, lane } => {
                        if let Some(shape) = self.var_vector_shape(*vector) {
                            if shape
                                .fixed_lane_count()
                                .is_some_and(|lane_count| *lane >= lane_count)
                            {
                                self.invalid_vector(
                                    block_idx,
                                    instr_idx,
                                    "extract lane is out of bounds",
                                );
                            }
                            if let Some(expected) = T::vector_descriptor_lane_type(shape) {
                                self.check_var_type(
                                    block_idx,
                                    instr_idx,
                                    *dest,
                                    &expected,
                                    "extract destination type does not match lane type",
                                );
                            }
                        }
                    }
                    SsaOp::VectorInsert {
                        dest,
                        vector,
                        lane,
                        value,
                    } => {
                        let shape = self.var_vector_shape(*vector);
                        if let Some(shape) = shape {
                            if shape
                                .fixed_lane_count()
                                .is_some_and(|lane_count| *lane >= lane_count)
                            {
                                self.invalid_vector(
                                    block_idx,
                                    instr_idx,
                                    "insert lane is out of bounds",
                                );
                            }
                            if let Some(expected) = T::vector_descriptor_lane_type(shape) {
                                self.check_var_type(
                                    block_idx,
                                    instr_idx,
                                    *value,
                                    &expected,
                                    "insert value type does not match lane type",
                                );
                            }
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, shape);
                    }
                    SsaOp::VectorSplat {
                        dest,
                        value,
                        vector_type,
                    } => {
                        let shape = T::vector_descriptor(vector_type);
                        if shape.is_none() && !T::is_unknown(vector_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "splat vector_type is not a supported vector",
                            );
                        }
                        if let Some(shape) = shape
                            && let Some(expected) = T::vector_descriptor_lane_type(shape)
                        {
                            self.check_var_type(
                                block_idx,
                                instr_idx,
                                *value,
                                &expected,
                                "splat value type does not match lane type",
                            );
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, shape);
                    }
                    SsaOp::VectorShuffle {
                        dest,
                        left,
                        right,
                        mask,
                    } => {
                        let left_shape = self.var_vector_shape(*left);
                        let right_shape = right.and_then(|v| self.var_vector_shape(v));
                        if let Some(left_shape) = left_shape {
                            if !mask.is_valid_for(
                                left_shape.min_lane_count(),
                                right_shape.map(VectorDescriptor::min_lane_count),
                            ) {
                                self.invalid_vector(
                                    block_idx,
                                    instr_idx,
                                    "shuffle mask selects an invalid lane",
                                );
                            }
                            if let Some(dest_shape) = self.var_vector_shape(*dest) {
                                if dest_shape.fixed_lane_count().is_some_and(|lane_count| {
                                    lane_count != mask.lanes().len() as u32
                                }) {
                                    self.invalid_vector(
                                        block_idx,
                                        instr_idx,
                                        "shuffle destination lane count does not match mask",
                                    );
                                }
                                if dest_shape.lane_bits() != left_shape.lane_bits()
                                    || dest_shape.lane_kind() != left_shape.lane_kind()
                                {
                                    self.invalid_vector(
                                        block_idx,
                                        instr_idx,
                                        "shuffle destination lane shape differs from input",
                                    );
                                }
                            }
                        }
                    }
                    SsaOp::VectorCast {
                        dest, target_type, ..
                    } => {
                        let target_shape = T::vector_descriptor(target_type);
                        if target_shape.is_none() && !T::is_unknown(target_type) {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "cast target_type is not a supported vector",
                            );
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, target_shape);
                    }
                    SsaOp::VectorReinterpret {
                        dest,
                        value,
                        target_type,
                    } => {
                        let source_shape = self.var_vector_shape(*value);
                        let target_shape = T::vector_descriptor(target_type);
                        if let (Some(source), Some(target)) = (source_shape, target_shape)
                            && source.total_bits().is_some()
                            && target.total_bits().is_some()
                            && source.total_bits() != target.total_bits()
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "reinterpret source and target widths differ",
                            );
                        }
                        self.check_dest_shape(block_idx, instr_idx, *dest, target_shape);
                    }
                    SsaOp::VectorMaskUnary { dest, mask, .. } => {
                        if let Some(shape) = self.var_mask_shape(*mask) {
                            self.check_mask_dest_shape(block_idx, instr_idx, *dest, shape);
                        }
                    }
                    SsaOp::VectorMaskBinary {
                        dest, left, right, ..
                    } => {
                        let left_shape = self.var_mask_shape(*left);
                        let right_shape = self.var_mask_shape(*right);
                        if let (Some(left_shape), Some(right_shape)) = (left_shape, right_shape)
                            && left_shape != right_shape
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "mask operand lane counts differ",
                            );
                        }
                        if let Some(shape) = left_shape {
                            self.check_mask_dest_shape(block_idx, instr_idx, *dest, shape);
                        }
                    }
                    SsaOp::VectorReduce { dest, value, .. } => {
                        if let Some(shape) = self.var_vector_shape(*value)
                            && let Some(expected) = T::vector_descriptor_lane_type(shape)
                        {
                            self.check_var_type(
                                block_idx,
                                instr_idx,
                                *dest,
                                &expected,
                                "reduction destination type does not match lane type",
                            );
                        }
                    }
                    SsaOp::VectorBitmask { dest, value, .. } => {
                        if let Some(ty) = self.var_type(*dest)
                            && !T::is_unknown(ty)
                            && !T::is_integer(ty)
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "bitmask destination must be an integer scalar",
                            );
                        }
                        if self.var_vector_shape(*value).is_none()
                            && self.var_mask_shape(*value).is_none()
                            && let Some(ty) = self.var_type(*value)
                            && !T::is_unknown(ty)
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "bitmask source must be a vector or mask",
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Checks native atomic ordering, width, and output-shape constraints.
    fn check_atomic_operations(&mut self) {
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                match instr.op() {
                    SsaOp::AtomicExchange {
                        dest,
                        addr,
                        value,
                        ordering,
                        width,
                        ..
                    }
                    | SsaOp::AtomicLockRmw {
                        dest,
                        addr,
                        value,
                        ordering,
                        width,
                        ..
                    } => {
                        self.check_atomic_address(block_idx, instr_idx, *addr);
                        self.check_atomic_width(block_idx, instr_idx, *dest, *width);
                        self.check_atomic_width(block_idx, instr_idx, *value, *width);
                        if matches!(ordering, AtomicOrdering::Release) {
                            self.invalid_atomic(
                                block_idx,
                                instr_idx,
                                "read-modify-write ordering cannot be release-only",
                            );
                        }
                    }
                    SsaOp::AtomicCmpXchg {
                        old,
                        success,
                        addr,
                        expected,
                        desired,
                        success_ordering,
                        failure_ordering,
                        width,
                        ..
                    } => {
                        self.check_atomic_address(block_idx, instr_idx, *addr);
                        self.check_atomic_width(block_idx, instr_idx, *old, *width);
                        self.check_atomic_width(block_idx, instr_idx, *expected, *width);
                        self.check_atomic_width(block_idx, instr_idx, *desired, *width);
                        if let Some(success) = success
                            && let Some(ty) = self.var_type(*success)
                            && !T::is_unknown(ty)
                            && !T::is_integer(ty)
                        {
                            self.invalid_atomic(
                                block_idx,
                                instr_idx,
                                "compare-exchange success output must be an integer boolean",
                            );
                        }
                        if matches!(
                            failure_ordering,
                            AtomicOrdering::Release | AtomicOrdering::AcqRel
                        ) {
                            self.invalid_atomic(
                                block_idx,
                                instr_idx,
                                "compare-exchange failure ordering cannot release",
                            );
                        }
                        if ordering_rank(*failure_ordering) > ordering_rank(*success_ordering) {
                            self.invalid_atomic(
                                block_idx,
                                instr_idx,
                                "compare-exchange failure ordering is stronger than success ordering",
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Checks native implicit-width arithmetic operand and result relationships.
    fn check_wide_arithmetic(&mut self) {
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                match instr.op() {
                    SsaOp::WideMul {
                        low,
                        high,
                        left,
                        right,
                        ..
                    } => {
                        self.check_same_width(
                            block_idx,
                            instr_idx,
                            &[*low, *high, *left, *right],
                            "wide multiply operands and outputs must have matching half width",
                        );
                    }
                    SsaOp::WideDiv {
                        quotient,
                        remainder,
                        high,
                        low,
                        divisor,
                        ..
                    } => {
                        self.check_same_width(
                            block_idx,
                            instr_idx,
                            &[*quotient, *remainder, *high, *low, *divisor],
                            "wide divide operands and outputs must have matching half width",
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    /// Checks native opaque clobber and effect-summary consistency.
    fn check_native_effects(&mut self) {
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                let effects = instr.op().effects();
                self.check_effect_summary(block_idx, instr_idx, effects);

                let is_terminator = instr.op().is_terminator();
                let ends_block = matches!(
                    effects.control,
                    ControlEffect::Terminator | ControlEffect::Return | ControlEffect::Throw
                );
                if is_terminator && !ends_block {
                    self.invalid_native(
                        block_idx,
                        instr_idx,
                        "terminator op must declare a block-ending control effect",
                    );
                }
                if ends_block && !is_terminator {
                    self.invalid_native(
                        block_idx,
                        instr_idx,
                        "op with a block-ending control effect must be a terminator",
                    );
                }

                if let SsaOp::NativeOpaque(data) = instr.op() {
                    let clobbers = &data.clobbers;
                    let effects = &data.effects;
                    for clobber in clobbers {
                        self.check_native_clobber(block_idx, instr_idx, clobber);
                    }
                    if effects.is_pure()
                        && clobbers.iter().any(|clobber| {
                            matches!(
                                clobber,
                                NativeClobber::Register(_)
                                    | NativeClobber::RegisterClass(_)
                                    | NativeClobber::Flags(_)
                                    | NativeClobber::Memory(_)
                                    | NativeClobber::Other(_)
                            ) || matches!(
                                clobber,
                                NativeClobber::MachineState(access) if access.writes()
                            )
                        })
                    {
                        self.invalid_native(
                            block_idx,
                            instr_idx,
                            "pure native opaque operation cannot declare clobbers",
                        );
                    }
                }
            }
        }
    }

    /// Returns the registered type for `var`.
    fn var_type(&self, var: SsaVarId) -> Option<&T::Type> {
        self.ssa.variable(var).map(SsaVariable::var_type)
    }

    /// Records a native/effect verifier error.
    fn invalid_native(&mut self, block: usize, instr_idx: usize, reason: &str) {
        self.errors.push(VerifierError::InvalidNativeOperation {
            block,
            instr_idx,
            reason: reason.to_owned(),
        });
    }

    /// Checks effect-summary invariants that passes rely on.
    fn check_effect_summary(&mut self, block: usize, instr_idx: usize, effects: SsaEffects) {
        if effects.is_pure() {
            if effects.memory != MemoryEffectLocation::None {
                self.invalid_native(block, instr_idx, "pure effect cannot reference memory");
            }
            if effects.memory_semantics != MemoryAccessSemantics::None {
                self.invalid_native(block, instr_idx, "pure effect cannot have memory semantics");
            }
            if effects.ordering.is_some() {
                self.invalid_native(block, instr_idx, "pure effect cannot have ordering");
            }
        }

        match effects.kind {
            SsaEffectKind::Atomic => {
                if effects.memory_semantics != MemoryAccessSemantics::Atomic {
                    self.invalid_native(
                        block,
                        instr_idx,
                        "atomic effect must use atomic memory semantics",
                    );
                }
                if effects.ordering.is_none() {
                    self.invalid_native(block, instr_idx, "atomic effect must declare ordering");
                }
            }
            SsaEffectKind::Fence => {
                if effects.memory_semantics != MemoryAccessSemantics::Fence {
                    self.invalid_native(
                        block,
                        instr_idx,
                        "fence effect must use fence memory semantics",
                    );
                }
                if effects.ordering.is_none() {
                    self.invalid_native(block, instr_idx, "fence effect must declare ordering");
                }
            }
            SsaEffectKind::Pure
            | SsaEffectKind::Read
            | SsaEffectKind::Write
            | SsaEffectKind::ReadWrite
            | SsaEffectKind::Call
            | SsaEffectKind::Opaque => {}
        }

        if effects.volatile && effects.memory_semantics == MemoryAccessSemantics::None {
            self.invalid_native(block, instr_idx, "volatile effect must access memory");
        }
    }

    /// Checks native clobber descriptors for structural validity.
    fn check_native_clobber(&mut self, block: usize, instr_idx: usize, clobber: &NativeClobber) {
        match clobber {
            NativeClobber::MachineState(access) => {
                if !access.is_valid() {
                    self.invalid_native(block, instr_idx, "invalid machine-state access");
                }
                if matches!(access.kind, NativeStateAccessKind::Clobber) && !access.writes() {
                    self.invalid_native(block, instr_idx, "clobber access must write state");
                }
            }
            NativeClobber::Register(register) => {
                if !register.is_valid() {
                    self.invalid_native(block, instr_idx, "invalid native register clobber");
                }
            }
            NativeClobber::RegisterClass(name)
            | NativeClobber::Flags(name)
            | NativeClobber::Memory(name)
            | NativeClobber::Other(name) => {
                if name.is_empty() {
                    self.invalid_native(block, instr_idx, "native clobber name cannot be empty");
                }
            }
        }
    }

    /// Records an atomic verifier error.
    fn invalid_atomic(&mut self, block: usize, instr_idx: usize, reason: &str) {
        self.errors.push(VerifierError::InvalidAtomicOperation {
            block,
            instr_idx,
            reason: reason.to_owned(),
        });
    }

    /// Checks that an atomic address is pointer-like when the type is known.
    fn check_atomic_address(&mut self, block: usize, instr_idx: usize, addr: SsaVarId) {
        if let Some(ty) = self.var_type(addr)
            && !T::is_unknown(ty)
            && !T::is_pointer(ty)
        {
            self.invalid_atomic(block, instr_idx, "atomic address must be pointer-like");
        }
    }

    /// Checks that an atomic value matches the declared access width when known.
    fn check_atomic_width(
        &mut self,
        block: usize,
        instr_idx: usize,
        var: SsaVarId,
        width: AtomicAccessWidth,
    ) {
        if let Some(expected) = width.bits()
            && let Some(ty) = self.var_type(var)
            && !T::is_unknown(ty)
            && T::bit_width(ty) != Some(expected)
        {
            self.invalid_atomic(
                block,
                instr_idx,
                "atomic value type width does not match access width",
            );
        }
    }

    /// Checks that all known integer widths in `vars` match.
    fn check_same_width(
        &mut self,
        block: usize,
        instr_idx: usize,
        vars: &[SsaVarId],
        reason: &str,
    ) {
        let mut expected = None;
        for var in vars {
            let Some(ty) = self.var_type(*var) else {
                continue;
            };
            if T::is_unknown(ty) {
                continue;
            }
            // Native wide multiply/divide operate on the machine word as a raw
            // integer bit-pattern. A pointer is an integer-width scalar and is a
            // faithful operand here (e.g. `div` of an address, pointer hashing):
            // accept integer *or* pointer types and let the width check below
            // enforce consistency. Rejecting pointers outright produced false
            // positives on lifted `div`/`mul` whose accumulator held a pointer.
            if !T::is_integer(ty) && !T::is_pointer(ty) {
                self.invalid_wide(block, instr_idx, "wide arithmetic values must be integers");
                continue;
            }
            let Some(width) = T::bit_width(ty) else {
                continue;
            };
            if let Some(expected) = expected {
                if width != expected {
                    self.invalid_wide(block, instr_idx, reason);
                    return;
                }
            } else {
                expected = Some(width);
            }
        }
    }

    /// Records a wide arithmetic verifier error.
    fn invalid_wide(&mut self, block: usize, instr_idx: usize, reason: &str) {
        self.errors.push(VerifierError::InvalidWideArithmetic {
            block,
            instr_idx,
            reason: reason.to_owned(),
        });
    }

    /// Returns the vector descriptor for `var` when its registered type is a known vector.
    fn var_vector_shape(&self, var: SsaVarId) -> Option<VectorDescriptor> {
        self.var_type(var).and_then(T::vector_descriptor)
    }

    /// Returns the mask descriptor for `var` when its registered type is a known mask.
    fn var_mask_shape(&self, var: SsaVarId) -> Option<VectorMaskDescriptor> {
        let ty = self.var_type(var)?;
        T::vector_mask_descriptor(ty).or_else(|| {
            (1..=u32::BITS).find_map(|lanes| {
                let shape = VectorMaskShape::new(lanes, 1)?;
                (T::vector_mask_type(shape).as_ref() == Some(ty))
                    .then_some(VectorMaskDescriptor::Fixed(shape))
            })
        })
    }

    /// Records a vector verifier error.
    fn invalid_vector(&mut self, block: usize, instr_idx: usize, reason: &str) {
        self.errors.push(VerifierError::InvalidVectorOperation {
            block,
            instr_idx,
            reason: reason.to_owned(),
        });
    }

    /// Checks that a destination variable has the expected vector shape when known.
    fn check_dest_shape(
        &mut self,
        block: usize,
        instr_idx: usize,
        dest: SsaVarId,
        expected: Option<VectorDescriptor>,
    ) {
        self.check_source_shape(block, instr_idx, dest, expected);
    }

    /// Checks that a variable has the expected vector shape when known.
    fn check_source_shape(
        &mut self,
        block: usize,
        instr_idx: usize,
        var: SsaVarId,
        expected: Option<VectorDescriptor>,
    ) {
        if let (Some(actual), Some(expected)) = (self.var_vector_shape(var), expected)
            && actual != expected
        {
            self.invalid_vector(
                block,
                instr_idx,
                "variable vector shape does not match expected shape",
            );
        }
    }

    /// Checks that a mask destination has a compatible shape when known.
    fn check_mask_dest_shape(
        &mut self,
        block: usize,
        instr_idx: usize,
        dest: SsaVarId,
        expected: VectorMaskDescriptor,
    ) {
        // A `None` expected mask type means the target does not model a
        // distinct mask type for this lane count — the mask lives in a plain
        // vector register (e.g. an AVX sign-bit mask), so there is no mask type
        // to compare against. Skip, mirroring `check_source_shape`'s Some/Some
        // guard; rejecting here would false-positive every masked op on such a
        // target.
        let Some(expected_ty) = T::vector_mask_descriptor_type(expected) else {
            return;
        };
        if let Some(ty) = self.var_type(dest) {
            if T::is_unknown(ty) {
                return;
            }
            if &expected_ty != ty {
                self.invalid_vector(
                    block,
                    instr_idx,
                    "compare destination is not a compatible vector mask",
                );
            }
        }
    }

    /// Checks that a mask source has a compatible shape when known.
    fn check_mask_source_shape(
        &mut self,
        block: usize,
        instr_idx: usize,
        mask: SsaVarId,
        expected: VectorMaskDescriptor,
    ) {
        // A `None` expected mask type means the target does not model a
        // distinct mask type for this lane count — the mask lives in a plain
        // vector register (e.g. an AVX2 `vmaskmov`/gather sign-bit mask), so
        // there is no mask type to compare against. Skip, mirroring
        // `check_source_shape`'s Some/Some guard; rejecting here would
        // false-positive every masked op whose mask register carries a
        // concrete vector type on such a target.
        let Some(expected_ty) = T::vector_mask_descriptor_type(expected) else {
            return;
        };
        if let Some(ty) = self.var_type(mask) {
            if T::is_unknown(ty) {
                return;
            }
            if &expected_ty != ty {
                self.invalid_vector(
                    block,
                    instr_idx,
                    "mask operand is not compatible with vector lane count",
                );
            }
        }
    }

    /// Checks a variable's registered type against an expected scalar type.
    fn check_var_type(
        &mut self,
        block: usize,
        instr_idx: usize,
        var: SsaVarId,
        expected: &T::Type,
        reason: &str,
    ) {
        if let Some(actual) = self.var_type(var)
            && !T::is_unknown(actual)
            && actual != expected
        {
            self.invalid_vector(block, instr_idx, reason);
        }
    }

    /// Checks dominance: every use of a variable must be dominated by its definition.
    ///
    /// For phi operands, the use is considered to be at the end of the predecessor
    /// block (not at the phi's block), following standard SSA semantics.
    fn check_dominance(
        &mut self,
        cfg: &SsaCfg<'_, T>,
        dom_tree: &DominatorTree,
        definitions: &VarMap<(usize, DefSite)>,
    ) {
        // Compute reachable blocks
        let block_count = self.ssa.block_count().max(1);
        let mut reachable = BitSet::new(block_count);
        let mut worklist = vec![0usize];
        while let Some(block_idx) = worklist.pop() {
            if block_idx < block_count && reachable.insert(block_idx) {
                for &succ in cfg.block_successors(block_idx) {
                    if succ < block_count {
                        worklist.push(succ);
                    }
                }
            }
        }

        // Check instruction uses
        let mut use_buf: Vec<SsaVarId> = Vec::new();
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            if !reachable.contains(block_idx) {
                continue;
            }

            for instr in block.instructions() {
                // Reuse one scratch buffer instead of `SsaOp::uses()`, which
                // allocates a fresh `Vec` per instruction — this runs over every
                // instruction of every function that gets verified.
                use_buf.clear();
                instr.op().for_each_use(|var| use_buf.push(var));
                for &used_var in &use_buf {
                    if let Some(&(def_block, _)) = definitions.get(used_var) {
                        // `def_block` is a variable's recorded `DefSite::block`,
                        // which malformed IR can put past the block count.
                        // `contains` asserts; an out-of-range block is by
                        // definition unreachable, so `false` is also the correct
                        // answer. The verifier is the net for malformed IR and
                        // must not abort on a shape it exists to report.
                        if !reachable.contains_checked(def_block) {
                            continue;
                        }
                        // Definition must dominate use block
                        let def_node = NodeId::new(def_block);
                        let use_node = NodeId::new(block_idx);
                        if def_node.index() < dom_tree.node_count()
                            && use_node.index() < dom_tree.node_count()
                            && !dom_tree.dominates(def_node, use_node)
                        {
                            self.errors.push(VerifierError::DominanceViolation {
                                var: used_var,
                                def_block,
                                use_block: block_idx,
                            });
                        }
                    }
                }
            }

            // Check phi operand uses: the use is at the end of the predecessor
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    let used_var = operand.value();
                    let pred_block = operand.predecessor();
                    if let Some(&(def_block, _)) = definitions.get(used_var) {
                        // Both indices come straight out of the IR — `pred_block`
                        // is a raw phi-operand predecessor, which
                        // `check_phi_operands` explicitly tolerates naming a
                        // non-existent block.
                        if !reachable.contains_checked(def_block)
                            || !reachable.contains_checked(pred_block)
                        {
                            continue;
                        }
                        // Definition must dominate the predecessor block
                        let def_node = NodeId::new(def_block);
                        let pred_node = NodeId::new(pred_block);
                        if def_node.index() < dom_tree.node_count()
                            && pred_node.index() < dom_tree.node_count()
                            && !dom_tree.dominates(def_node, pred_node)
                        {
                            self.errors.push(VerifierError::DominanceViolation {
                                var: used_var,
                                def_block,
                                use_block: pred_block,
                            });
                        }
                    }
                }
            }
        }
    }
}
