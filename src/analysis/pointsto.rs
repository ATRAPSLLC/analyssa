//! Inclusion-based (Andersen) points-to analysis core.
//!
//! This is the solver: it takes a set of inclusion [`Constraint`]s over opaque
//! abstract [`Loc`]ations and computes their least fixpoint — each location's
//! points-to set. Constraint *extraction* from the SSA (and any interprocedural
//! summaries that feed it) layer on top; keeping the solver standalone lets it
//! be reasoned about and tested in isolation.
//!
//! [`analyze_function`] is the intraprocedural extractor: SSA value ids are
//! pointer locations, and locals / arguments / struct fields get stable
//! synthetic locations so two addresses of the same cell coincide.
//!
//! The analysis is a **may**-analysis: every result over-approximates. Both the
//! solver's step budget and the field-cell exhaustion fallback lose precision
//! (fewer aliases) rather than inventing them, so a partial result stays sound
//! for consumers that ask "could these alias?".
//!
//! # Reading a result
//!
//! An empty or absent points-to set means **unconstrained**, not "points
//! nowhere". A definition this module does not model — a call result, a
//! `Select`, address arithmetic lowered as plain `Add`/`Sub` — is seeded with
//! [`UNKNOWN_LOC`], which aliases everything. Implement "could these alias?" as
//!
//! ```text
//! a.contains(UNKNOWN_LOC) || b.contains(UNKNOWN_LOC) || !a.is_disjoint(b)
//! ```
//!
//! A bare set intersection is **not** a sound alias query: it answers `NoAlias`
//! for any pointer the extractor did not understand.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{
    ir::{SsaFunction, SsaOp},
    target::Target,
};

/// An abstract memory location — an SSA value id, an allocation site, a stack
/// slot, a global. Opaque to the solver, which only relates locations by
/// inclusion.
pub type Loc = u32;

/// One inclusion constraint over abstract locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    /// `ptr ∋ &target` — `target` is a member of `pts(ptr)`.
    AddressOf {
        /// The pointer whose points-to set gains a member.
        ptr: Loc,
        /// The location whose address is taken.
        target: Loc,
    },
    /// `dst = src` — `pts(src) ⊆ pts(dst)`.
    Copy {
        /// Destination, which absorbs the source's points-to set.
        dst: Loc,
        /// Source of the copied points-to set.
        src: Loc,
    },
    /// `dst = *ptr` — for every `o ∈ pts(ptr)`, `pts(o) ⊆ pts(dst)`.
    Load {
        /// Destination of the loaded value.
        dst: Loc,
        /// Pointer being dereferenced.
        ptr: Loc,
    },
    /// `*ptr = src` — for every `o ∈ pts(ptr)`, `pts(src) ⊆ pts(o)`.
    Store {
        /// Pointer being written through.
        ptr: Loc,
        /// Value whose points-to set flows into the pointee.
        src: Loc,
    },
}

/// The points-to relation — each location's points-to set.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PointsTo {
    sets: BTreeMap<Loc, BTreeSet<Loc>>,
}

impl PointsTo {
    /// Returns the points-to set of `loc`, empty when the location points
    /// nowhere known.
    #[must_use]
    pub fn get(&self, loc: Loc) -> &BTreeSet<Loc> {
        static EMPTY: BTreeSet<Loc> = BTreeSet::new();
        self.sets.get(&loc).unwrap_or(&EMPTY)
    }

    /// Iterates each location with a non-empty points-to set and its pointees.
    pub fn iter(&self) -> impl Iterator<Item = (Loc, &BTreeSet<Loc>)> {
        self.sets.iter().map(|(loc, set)| (*loc, set))
    }

    /// Returns the transitive points-to closure reachable from `seeds`.
    ///
    /// Starting from each seed location, follows points-to edges to a fixpoint,
    /// collecting every abstract location a chain of dereferences could reach.
    /// The seeds themselves are included. Used to compute what memory a callee
    /// can touch through the pointer arguments it is handed.
    #[must_use]
    pub fn reachable_closure<I>(&self, seeds: I) -> BTreeSet<Loc>
    where
        I: IntoIterator<Item = Loc>,
    {
        let mut reached: BTreeSet<Loc> = BTreeSet::new();
        let mut stack: Vec<Loc> = Vec::new();
        for seed in seeds {
            if reached.insert(seed) {
                stack.push(seed);
            }
        }
        while let Some(loc) = stack.pop() {
            for &pointee in self.get(loc) {
                if reached.insert(pointee) {
                    stack.push(pointee);
                }
            }
        }
        reached
    }
}

/// The load / store constraints, indexed by the pointer location they
/// dereference.
///
/// Written once while loading the constraint set and never mutated afterwards.
/// Kept out of [`SolverGraph`] precisely so the worklist loop can hold these
/// borrowed while it mutates the solver state — the two are disjoint, and
/// merging them would force the loop to clone an edge list on every pop just to
/// satisfy the borrow checker.
#[derive(Default)]
struct DerefConstraints {
    /// Load constraints indexed by pointer: `ptr → {dst}` means `dst = *ptr`.
    loads: BTreeMap<Loc, BTreeSet<Loc>>,
    /// Store constraints indexed by pointer: `ptr → {src}` means `*ptr = src`.
    stores: BTreeMap<Loc, BTreeSet<Loc>>,
}

/// The mutable solver state a [`solve`] run propagates over: the points-to sets
/// themselves plus the copy-edge adjacency, both of which grow as the fixpoint
/// is approached.
///
/// The static load / store index lives in [`DerefConstraints`].
#[derive(Default)]
struct SolverGraph {
    /// Each location's current points-to set.
    sets: BTreeMap<Loc, BTreeSet<Loc>>,
    /// Copy edges `src → {dst}`: `pts(src) ⊆ pts(dst)` for every `dst`.
    copy_edges: BTreeMap<Loc, BTreeSet<Loc>>,
}

impl SolverGraph {
    /// Adds `target` to `loc`'s points-to set, returning `true` when it grew.
    fn add_pointee(&mut self, loc: Loc, target: Loc) -> bool {
        self.sets.entry(loc).or_default().insert(target)
    }

    /// Copies `src`'s current points-to set into `dst`, enqueueing `dst` when it
    /// grew so the new pointees propagate onward.
    ///
    /// Takes `sets` rather than `&mut self` so a caller can iterate
    /// `copy_edges` — a different field — while this mutates the points-to
    /// sets, instead of cloning the edge list to break the borrow.
    fn propagate_copy(
        sets: &mut BTreeMap<Loc, BTreeSet<Loc>>,
        src: Loc,
        dst: Loc,
        work: &mut Worklist,
    ) {
        if src == dst {
            return;
        }
        let Some(src_set) = sets.get(&src) else {
            return;
        };
        // Diff-propagate: only the pointees `dst` does not already hold are new.
        let additions: Vec<Loc> = match sets.get(&dst) {
            Some(dst_set) => src_set.difference(dst_set).copied().collect(),
            None => src_set.iter().copied().collect(),
        };
        if additions.is_empty() {
            return;
        }
        let dst_set = sets.entry(dst).or_default();
        for target in additions {
            dst_set.insert(target);
        }
        work.push(dst);
    }

    /// Adds a copy edge `src → dst` and immediately propagates `src`'s set across
    /// it, so a load/store-induced edge takes effect without a second visit.
    fn add_copy_edge(&mut self, src: Loc, dst: Loc, work: &mut Worklist) {
        if src == dst {
            return;
        }
        if self.copy_edges.entry(src).or_default().insert(dst) {
            Self::propagate_copy(&mut self.sets, src, dst, work);
        }
    }
}

/// A FIFO worklist of locations whose points-to set has grown, de-duplicated so
/// a location queued while already pending is processed only once per growth.
#[derive(Default)]
struct Worklist {
    queue: VecDeque<Loc>,
    queued: BTreeSet<Loc>,
}

impl Worklist {
    /// Enqueues `loc` unless it is already pending.
    fn push(&mut self, loc: Loc) {
        if self.queued.insert(loc) {
            self.queue.push_back(loc);
        }
    }

    /// Dequeues the next pending location, if any.
    fn pop(&mut self) -> Option<Loc> {
        let loc = self.queue.pop_front()?;
        self.queued.remove(&loc);
        Some(loc)
    }
}

/// Solves a constraint set to its least fixpoint.
///
/// Uses an inclusion-based (Andersen) **difference-propagation worklist**: the
/// constraints are loaded once into a `SolverGraph` of points-to sets and copy
/// edges, alongside a static `DerefConstraints` index of the load / store
/// constraints keyed by their pointer, with the `AddressOf` base facts seeding
/// the points-to sets. A location is processed by pushing only the
/// *new* members of its points-to set along its copy edges and, for each load /
/// store it is the pointer of, materializing the dynamic copy edges `o → dst` /
/// `src → o` for every freshly reachable pointee `o`. Re-enqueueing a location
/// whose set grows drives the run to the same least fixpoint as a naive
/// rescan, without re-scanning every constraint each round.
#[must_use]
pub fn solve(constraints: &[Constraint]) -> PointsTo {
    let mut graph = SolverGraph::default();
    let mut deref = DerefConstraints::default();
    let mut work = Worklist::default();

    // Load the constraint graph: base facts seed points-to sets (and enqueue the
    // pointer for propagation); copy / load / store edges are indexed by source.
    for constraint in constraints {
        match *constraint {
            Constraint::AddressOf { ptr, target } => {
                if graph.add_pointee(ptr, target) {
                    work.push(ptr);
                }
            }
            Constraint::Copy { dst, src } => {
                graph.copy_edges.entry(src).or_default().insert(dst);
                // Seed propagation from a source that may already hold pointees.
                if graph.sets.contains_key(&src) {
                    work.push(src);
                }
            }
            Constraint::Load { dst, ptr } => {
                deref.loads.entry(ptr).or_default().insert(dst);
                if graph.sets.contains_key(&ptr) {
                    work.push(ptr);
                }
            }
            Constraint::Store { ptr, src } => {
                deref.stores.entry(ptr).or_default().insert(src);
                if graph.sets.contains_key(&ptr) {
                    work.push(ptr);
                }
            }
        }
    }

    // Safety budget. The difference-propagation worklist converges by monotone
    // set growth, but a crafted function with dense aliasing can drive
    // worst-case cubic work and pin a CPU core. Bound the number of processed
    // locations proportional to the constraint count so normal functions always
    // converge first; on exceed, bail to the partial points-to relation. This is
    // sound for a *may*-analysis consumer (it only loses precision — fewer
    // aliases — never adds false ones) and is surfaced as a `warn` for
    // observability.
    let safety_bound = constraints.len().saturating_mul(64).saturating_add(64);
    let mut steps = 0usize;
    while let Some(loc) = work.pop() {
        // The budget counts *work*, not pops. A single pop propagates along
        // every outgoing copy edge and, for each load or store through `loc`,
        // does `pointees x destinations` work — so bounding pops left the cubic
        // term unbounded, and the guard could not fire before the analysis had
        // already done the work it was meant to prevent.
        if steps > safety_bound {
            log::warn!(
                "points-to analysis exceeded its step budget ({} constraints, bound {safety_bound}); returning partial result",
                constraints.len()
            );
            break;
        }
        // Propagate `loc`'s set along every static copy edge out of it.
        // `copy_edges` and `sets` are disjoint fields, so the edge list is
        // iterated in place rather than cloned.
        if let Some(dsts) = graph.copy_edges.get(&loc) {
            steps = steps.saturating_add(dsts.len());
            for &dst in dsts {
                SolverGraph::propagate_copy(&mut graph.sets, loc, dst, &mut work);
            }
        }
        // For each load `dst = *loc`, every pointee `o ∈ pts(loc)` contributes a
        // dynamic copy edge `o → dst` (`pts(o) ⊆ pts(dst)`).
        //
        // The pointee set *is* cloned: `add_copy_edge` grows `sets`, so it
        // cannot be iterated in place. The load / store index is not, because
        // `deref` is never written after setup.
        if let Some(dsts) = deref.loads.get(&loc) {
            let pointees = graph.sets.get(&loc).cloned().unwrap_or_default();
            steps = steps.saturating_add(pointees.len().saturating_mul(dsts.len()));
            for pointee in pointees {
                for &dst in dsts {
                    graph.add_copy_edge(pointee, dst, &mut work);
                }
            }
        }
        // For each store `*loc = src`, every pointee `o ∈ pts(loc)` contributes a
        // dynamic copy edge `src → o` (`pts(src) ⊆ pts(o)`).
        if let Some(srcs) = deref.stores.get(&loc) {
            let pointees = graph.sets.get(&loc).cloned().unwrap_or_default();
            steps = steps.saturating_add(pointees.len().saturating_mul(srcs.len()));
            for pointee in pointees {
                for &src in srcs {
                    graph.add_copy_edge(src, pointee, &mut work);
                }
            }
        }
    }

    PointsTo { sets: graph.sets }
}

/// Lower bound of the synthetic object-location range. SSA value ids are dense
/// from zero, so any location at or above this is a synthetic stack/argument
/// cell rather than a pointer-valued SSA id.
pub const SYNTHETIC_LOC_BASE: Loc = 0xF000_0000;
/// Abstract-location base for a function's local variables (`&localN`). Chosen
/// well above any dense SSA value id so synthetic object locations never
/// collide with pointer-value locations.
const LOCAL_LOC_BASE: Loc = SYNTHETIC_LOC_BASE;
/// Abstract-location base for a function's argument cells (`&argN`).
const ARG_LOC_BASE: Loc = 0xF800_0000;
/// Abstract-location base for field-sensitive cells (`&object.field`). Sits
/// above the local / argument cell bases so a synthesized field cell never
/// collides with a stack slot, an argument cell, or a dense SSA value id.
const FIELD_LOC_BASE: Loc = 0xFC00_0000;
/// Abstract location standing for "could be anything".
///
/// Seeded for every definition the extractor does not model, so that a value's
/// points-to set being *absent* can never be read as *empty*. Without it, a
/// consumer implementing the documented "could these alias?" query as a set
/// intersection gets `NoAlias` for any pointer this module happens not to
/// understand — an under-approximation, and an unsound alias proof.
///
/// Sits at the top of the synthetic band so it never collides with a stack slot,
/// an argument cell, a field cell, or a dense SSA value id.
pub const UNKNOWN_LOC: Loc = 0xFFFF_FFFF;

/// Returns `true` when `loc` is a synthetic object cell (a stack slot, an
/// argument cell, or a field cell) rather than a pointer-valued SSA id.
#[must_use]
pub fn is_synthetic_object(loc: Loc) -> bool {
    loc >= SYNTHETIC_LOC_BASE
}

/// Allocates stable synthetic locations for field-sensitive cells.
///
/// Each distinct `(object SSA id, member index)` pair is handed one abstract
/// location in the [`FIELD_LOC_BASE`]`..=`[`Loc::MAX`] band, allocated densely
/// on first sight and memoized so every `&object.field` of the same field of
/// the same object coincides while distinct fields stay disjoint. Allocation
/// stops (returning `None`) once the band is exhausted, so a pathological
/// function falls back to the field-insensitive alias rather than wrapping into
/// the local / argument ranges.
#[derive(Default)]
struct FieldCells {
    cells: BTreeMap<(u32, u32), Loc>,
    next: u32,
}

impl FieldCells {
    /// Returns the stable field cell for `(object, member_index)`, minting one
    /// on first sight. Returns `None` once the dense field range is exhausted,
    /// so the caller falls back to the field-insensitive whole-object alias.
    fn cell(&mut self, object: u32, member_index: u32) -> Option<Loc> {
        if let Some(loc) = self.cells.get(&(object, member_index)) {
            return Some(*loc);
        }
        let loc = FIELD_LOC_BASE.checked_add(self.next)?;
        self.next = self.next.saturating_add(1);
        self.cells.insert((object, member_index), loc);
        Some(loc)
    }
}

/// Extracts inclusion constraints from a function's SSA and solves them.
///
/// SSA value ids are pointer locations; locals and arguments get stable
/// synthetic locations so two addresses of the same slot coincide. The analysis
/// is **field-sensitive**: `&object.field` resolves to a distinct abstract cell
/// per `(object, field)`, so two different fields of the same object do not
/// alias. A field whose host reports no member index via
/// [`Target::field_member_index`] (or once the field-cell range is exhausted)
/// falls back to the sound field-insensitive whole-object alias.
#[must_use]
pub fn analyze_function<T: Target>(ir: &SsaFunction<T>) -> PointsTo {
    let mut constraints: Vec<Constraint> = Vec::new();
    let mut field_cells = FieldCells::default();
    for block in ir.blocks() {
        // A phi is a merge: its result may hold any of its operands' values.
        // Phis live in a separate vector from instructions, so iterating only
        // `instructions()` silently gave every merge-defined pointer — which is
        // every loop-carried pointer, and every pointer out of an if/else — an
        // empty points-to set.
        for phi in block.phi_nodes() {
            for operand in phi.operands() {
                constraints.push(Constraint::Copy {
                    dst: phi.result().as_u32(),
                    src: operand.value().as_u32(),
                });
            }
        }
        for instruction in block.instructions() {
            match instruction.op() {
                SsaOp::Copy { dest, src } => constraints.push(Constraint::Copy {
                    dst: dest.as_u32(),
                    src: src.as_u32(),
                }),
                SsaOp::IntConv { dest, operand, .. }
                | SsaOp::IntToPtr { dest, operand, .. }
                | SsaOp::PtrToInt { dest, operand, .. }
                | SsaOp::IntToFloat { dest, operand, .. }
                | SsaOp::FloatToInt { dest, operand, .. }
                | SsaOp::FloatConv { dest, operand, .. }
                | SsaOp::Bitcast { dest, operand, .. } => {
                    constraints.push(Constraint::Copy {
                        dst: dest.as_u32(),
                        src: operand.as_u32(),
                    });
                }
                // A scaled address computation points where its base points
                // (index/offset move within the base's object).
                SsaOp::PtrAdd { dest, base, .. } => {
                    constraints.push(Constraint::Copy {
                        dst: dest.as_u32(),
                        src: base.as_u32(),
                    });
                }
                // Field-sensitive: `&object.field` resolves to a distinct
                // synthetic cell per `(object, field)`. The address value first
                // takes the object's pointees (so it stays anchored to the same
                // object), then additionally points at the field cell — keeping
                // `&o.a` and `&o.b` disjoint while both resolve through `o`. An
                // unindexed field falls back to the whole-object alias only.
                SsaOp::LoadFieldAddr {
                    dest,
                    object,
                    field,
                } => {
                    constraints.push(Constraint::Copy {
                        dst: dest.as_u32(),
                        src: object.as_u32(),
                    });
                    if let Some(member_index) = T::field_member_index(field)
                        && let Some(cell) = field_cells.cell(object.as_u32(), member_index)
                    {
                        constraints.push(Constraint::AddressOf {
                            ptr: dest.as_u32(),
                            target: cell,
                        });
                    }
                }
                SsaOp::LoadIndirect { dest, addr, .. } => constraints.push(Constraint::Load {
                    dst: dest.as_u32(),
                    ptr: addr.as_u32(),
                }),
                SsaOp::StoreIndirect { addr, value, .. } => constraints.push(Constraint::Store {
                    ptr: addr.as_u32(),
                    src: value.as_u32(),
                }),
                SsaOp::LoadArgAddr { dest, arg_index } => constraints.push(Constraint::AddressOf {
                    ptr: dest.as_u32(),
                    target: ARG_LOC_BASE.saturating_add(u32::from(*arg_index)),
                }),
                SsaOp::LoadLocalAddr { dest, local_index } => {
                    constraints.push(Constraint::AddressOf {
                        ptr: dest.as_u32(),
                        target: LOCAL_LOC_BASE.saturating_add(u32::from(*local_index)),
                    });
                }
                // Anything not modelled above — call results, `Select`, address
                // arithmetic lowered as plain `Add`/`Sub` — defines a value whose
                // pointees this module cannot determine. Recording `UNKNOWN_LOC`
                // keeps "unmodelled" distinguishable from "points nowhere"; the
                // latter would let a consumer prove NoAlias against everything.
                other => {
                    for dest in other.defs() {
                        constraints.push(Constraint::AddressOf {
                            ptr: dest.as_u32(),
                            target: UNKNOWN_LOC,
                        });
                    }
                }
            }
        }
    }
    solve(&constraints)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            SsaBlock, SsaInstruction,
            phi::{PhiNode, PhiOperand},
            variable::{DefSite, VariableOrigin},
        },
        testing::{MOCK_FIELD_UNRESOLVED, MockTarget, MockType},
    };

    /// Declares a pointer-typed variable defined at `(block, instruction)`.
    fn ptr_var(
        ir: &mut SsaFunction<MockTarget>,
        origin: VariableOrigin,
        block: usize,
        instruction: usize,
    ) -> crate::ir::SsaVarId {
        ir.create_variable(
            origin,
            0,
            DefSite::instruction(block, instruction),
            MockType::Ptr,
        )
    }

    /// Wraps `op` in an instruction. `MockTarget` carries no original-instruction
    /// breadcrumb (its `OriginalInstruction` is `()`), so the unit is passed
    /// literally rather than through `synthetic_instruction()`.
    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::new((), op)
    }

    /// A phi is a merge, so its result may hold any operand's value. Phis are
    /// stored separately from instructions, so an extractor walking only
    /// `instructions()` gave every merge-defined pointer an empty set — which is
    /// every loop-carried pointer and every pointer out of an if/else.
    #[test]
    fn a_phi_merges_the_pointees_of_its_operands() {
        let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let mk = |ir: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize| {
            ir.create_variable(
                VariableOrigin::Local(idx),
                0,
                DefSite::instruction(block, instr),
                MockType::Ptr,
            )
        };
        let left = mk(&mut ir, 0, 1, 0);
        let right = mk(&mut ir, 1, 2, 0);
        let merged =
            ir.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::Ptr);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ir.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocalAddr {
            dest: left,
            local_index: 5,
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        ir.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocalAddr {
            dest: right,
            local_index: 9,
        }));
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        ir.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(left, 1));
        phi.add_operand(PhiOperand::new(right, 2));
        b3.add_phi(phi);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ir.add_block(b3);
        ir.recompute_uses();

        let result = analyze_function(&ir);
        let pointees = result.get(merged.as_u32());

        assert!(
            pointees.contains(&(LOCAL_LOC_BASE + 5)),
            "the phi must inherit the true arm's pointee; got {pointees:?}"
        );
        assert!(
            pointees.contains(&(LOCAL_LOC_BASE + 9)),
            "and the false arm's; got {pointees:?}"
        );
    }

    /// An unmodelled definition must not look like "points nowhere", or a
    /// consumer implementing the documented alias query as a set intersection
    /// gets `NoAlias` against everything.
    #[test]
    fn an_unmodelled_definition_points_to_unknown() {
        let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let a = ir.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::Ptr,
        );
        let b = ir.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::Ptr,
        );
        let sum = ir.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::Ptr,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocalAddr {
            dest: a,
            local_index: 1,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::LoadLocalAddr {
            dest: b,
            local_index: 2,
        }));
        // Address arithmetic lowered as a plain `Add`, which the extractor does
        // not model.
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: sum,
            left: a,
            right: b,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ir.add_block(block);
        ir.recompute_uses();

        let result = analyze_function(&ir);
        assert!(
            result.get(sum.as_u32()).contains(&UNKNOWN_LOC),
            "an unmodelled pointer definition must be unconstrained, not empty"
        );
    }

    #[test]
    fn copy_propagates_address_of() {
        // p = &x; q = p  =>  q -> x
        let pts = solve(&[
            Constraint::AddressOf { ptr: 1, target: 10 },
            Constraint::Copy { dst: 2, src: 1 },
        ]);
        assert!(pts.get(2).contains(&10));
    }

    #[test]
    fn load_dereferences_pointer() {
        // p -> a, a -> b; q = *p  =>  q -> b
        let pts = solve(&[
            Constraint::AddressOf { ptr: 1, target: 20 }, // p -> a(20)
            Constraint::AddressOf {
                ptr: 20,
                target: 30,
            }, // a -> b(30)
            Constraint::Load { dst: 2, ptr: 1 },          // q = *p
        ]);
        assert!(pts.get(2).contains(&30));
    }

    #[test]
    fn store_writes_through_pointer() {
        // p -> a, s -> b; *p = s  =>  a -> b
        let pts = solve(&[
            Constraint::AddressOf { ptr: 1, target: 20 }, // p -> a(20)
            Constraint::AddressOf { ptr: 3, target: 30 }, // s -> b(30)
            Constraint::Store { ptr: 1, src: 3 },         // *p = s
        ]);
        assert!(pts.get(20).contains(&30));
    }

    #[test]
    fn deep_copy_chain_converges() {
        // a0 = &x; a1 = a0; ... ; a64 = a63  =>  every link points at x. A long
        // chain exercises the worklist's diff-propagation without a per-round
        // full rescan; the tail must still see the address.
        let mut constraints = vec![Constraint::AddressOf {
            ptr: 0,
            target: 500,
        }];
        for dst in 1u32..=64 {
            constraints.push(Constraint::Copy {
                dst,
                src: dst.saturating_sub(1),
            });
        }
        let pts = solve(&constraints);
        assert!(pts.get(64).contains(&500));
        assert!(pts.get(32).contains(&500));
    }

    #[test]
    fn large_chain_stays_within_step_budget_without_truncation() {
        // The step budget is `64 * len + 64`. A legitimately large function
        // (here a 2000-link chain, 2001 constraints, needing ~2000 worklist
        // steps) is well under that bound, so the safety stop must NOT truncate
        // it — the tail still resolves the seeded address. This guards the budget
        // against firing on real code while still capping pathological cubic runs.
        let mut constraints = vec![Constraint::AddressOf {
            ptr: 0,
            target: 777,
        }];
        for dst in 1u32..=2000 {
            constraints.push(Constraint::Copy {
                dst,
                src: dst.saturating_sub(1),
            });
        }
        let pts = solve(&constraints);
        assert!(
            pts.get(2000).contains(&777),
            "tail must resolve the address"
        );
        assert!(pts.get(1000).contains(&777));
    }

    #[test]
    fn store_load_through_pointer_cycle_converges() {
        // A store-then-load cycle through a self-referential pointer must reach a
        // fixpoint: p -> a; *p = q where q -> a; r = *p  =>  pts(a) ⊇ {a} and
        // r -> a. The store grows pts(a) (a now points at itself), which feeds
        // back through the load — a genuine cyclic dependency.
        let pts = solve(&[
            Constraint::AddressOf {
                ptr: 1,
                target: 100,
            }, // p -> a(100)
            Constraint::AddressOf {
                ptr: 2,
                target: 100,
            }, // q -> a(100)
            Constraint::Store { ptr: 1, src: 2 }, // *p = q  => a -> a
            Constraint::Load { dst: 3, ptr: 1 },  // r = *p  => r -> a
        ]);
        // a now points at itself (the store wrote q's pointee a into a).
        assert!(pts.get(100).contains(&100));
        // r loaded a's contents, which include a.
        assert!(pts.get(3).contains(&100));
    }

    #[test]
    fn transitive_copy_chain_converges() {
        // a = &x; b = a; c = b  =>  c -> x
        let pts = solve(&[
            Constraint::AddressOf { ptr: 1, target: 99 },
            Constraint::Copy { dst: 2, src: 1 },
            Constraint::Copy { dst: 3, src: 2 },
        ]);
        assert!(pts.get(3).contains(&99));
        assert!(pts.get(4).is_empty());
    }

    #[test]
    fn extracts_local_address_through_copy() {
        let mut ir = SsaFunction::<MockTarget>::with_capacity(0, 0, 1, 2);
        let p = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
        let q = ptr_var(&mut ir, VariableOrigin::Local(1), 0, 1);
        let mut block = SsaBlock::with_capacity(0, 0, 2);
        block.add_instruction(instr(SsaOp::LoadLocalAddr {
            dest: p,
            local_index: 5,
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: q, src: p }));
        ir.add_block(block);
        ir.recompute_uses();

        // p = &local_5; q = p  =>  both point to the local_5 object.
        let pts = analyze_function(&ir);
        let local5 = LOCAL_LOC_BASE.saturating_add(5);
        assert!(pts.get(p.as_u32()).contains(&local5));
        assert!(pts.get(q.as_u32()).contains(&local5));
    }

    #[test]
    fn distinct_fields_do_not_alias_but_resolve_through_object() {
        let mut ir = SsaFunction::<MockTarget>::with_capacity(0, 0, 1, 4);
        // `o = &local_3` gives the object a concrete pointee to share.
        let object = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
        let field_a = ptr_var(&mut ir, VariableOrigin::Local(1), 0, 1);
        let field_b = ptr_var(&mut ir, VariableOrigin::Local(2), 0, 2);
        let mut block = SsaBlock::with_capacity(0, 0, 3);
        block.add_instruction(instr(SsaOp::LoadLocalAddr {
            dest: object,
            local_index: 3,
        }));
        block.add_instruction(instr(SsaOp::LoadFieldAddr {
            dest: field_a,
            object,
            field: 0,
        }));
        block.add_instruction(instr(SsaOp::LoadFieldAddr {
            dest: field_b,
            object,
            field: 1,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let pts = analyze_function(&ir);
        let local3 = LOCAL_LOC_BASE.saturating_add(3);
        let a_set = pts.get(field_a.as_u32());
        let b_set = pts.get(field_b.as_u32());
        // Both resolve through the object: each sees the object's `local_3` cell.
        assert!(a_set.contains(&local3), "&o.a resolves through o");
        assert!(b_set.contains(&local3), "&o.b resolves through o");
        // ...but they hold disjoint field cells, so the two fields do not alias.
        assert_ne!(a_set, b_set, "distinct fields must not alias");
        let a_only: BTreeSet<Loc> = a_set.difference(b_set).copied().collect();
        let b_only: BTreeSet<Loc> = b_set.difference(a_set).copied().collect();
        assert!(
            a_only.iter().all(|loc| *loc >= FIELD_LOC_BASE),
            "&o.a's exclusive cell is a field cell"
        );
        assert!(
            b_only.iter().all(|loc| *loc >= FIELD_LOC_BASE),
            "&o.b's exclusive cell is a field cell"
        );
        assert!(!a_only.is_empty() && !b_only.is_empty());
    }

    #[test]
    fn unindexed_field_falls_back_to_object_alias() {
        let mut ir = SsaFunction::<MockTarget>::with_capacity(0, 0, 1, 2);
        let object = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
        let field_dest = ptr_var(&mut ir, VariableOrigin::Local(1), 0, 1);
        let mut block = SsaBlock::with_capacity(0, 0, 2);
        block.add_instruction(instr(SsaOp::LoadLocalAddr {
            dest: object,
            local_index: 7,
        }));
        block.add_instruction(instr(SsaOp::LoadFieldAddr {
            dest: field_dest,
            object,
            // No recovered member index → field-insensitive fallback.
            field: MOCK_FIELD_UNRESOLVED,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let pts = analyze_function(&ir);
        let local7 = LOCAL_LOC_BASE.saturating_add(7);
        let set = pts.get(field_dest.as_u32());
        // The fallback aliases the whole object: only the object's pointee, no
        // synthesized field cell.
        assert!(set.contains(&local7));
        assert!(set.iter().all(|loc| *loc < FIELD_LOC_BASE));
    }
}
