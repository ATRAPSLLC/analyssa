//! Phi node representation for SSA form.
//!
//! Phi nodes are the cornerstone of SSA form - they merge values at control flow
//! join points. When multiple control flow paths converge, a phi node selects
//! which value to use based on which predecessor path was taken.
//!
//! # Semantics
//!
//! A phi node `v3 = phi(v1 from B1, v2 from B2)` means:
//! - If control reached this block from predecessor B1, the phi evaluates to `v1`
//! - If control reached from predecessor B2, the phi evaluates to `v2`
//!
//! Phi nodes have "instantaneous" semantics: all phi nodes at the start of a block
//! are conceptually evaluated simultaneously before any regular instruction executes.
//! This avoids ordering issues when one phi's result feeds into another phi's operand.
//!
//! # Placement Algorithm (Cytron et al.)
//!
//! During SSA construction, phi nodes are placed at dominance frontiers:
//!
//! 1. Compute the dominator tree of the CFG
//! 2. Compute dominance frontiers for each block
//! 3. For each variable, find all blocks that define it
//! 4. Place phi nodes at the dominance frontiers of those definition blocks
//! 5. Prune: skip phi placement for variables not live at the frontier block entry
//!
//! # Block Structure
//!
//! ```text
//! Block B3:
//!   v5 = phi(v1 from B1, v2 from B2)   // Merging values
//!   v6 = add v5, v3
//!   jump B4
//! ```
//!
//! # Invariants
//!
//! - Each phi has exactly one operand per predecessor of its containing block
//! - All operands should have the same type
//! - A phi's result is derived from its `origin` (the original variable being merged)
//! - Phi nodes appear ONLY at block start, before any instructions

use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::variable::{SsaVarId, VariableOrigin};

/// Which kind of incoming edge a phi operand names.
///
/// A phi merges the values that reach a block, and control reaches a block two
/// ways: a terminator transfers to it, or a throw somewhere inside a protected
/// region dispatches to the handler that guards it. The second kind has no
/// edge to place a copy on and no single point it leaves from, so telling the
/// two apart is what lets a consumer treat each correctly rather than treating
/// all operands as one kind and being wrong about the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PhiEdge {
    /// A terminator in the named block transfers control here.
    #[default]
    Normal,
    /// A throw anywhere in the named block - a block of a protected region -
    /// may dispatch here.
    ///
    /// One block can contribute several values this way: the throw can happen
    /// at any point in it, so every value the merged storage may hold while the
    /// block runs is a value the handler may see.
    Exceptional,
}

/// What [`PhiNode::replace_operand_value`] did to the operands naming the
/// replaced value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OperandReplacement {
    /// Operands that name the new value instead.
    pub rewritten: usize,
    /// Operands dropped because the block already listed the new value for a
    /// throw.
    pub merged: usize,
}

impl OperandReplacement {
    /// How many operands no longer name the replaced value.
    #[must_use]
    pub const fn total(self) -> usize {
        self.rewritten.saturating_add(self.merged)
    }
}

/// An operand of a phi node - a value coming from a specific predecessor block.
///
/// Each phi operand represents one possible value that could be selected,
/// associated with the predecessor block from which that value comes.
///
/// # Examples
///
/// ```rust
/// use analyssa::ir::{phi::PhiOperand, variable::SsaVarId};
///
/// // Value coming from block 1
/// let var = SsaVarId::from_index(0);
/// let operand = PhiOperand::new(var, 1);
/// assert_eq!(operand.value(), var);
/// assert_eq!(operand.predecessor(), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhiOperand {
    /// The SSA variable providing the value.
    value: SsaVarId,
    /// The predecessor block from which this value comes. Stored as `u32`
    /// (block counts never approach 2^32) to keep `PhiOperand` compact, since
    /// phi nodes hold one operand per predecessor.
    predecessor: u32,
    /// How control gets here from that block.
    #[cfg_attr(feature = "serde", serde(default))]
    edge: PhiEdge,
}

impl PhiOperand {
    /// Creates a new phi operand.
    ///
    /// # Arguments
    ///
    /// * `value` - The SSA variable providing the value
    /// * `predecessor` - The block index from which this value comes
    #[must_use]
    pub const fn new(value: SsaVarId, predecessor: usize) -> Self {
        Self {
            value,
            predecessor: predecessor as u32,
            edge: PhiEdge::Normal,
        }
    }

    /// Creates an operand for a value a throw in `source` may bring here.
    ///
    /// `source` is a block of the protected region this phi's block guards, not
    /// a block whose terminator transfers here.
    ///
    /// # Arguments
    ///
    /// * `value` - The SSA variable the merged storage may hold at the throw.
    /// * `source` - The protected block the throw may happen in.
    #[must_use]
    pub const fn exceptional(value: SsaVarId, source: usize) -> Self {
        Self {
            value,
            predecessor: source as u32,
            edge: PhiEdge::Exceptional,
        }
    }

    /// Returns how control gets here from the named block.
    #[must_use]
    pub const fn edge(&self) -> PhiEdge {
        self.edge
    }

    /// Returns whether this operand names an exceptional edge.
    #[must_use]
    pub const fn is_exceptional(&self) -> bool {
        matches!(self.edge, PhiEdge::Exceptional)
    }

    /// Returns the SSA variable providing the value.
    #[must_use]
    pub const fn value(&self) -> SsaVarId {
        self.value
    }

    /// Replaces the value this operand carries, keeping the edge it names.
    ///
    /// Rebuilding the operand instead would reset its [`PhiEdge`] to `Normal`,
    /// turning a value a throw brings into a value a terminator brings.
    ///
    /// # Arguments
    ///
    /// * `value` - The SSA variable the operand now carries.
    pub const fn set_value(&mut self, value: SsaVarId) {
        self.value = value;
    }

    /// Returns the predecessor block index.
    #[must_use]
    pub const fn predecessor(&self) -> usize {
        self.predecessor as usize
    }

    /// Updates the predecessor block index.
    ///
    /// Used when block merging redirects edges, requiring phi operands
    /// to reference the new predecessor instead of the eliminated trampoline.
    ///
    /// **Unchecked**: a `PhiOperand` cannot see its siblings, so this cannot
    /// tell whether the target edge already has an operand. Retargeting onto an
    /// occupied edge produces a phi with two operands for one predecessor, which
    /// the verifier rejects as
    /// [`VerifierError::DuplicatePhiOperand`](crate::analysis::verifier::VerifierError::DuplicatePhiOperand).
    /// Prefer
    /// [`SsaEditor::replace_phi_predecessor`](crate::ir::function::SsaEditor::replace_phi_predecessor),
    /// which checks for the collision and reconciles it.
    pub fn set_predecessor(&mut self, predecessor: usize) {
        self.predecessor = predecessor as u32;
    }
}

impl fmt::Display for PhiOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.edge {
            PhiEdge::Normal => write!(f, "{} from B{}", self.value, self.predecessor),
            PhiEdge::Exceptional => write!(f, "{} thrown in B{}", self.value, self.predecessor),
        }
    }
}

/// A phi node that merges values from multiple control flow predecessors.
///
/// Placed at the beginning of basic blocks with multiple predecessors. Each phi
/// node has one [`PhiOperand`] per predecessor, linking the predecessor block
/// to the SSA variable providing the value along that edge.
///
/// # Invariants
///
/// - Each phi has at most one operand per predecessor (enforced by `set_operand`)
/// - The result variable is defined by this phi node at the block entry
/// - Phi nodes appear before all regular instructions in a block
///
/// # Examples
///
/// ```rust
/// use analyssa::ir::{phi::{PhiNode, PhiOperand}, variable::{SsaVarId, VariableOrigin}};
///
/// let v1 = SsaVarId::from_index(0);
/// let v2 = SsaVarId::from_index(1);
/// let result = SsaVarId::from_index(2);
/// let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
/// phi.add_operand(PhiOperand::new(v1, 1));
/// phi.add_operand(PhiOperand::new(v2, 2));
/// assert_eq!(phi.result(), result);
/// assert_eq!(phi.operand_count(), 2);
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhiNode {
    /// The SSA variable that this phi node defines (its result).
    /// This is the variable that subsequent instructions in this block
    /// should use to read the merged value.
    result: SsaVarId,

    /// The original variable this phi node merges - identifies which
    /// CIL argument or local variable this phi corresponds to. Used
    /// during SSA construction and rebuild for rename grouping.
    origin: VariableOrigin,

    /// Operands, one per predecessor block, each linking a predecessor
    /// to the SSA variable providing the value along that edge.
    operands: Vec<PhiOperand>,
}

impl Clone for PhiNode {
    fn clone(&self) -> Self {
        let Self {
            result,
            origin,
            operands,
        } = self;
        Self {
            result: *result,
            origin: *origin,
            operands: operands.clone(),
        }
    }

    /// Overwrites `self` from `source`, **reusing the existing `operands`
    /// allocation**.
    ///
    /// Hand-written for the same reason as
    /// [`SsaFunction::clone_from`](crate::ir::function::SsaFunction): the
    /// derived `Clone` supplies only the default `clone_from`
    /// (`*self = source.clone()`), which drops and reallocates the operand
    /// vector on every call. Specializing here lets a whole-function snapshot's
    /// capacity reuse reach each phi's operand list rather than stopping at the
    /// owning block's `phi_nodes` spine.
    ///
    /// The exhaustive destructure is deliberate: adding a field makes this fail
    /// to compile rather than silently producing a stale-in-one-field clone.
    fn clone_from(&mut self, source: &Self) {
        let Self {
            result,
            origin,
            operands,
        } = source;
        self.result = *result;
        self.origin = *origin;
        self.operands.clone_from(operands);
    }
}

impl PhiNode {
    /// Creates a new phi node for the given variable origin.
    ///
    /// The phi node is created with no operands - they must be added
    /// during SSA construction as predecessor blocks are processed.
    ///
    /// # Arguments
    ///
    /// * `result` - The SSA variable that this phi node defines
    /// * `origin` - The original variable (Argument or Local) this phi merges
    #[must_use]
    pub fn new(result: SsaVarId, origin: VariableOrigin) -> Self {
        Self {
            result,
            origin,
            operands: Vec::new(),
        }
    }

    /// Creates a new phi node with pre-allocated operand capacity.
    ///
    /// Use this when the number of predecessors is known in advance
    /// to avoid reallocations.
    ///
    /// # Arguments
    ///
    /// * `result` - The SSA variable that this phi node defines
    /// * `origin` - The original variable this phi merges
    /// * `predecessor_count` - Expected number of predecessor blocks
    #[must_use]
    pub fn with_capacity(
        result: SsaVarId,
        origin: VariableOrigin,
        predecessor_count: usize,
    ) -> Self {
        Self {
            result,
            origin,
            operands: Vec::with_capacity(predecessor_count),
        }
    }

    /// Returns the SSA variable defined by this phi node.
    #[must_use]
    pub const fn result(&self) -> SsaVarId {
        self.result
    }

    /// Returns the original variable origin this phi merges.
    #[must_use]
    pub const fn origin(&self) -> VariableOrigin {
        self.origin
    }

    /// Sets the SSA variable defined by this phi node.
    ///
    /// Used during SSA construction when renaming variables.
    pub fn set_result(&mut self, var: SsaVarId) {
        self.result = var;
    }

    /// Returns the operands of this phi node.
    #[must_use]
    pub fn operands(&self) -> &[PhiOperand] {
        &self.operands
    }

    /// Returns a mutable reference to the operands.
    pub fn operands_mut(&mut self) -> &mut Vec<PhiOperand> {
        &mut self.operands
    }

    /// Adds an operand to this phi node.
    ///
    /// **Unchecked**: this does not verify that the operand's predecessor is a
    /// real CFG predecessor, nor that no operand already names it. A phi with two
    /// operands on one edge has no defined meaning - [`Self::operand_from`]
    /// returns the first and discards the rest, while SCCP meets them all and
    /// yields Bottom - and the verifier reports it as
    /// [`VerifierError::DuplicatePhiOperand`]. Callers redirecting an existing
    /// edge should use
    /// [`SsaEditor::replace_phi_predecessor`](crate::ir::function::SsaEditor::replace_phi_predecessor),
    /// which reconciles the collision instead of creating one.
    ///
    /// [`VerifierError::DuplicatePhiOperand`]: crate::analysis::verifier::VerifierError::DuplicatePhiOperand
    ///
    /// # Arguments
    ///
    /// * `operand` - The phi operand to add
    pub fn add_operand(&mut self, operand: PhiOperand) {
        self.operands.push(operand);
    }

    /// Returns the number of operands.
    #[must_use]
    pub fn operand_count(&self) -> usize {
        self.operands.len()
    }

    /// Returns `true` if this phi node has no operands.
    ///
    /// A phi node with no operands is incomplete and should not appear
    /// in a fully-constructed SSA form.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operands.is_empty()
    }

    /// Finds the operand coming from the specified predecessor block.
    ///
    /// # Arguments
    ///
    /// * `predecessor` - The block index to look up
    ///
    /// # Returns
    ///
    /// The phi operand if found, or `None` if no operand comes from that predecessor.
    #[must_use]
    pub fn operand_from(&self, predecessor: usize) -> Option<&PhiOperand> {
        self.operands
            .iter()
            .find(|op| !op.is_exceptional() && op.predecessor() == predecessor)
    }

    /// Returns the operands naming terminator edges.
    pub fn normal_operands(&self) -> impl Iterator<Item = &PhiOperand> + '_ {
        self.operands.iter().filter(|op| !op.is_exceptional())
    }

    /// Returns the operands naming values a throw may bring here.
    pub fn exceptional_operands(&self) -> impl Iterator<Item = &PhiOperand> + '_ {
        self.operands.iter().filter(|op| op.is_exceptional())
    }

    /// Returns every operand naming `block`, of either kind.
    ///
    /// # Arguments
    ///
    /// * `block` - The block the operands name.
    pub fn operands_from(&self, block: usize) -> impl Iterator<Item = &PhiOperand> + '_ {
        self.operands
            .iter()
            .filter(move |op| op.predecessor() == block)
    }

    /// Adds an operand for a value a throw in `source` may bring here, unless
    /// that pairing is already recorded.
    ///
    /// A protected block contributes one operand per value the merged storage
    /// may hold while it runs, so several operands may name the same block -
    /// but the same `(block, value)` twice says nothing new.
    ///
    /// # Arguments
    ///
    /// * `source` - The protected block a throw may happen in.
    /// * `value` - The value the storage may hold at that point.
    ///
    /// # Returns
    ///
    /// `true` when the operand was added.
    pub fn add_exceptional_operand(&mut self, source: usize, value: SsaVarId) -> bool {
        let known = self
            .operands
            .iter()
            .any(|op| op.is_exceptional() && op.predecessor() == source && op.value == value);
        if known {
            return false;
        }
        self.operands.push(PhiOperand::exceptional(value, source));
        true
    }

    /// Returns all the SSA variables used by this phi node.
    ///
    /// This is useful for building def-use chains and liveness analysis.
    pub fn used_variables(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.operands.iter().map(|op| op.value)
    }

    /// Retains only the operands the predicate accepts.
    ///
    /// Takes the whole operand rather than its predecessor: an exceptional
    /// operand and a normal one can name the same block and mean different
    /// things, so a rule stated over block indices alone cannot separate them.
    pub fn retain_operands<F: Fn(&PhiOperand) -> bool>(&mut self, pred: F) {
        self.operands.retain(|op| pred(op));
    }

    /// Sets the origin of this phi node.
    ///
    /// This is used during local variable optimization to update indices
    /// after unused locals are removed.
    pub fn set_origin(&mut self, origin: VariableOrigin) {
        self.origin = origin;
    }

    /// Replaces `old` with `new` in the operands of kind `edge` naming
    /// `predecessor`.
    ///
    /// Only operands that name that edge, that kind and that value change:
    /// unlike [`Self::set_operand`], this never adds an operand and never
    /// rewrites an operand carrying some other value on the same edge. The
    /// kind is the caller's to state, because a block can owe a phi both: the
    /// value it hands over when its terminator transfers here, and the values
    /// a throw in it may bring. A substitution that reached only the first
    /// would leave the handler reading a value whose definition the caller
    /// then removes.
    ///
    /// An exceptional operand is keyed by its value as well as its block
    /// ([`Self::add_exceptional_operand`]), so where the block already lists
    /// `new` the operand naming `old` is dropped rather than rewritten into a
    /// second copy of the same claim.
    ///
    /// # Arguments
    ///
    /// * `predecessor` - The block whose operands are rewritten.
    /// * `edge` - Which of that block's operands: the transfer's, or a throw's.
    /// * `old` - The value being replaced.
    /// * `new` - The value substituted for it.
    ///
    /// # Returns
    ///
    /// How many operands now name `new` instead, and how many were dropped
    /// because one naming `new` was already there.
    pub fn replace_operand_value(
        &mut self,
        predecessor: usize,
        edge: PhiEdge,
        old: SsaVarId,
        new: SsaVarId,
    ) -> OperandReplacement {
        let names = |operand: &PhiOperand, value: SsaVarId| {
            operand.edge == edge && operand.predecessor() == predecessor && operand.value == value
        };
        let mut replacement = OperandReplacement::default();
        if edge == PhiEdge::Exceptional && self.operands.iter().any(|operand| names(operand, new)) {
            let before = self.operands.len();
            self.operands.retain(|operand| !names(operand, old));
            replacement.merged = before.saturating_sub(self.operands.len());
            return replacement;
        }
        for operand in &mut self.operands {
            if names(operand, old) {
                operand.value = new;
                replacement.rewritten = replacement.rewritten.saturating_add(1);
            }
        }
        replacement
    }

    /// Sets the operand value for a specific predecessor.
    ///
    /// If an operand from that predecessor already exists, it is updated.
    /// Otherwise, a new operand is added.
    ///
    /// # Arguments
    ///
    /// * `predecessor` - The predecessor block index
    /// * `value` - The SSA variable value from that predecessor
    pub fn set_operand(&mut self, predecessor: usize, value: SsaVarId) {
        if let Some(existing) = self
            .operands
            .iter_mut()
            .find(|op| !op.is_exceptional() && op.predecessor() == predecessor)
        {
            existing.value = value;
        } else {
            self.operands.push(PhiOperand::new(value, predecessor));
        }
    }
}

impl fmt::Display for PhiNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = phi(", self.result)?;
        for (i, operand) in self.operands.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{operand}")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A replacement names the kind of operand it rewrites.** A block owes
    /// the phi the value its transfer hands over and the values a throw in it
    /// may bring; rewriting one leaves the other. An exceptional operand
    /// replaced by a value the block already lists for a throw merges into it.
    #[test]
    fn a_replacement_names_the_kind_of_operand_it_rewrites() {
        let (result, old, new) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let phi = || {
            let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
            phi.add_operand(PhiOperand::new(old, 1));
            phi.add_exceptional_operand(1, old);
            phi
        };
        let operands = |phi: &PhiNode| {
            phi.operands()
                .iter()
                .map(|operand| (operand.edge(), operand.value()))
                .collect::<Vec<_>>()
        };

        let mut normal = phi();
        let done = normal.replace_operand_value(1, PhiEdge::Normal, old, new);
        assert_eq!((done.rewritten, done.merged), (1, 0));
        assert_eq!(
            operands(&normal),
            vec![(PhiEdge::Normal, new), (PhiEdge::Exceptional, old)]
        );

        let mut thrown = phi();
        let done = thrown.replace_operand_value(1, PhiEdge::Exceptional, old, new);
        assert_eq!((done.rewritten, done.merged), (1, 0));
        assert_eq!(
            operands(&thrown),
            vec![(PhiEdge::Normal, old), (PhiEdge::Exceptional, new)]
        );

        let mut merged = phi();
        merged.add_exceptional_operand(1, new);
        let done = merged.replace_operand_value(1, PhiEdge::Exceptional, old, new);
        assert_eq!((done.rewritten, done.merged), (0, 1));
        assert_eq!(
            operands(&merged),
            vec![(PhiEdge::Normal, old), (PhiEdge::Exceptional, new)]
        );
    }

    /// A phi operand round-trips the value and predecessor it was built with.
    /// **A block's operands are of either kind.** A block that hands the phi
    /// a value at its terminator and another at a throw names two operands;
    /// [`PhiNode::operand_from`] answers the terminator's alone.
    #[test]
    fn a_block_names_operands_of_either_kind() {
        let (result, handed, thrown) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(handed, 1));
        phi.add_exceptional_operand(1, thrown);
        phi.add_operand(PhiOperand::new(handed, 2));
        let from = |block| {
            phi.operands_from(block)
                .map(|operand| (operand.edge(), operand.value()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            from(1),
            vec![(PhiEdge::Normal, handed), (PhiEdge::Exceptional, thrown)]
        );
        assert_eq!(from(2), vec![(PhiEdge::Normal, handed)]);
        assert!(from(3).is_empty());
        assert_eq!(phi.operand_from(1).map(PhiOperand::value), Some(handed));
    }

    #[test]
    fn test_phi_operand_creation() {
        let v = SsaVarId::from_index(0);
        let operand = PhiOperand::new(v, 2);
        assert_eq!(operand.value(), v);
        assert_eq!(operand.predecessor(), 2);
    }

    /// A phi operand renders as `<value> from B<predecessor>`.
    #[test]
    fn test_phi_operand_display() {
        let operand = PhiOperand::new(SsaVarId::from_index(3), 1);
        assert_eq!(format!("{operand}"), "v3 from B1");
    }

    /// A freshly created phi carries its result and origin and starts empty.
    #[test]
    fn test_phi_node_creation() {
        let result = SsaVarId::from_index(0);
        let phi = PhiNode::new(result, VariableOrigin::Local(0));
        assert_eq!(phi.result(), result);
        assert_eq!(phi.origin(), VariableOrigin::Local(0));
        assert!(phi.is_empty());
        assert_eq!(phi.operand_count(), 0);
    }

    /// Pre-allocating operand capacity does not add operands: the phi is still
    /// empty and reports its result and origin.
    #[test]
    fn test_phi_node_with_capacity() {
        let result = SsaVarId::from_index(0);
        let phi = PhiNode::with_capacity(result, VariableOrigin::Argument(1), 3);
        assert_eq!(phi.result(), result);
        assert_eq!(phi.origin(), VariableOrigin::Argument(1));
        assert!(phi.is_empty());
    }

    /// Added operands are appended in order, each preserving its value and
    /// predecessor.
    #[test]
    fn test_phi_node_add_operands() {
        let result = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));

        phi.add_operand(PhiOperand::new(v1, 0));
        phi.add_operand(PhiOperand::new(v2, 1));

        assert!(!phi.is_empty());
        assert_eq!(phi.operand_count(), 2);

        let ops = phi.operands();
        assert_eq!(ops[0].value(), v1);
        assert_eq!(ops[0].predecessor(), 0);
        assert_eq!(ops[1].value(), v2);
        assert_eq!(ops[1].predecessor(), 1);
    }

    /// `operand_from` returns the operand on a matching predecessor edge and
    /// `None` for edges the phi has no operand on.
    #[test]
    fn test_phi_node_operand_from() {
        let result = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v3 = SsaVarId::from_index(2);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(v1, 2));
        phi.add_operand(PhiOperand::new(v3, 4));

        assert!(phi.operand_from(2).is_some());
        assert_eq!(phi.operand_from(2).unwrap().value(), v1);

        assert!(phi.operand_from(4).is_some());
        assert_eq!(phi.operand_from(4).unwrap().value(), v3);

        assert!(phi.operand_from(0).is_none());
        assert!(phi.operand_from(99).is_none());
    }

    /// `set_operand` on an unseen predecessor adds a fresh operand.
    #[test]
    fn test_phi_node_set_operand_new() {
        let result = SsaVarId::from_index(0);
        let v10 = SsaVarId::from_index(1);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));

        phi.set_operand(1, v10);
        assert_eq!(phi.operand_count(), 1);
        assert_eq!(phi.operand_from(1).unwrap().value(), v10);
    }

    /// `set_operand` on an existing predecessor updates its value in place
    /// rather than adding a duplicate operand.
    #[test]
    fn test_phi_node_set_operand_update() {
        let result = SsaVarId::from_index(0);
        let v10 = SsaVarId::from_index(1);
        let v20 = SsaVarId::from_index(2);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));

        phi.set_operand(1, v10);
        phi.set_operand(1, v20); // Update existing

        assert_eq!(phi.operand_count(), 1);
        assert_eq!(phi.operand_from(1).unwrap().value(), v20);
    }

    /// `used_variables` yields every operand's value variable.
    #[test]
    fn test_phi_node_used_variables() {
        let result = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let v2 = SsaVarId::from_index(2);
        let v3 = SsaVarId::from_index(3);
        let mut phi = PhiNode::new(result, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(v1, 0));
        phi.add_operand(PhiOperand::new(v2, 1));
        phi.add_operand(PhiOperand::new(v3, 2));

        let used: Vec<_> = phi.used_variables().collect();
        assert_eq!(used.len(), 3);
        assert!(used.contains(&v1));
        assert!(used.contains(&v2));
        assert!(used.contains(&v3));
    }

    /// A multi-operand phi renders as `<result> = phi(<op>, <op>)`.
    #[test]
    fn test_phi_node_display() {
        let mut phi = PhiNode::new(SsaVarId::from_index(5), VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(SsaVarId::from_index(1), 0));
        phi.add_operand(PhiOperand::new(SsaVarId::from_index(2), 1));

        let display = format!("{phi}");
        assert_eq!(display, "v5 = phi(v1 from B0, v2 from B1)");
    }

    /// An operand-less phi renders with empty parentheses.
    #[test]
    fn test_phi_node_display_empty() {
        let phi = PhiNode::new(SsaVarId::from_index(3), VariableOrigin::Local(0));
        assert_eq!(format!("{phi}"), "v3 = phi()");
    }

    /// A single-operand phi renders without an operand separator.
    #[test]
    fn test_phi_node_display_single_operand() {
        let mut phi = PhiNode::new(SsaVarId::from_index(7), VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(SsaVarId::from_index(4), 2));
        assert_eq!(format!("{phi}"), "v7 = phi(v4 from B2)");
    }
}
