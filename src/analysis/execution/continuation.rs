//! Explicit state mappings for a real transfer between immutable source bodies.

use crate::{
    analysis::execution::CallStack,
    ir::{
        function::{SsaFunction, Widening},
        variable::SsaVarId,
    },
    target::Target,
};

/// The actual source location of an indirect branch discovery request.
pub struct BranchSite<'a, T: Target> {
    /// Source body executing the branch.
    pub body: &'a SsaFunction<T>,
    /// Fully qualified identity of that body, when established.
    pub method: Option<&'a T::MethodRef>,
    /// Source block containing the branch.
    pub block: usize,
    /// Actual terminator position in that block.
    pub instruction: usize,
}

/// Where control enters a block of the running activation's body, the
/// point a write that changed the code there is met.
pub struct EntrySite<'a, T: Target> {
    /// Source body the activation runs.
    pub body: &'a SsaFunction<T>,
    /// Fully qualified identity of that body, when established.
    pub method: Option<&'a T::MethodRef>,
    /// The block control leaves.
    pub from: usize,
    /// Whether control leaves `from` by an edge of its terminator, and so
    /// runs what that edge passes through
    /// ([`BlockProvenance::passed_on`](crate::ir::block::BlockProvenance::passed_on)),
    /// rather than raising from inside it.
    pub edge: bool,
    /// The block control enters.
    pub to: usize,
}

/// What control entering a block meets of code a write changed
/// ([`ExecutionHost::entered_code_change`](crate::analysis::execution::ExecutionHost::entered_code_change)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnteredCode {
    /// Nothing it runs on the way changed.
    Unchanged,
    /// Code it runs on the way changed. The activation goes on from this
    /// host block ([`BlockProvenance`](crate::ir::block::BlockProvenance)),
    /// or, where the host can name none, from nowhere.
    Changed(Option<u64>),
}

/// One unchanged physical representation carried into a new source namespace.
///
/// The executor checks both values and their layouts. A binding neither computes
/// new bits nor turns a destination definition into a function input; the one
/// conversion it applies is the widening a machine state says its value takes
/// to fill the register
/// ([`StatedRegister`](crate::ir::function::StatedRegister)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuationBinding {
    /// Value in the currently executing source body.
    pub from: SsaVarId,
    /// Entry value in the body selected for the actual source continuation.
    pub into: SsaVarId,
    /// How `from` fills the register `into` enters with, where it is
    /// narrower.
    pub widened: Option<Widening>,
}

/// A value the code before a continuation holds in SSA that belongs in the
/// activation's frame: a stack slot its lift held as a value, which the code
/// it continues in reads from memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameWrite {
    /// Where it belongs, relative to the stack pointer the activation was
    /// entered with.
    pub offset: i64,
    /// Value in the currently executing source body.
    pub from: SsaVarId,
    /// How `from` fills the slot, where it is narrower.
    pub widened: Option<Widening>,
}

/// An established source continuation and its source-to-source input correspondence.
///
/// This continues one physical activation. It does not invoke a language call,
/// stage arguments, adjust the stack, release objects, or rerun a prefix. The
/// host supplies the actual source body through ordinary body lookup. Every
/// machine component needed by that body must have an explicit justified input;
/// omitted inputs stay unknown in a fresh source namespace and may alias the
/// current activation's preserved stack.
#[derive(Debug, Clone)]
pub struct BranchContinuation<T: Target> {
    /// Fully qualified source version at the actual target.
    pub method: T::MethodRef,
    /// Unchanged values passed to the selected body's entry.
    pub bindings: Vec<ContinuationBinding>,
    /// The stack pointer at the transfer, where the source segment moved it:
    /// the caller's value of a stack pointer, moved by an adjustment (its
    /// entry stack pointer and the depth the lift measured there).
    ///
    /// `None` promises the source body's entry stack coordinate is unchanged.
    /// A supplied pointer must retain the current stack object and target width;
    /// another root or an unknown value cannot create new stack backing.
    pub stack: Option<CallStack>,
    /// Values the code before holds that belong in the frame, written there
    /// before the code it continues in runs.
    pub frame: Vec<FrameWrite>,
}
