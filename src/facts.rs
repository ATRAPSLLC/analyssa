//! Host data carried inside the IR.
//!
//! A host knows things about a value, an instruction or a function that this
//! crate does not: which machine register a value came out of, what a call's
//! arguments were, that a variable is known to be a particular constant. Kept
//! in the host's own table, those facts go stale silently - the passes renumber
//! variables, move instructions and roll edits back, and a table keyed by
//! `SsaVarId` or by position then describes whichever value inherited the key.
//!
//! Facts declared here live *in* the IR instead. They are cloned with the
//! function, restored with it when a pass group rolls back, serialized with it,
//! and - this is the part a side table cannot do - carried across the moments
//! where one value becomes another:
//!
//! - a **derivation**, where a pass mints a value to stand for an existing one
//!   (renaming during reconstruction, a loop split, a value reconstructed on a
//!   threaded path);
//! - a **join**, where a phi merges several values into one;
//! - a **substitution**, where one value replaces another and the replaced
//!   value's uses become the survivor's (global value numbering, copy
//!   propagation, a phi with one distinct operand).
//!
//! Each is a question only the host can answer - does a register origin
//! survive a copy? does a known constant survive a merge? - so the traits state
//! the question and the host answers it. A host with nothing to carry uses
//! `()`, which implements all three and costs nothing: the calls monomorphize
//! away and the fields are zero-sized.

use std::fmt::Debug;

/// Why a value was derived from another.
///
/// Passed to [`VariableFacts::derive`] so a host can distinguish a rename,
/// which preserves everything about a value, from a split or a reconstruction,
/// which may not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Derivation {
    /// SSA reconstruction minted a new version of the same storage.
    ///
    /// The value is the same value under a new id: whatever was true of the
    /// parent is true of it.
    Rename,
    /// A value the function entry supplies, standing in for a read whose
    /// definition reconstruction could not find.
    EntryValue,
    /// Loop canonicalization split a value across a preheader or a latch.
    LoopSplit,
    /// A block was duplicated and this value is the copy's version of one in
    /// the original.
    BlockDuplication,
    /// A value rebuilt on a rerouted path, standing for the one the bypassed
    /// path produced.
    Reconstruction,
    /// A phi was removed and its result re-expressed as a plain value.
    PhiDemotion,
}

/// Why several values were joined into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Join {
    /// A phi placed where control flow merges.
    PlacedPhi,
    /// A phi at a handler or filter entry, merging what the protected region
    /// may have produced before the throw.
    ExceptionRootPhi,
    /// A loop header phi, merging the value entering the loop with the one
    /// coming back around.
    LoopHeaderPhi,
}

/// Why one value replaced another.
///
/// The replaced value is about to have no uses left, so whatever the host knew
/// about it has to be folded into the survivor or dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Substitution {
    /// Global value numbering proved the two compute the same thing.
    Gvn,
    /// Copy propagation forwarded a copy's source in place of its result.
    CopyPropagation,
    /// A phi with one distinct operand was folded away.
    TrivialPhi,
    /// A load of a slot was resolved to the value the slot holds.
    SlotLoadCollapse,
    /// SSA reconstruction mapped an old value onto its renamed version.
    Rename,
    /// The host replaced the value itself.
    Host,
    /// An algebraic identity proved the two compute the same thing: a half
    /// of a value summed in halves, read as that half of the sum.
    Algebraic,
}

/// Why an instruction's operation was rewritten in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rewrite {
    /// The operation folded to a constant.
    ConstantFold,
    /// An algebraic identity rewrote it.
    Algebraic,
    /// Strength reduction replaced it with a cheaper operation.
    StrengthReduction,
    /// It became a placeholder, pending removal.
    Nop,
    /// The host rewrote it.
    Host,
    /// Construction restated it for the type its operand settled on.
    Settled,
}

/// Host data carried on every SSA value.
///
/// The default policy is the conservative one for a fact that describes *the
/// value*: a derived value inherits it, a join starts empty rather than
/// guessing which operand to believe, and a substitution leaves the survivor's
/// facts alone. A host overrides whichever of the three its facts have a better
/// answer for.
pub trait VariableFacts: Clone + Debug + Default {
    /// The facts a value minted to stand for `parent` starts with.
    ///
    /// # Arguments
    ///
    /// * `parent` - The value the new one was derived from.
    /// * `cause` - What derived it.
    fn derive(parent: &Self, cause: Derivation) -> Self {
        let _ = cause;
        parent.clone()
    }

    /// The facts a phi result starts with, given its operands'.
    ///
    /// # Arguments
    ///
    /// * `operands` - The facts of the values being merged, in operand order.
    /// * `cause` - What kind of merge this is.
    fn join(operands: &mut dyn Iterator<Item = &Self>, cause: Join) -> Self {
        let _ = (operands, cause);
        Self::default()
    }

    /// The facts an existing phi result has once its operands are known again.
    ///
    /// Called on the result, with its operands' facts, wherever the crate
    /// refreshes phi facts after an edit. The default recomputes the join and
    /// discards what the result held, which is right for a fact that is only
    /// ever a function of the operands. A host whose facts also record what the
    /// phi was placed *for* - the storage it merges, which an operand bound to
    /// several locations cannot say - overrides it to keep what it knows.
    ///
    /// # Arguments
    ///
    /// * `operands` - The facts of the values being merged, in operand order.
    /// * `cause` - What kind of merge this is.
    fn rejoin(&self, operands: &mut dyn Iterator<Item = &Self>, cause: Join) -> Self {
        Self::join(operands, cause)
    }

    /// Folds the facts of a value that `self` now stands for.
    ///
    /// Called on the survivor, with the facts of the value whose uses have just
    /// become the survivor's.
    ///
    /// # Arguments
    ///
    /// * `replaced` - The facts of the value being replaced.
    /// * `cause` - What replaced it.
    fn absorb(&mut self, replaced: &Self, cause: Substitution) {
        let _ = (replaced, cause);
    }

    /// Folds the facts of the parts `self` is read from where `self` is
    /// kept whole.
    ///
    /// The storage `start` was read from is where this value is kept, at its
    /// own width: a doubleword stored to a frame cell and read back a word at
    /// a time is the cell's value, and its words are parts of it; one load of
    /// a doubleword that replaces its word loads reads it where the first of
    /// them read; a double-width sum composed of two halves is kept where its
    /// halves were. `rest` is the other half, where the whole is composed of
    /// two values: the value is kept whole only if that half was kept too, a
    /// sign-extended word is not. The parts are parts of this value, not this
    /// value, so this is no substitution: this value takes where its start
    /// was read from, and nothing the parts held besides. Called on the
    /// whole.
    ///
    /// # Arguments
    ///
    /// * `start` - The facts of the part at the whole's lowest address.
    /// * `rest` - The facts of the other half, where the whole is two values;
    ///   `None` where one stored value was read in part.
    fn hold_whole(&mut self, start: &Self, rest: Option<&Self>) {
        let _ = (start, rest);
    }
}

/// Host data carried on every instruction.
pub trait InstructionFacts: Clone + Debug + Default {
    /// The facts a copy of this instruction starts with.
    ///
    /// The default copies them, which is right for a fact about *what the
    /// instruction is*. A host whose facts identify a program point overrides
    /// it, because a copy is a different point.
    fn duplicate(&self) -> Self {
        self.clone()
    }

    /// Updates the facts after the instruction's operation is rewritten.
    ///
    /// The instruction is the same instruction - same identity, same position -
    /// computing something else, so a fact about its operation may no longer
    /// hold while a fact about where it came from still does.
    ///
    /// # Arguments
    ///
    /// * `cause` - What rewrote it.
    fn rewritten(&mut self, cause: Rewrite) {
        let _ = cause;
    }

    /// Folds in the facts of an instruction whose computation this one now
    /// holds.
    ///
    /// Called when the instruction stops reading a value it read and computes
    /// what that value contributed itself: an operation rewritten to no longer
    /// read an operand (a fold to a constant, an identity that drops one),
    /// with the facts of the operand's definition; and a reader whose operand
    /// is replaced by a constant, or by a value computed apart from the one it
    /// replaces, with the facts of the replaced value's definition. A fact
    /// about where the computation came from (the host's rows) is the one to
    /// gather; a fact about the operation itself is not the absorbed one's.
    ///
    /// # Arguments
    ///
    /// * `absorbed` - The facts of the instruction whose computation this one
    ///   now holds.
    fn absorb_computation(&mut self, absorbed: &Self) {
        let _ = absorbed;
    }

    /// Folds in the facts of an instruction this one stands for: the same
    /// work at another point the program goes, which is deleted as this one
    /// does it on that path too. Two arms of a test that do the same are
    /// kept as one ([`crate::passes::moot`], [`crate::passes::arms`]): the
    /// kept arm's call is also the dropped arm's call site.
    ///
    /// Apart from [`Self::absorb_computation`]: a twin is a point of the
    /// program in its own right, where an absorbed computation is part of
    /// this one's. The default gathers it as an absorbed computation.
    ///
    /// # Arguments
    ///
    /// * `twin` - The facts of the instruction this one stands for.
    fn absorb_twin(&mut self, twin: &Self) {
        self.absorb_computation(twin);
    }
}

/// Host data carried on the function as a whole.
///
/// Cloned, rolled back and serialized with the function, which is what
/// distinguishes it from a table the host keeps beside one.
pub trait FunctionFacts: Clone + Debug + Default {}

impl VariableFacts for () {}
impl InstructionFacts for () {}
impl FunctionFacts for () {}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    /// A fact that survives a rename, agrees or gives up on a join, and takes
    /// the replaced value's answer when it has none of its own - the shape a
    /// host's storage-origin lane has.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct Origin(Option<u8>);

    impl VariableFacts for Origin {
        fn derive(parent: &Self, cause: Derivation) -> Self {
            match cause {
                Derivation::Rename | Derivation::LoopSplit => parent.clone(),
                _ => Self::default(),
            }
        }

        fn join(operands: &mut dyn Iterator<Item = &Self>, _cause: Join) -> Self {
            let mut agreed: Option<Option<u8>> = None;
            for operand in operands {
                match agreed {
                    None => agreed = Some(operand.0),
                    Some(seen) if seen == operand.0 => {}
                    Some(_) => return Self::default(),
                }
            }
            Self(agreed.flatten())
        }

        fn absorb(&mut self, replaced: &Self, _cause: Substitution) {
            if self.0.is_none() {
                self.0 = replaced.0;
            }
        }
    }

    /// A host that carries nothing answers every question without a policy of
    /// its own, which is what makes `()` the zero-cost default.
    #[test]
    fn the_unit_policy_is_total() {
        let () = <() as VariableFacts>::derive(&(), Derivation::Rename);
        let mut empty = iter::empty::<&()>();
        let () = <() as VariableFacts>::join(&mut empty, Join::PlacedPhi);
        let mut survivor = ();
        survivor.absorb(&(), Substitution::Gvn);
        let () = <() as InstructionFacts>::duplicate(&());
        let mut rewritten = ();
        rewritten.rewritten(Rewrite::ConstantFold);
    }

    #[test]
    fn a_derivation_carries_what_its_cause_preserves() {
        let parent = Origin(Some(7));
        assert_eq!(Origin::derive(&parent, Derivation::Rename), Origin(Some(7)));
        assert_eq!(
            Origin::derive(&parent, Derivation::BlockDuplication),
            Origin(None),
            "a copy of a value is not that value's storage"
        );
    }

    #[test]
    fn a_join_agrees_or_gives_up() {
        let agree = [Origin(Some(3)), Origin(Some(3))];
        assert_eq!(
            Origin::join(&mut agree.iter(), Join::PlacedPhi),
            Origin(Some(3))
        );
        let disagree = [Origin(Some(3)), Origin(Some(4))];
        assert_eq!(
            Origin::join(&mut disagree.iter(), Join::PlacedPhi),
            Origin(None)
        );
    }

    #[test]
    fn a_survivor_keeps_its_own_answer() {
        let mut survivor = Origin(Some(1));
        survivor.absorb(&Origin(Some(2)), Substitution::Gvn);
        assert_eq!(survivor, Origin(Some(1)));

        let mut unknown = Origin(None);
        unknown.absorb(&Origin(Some(2)), Substitution::CopyPropagation);
        assert_eq!(unknown, Origin(Some(2)));
    }
}
