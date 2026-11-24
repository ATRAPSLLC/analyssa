//! What a run finds wrong with the program, while it goes on.
//!
//! A run that reaches storage past the extent of its object, storage whose
//! lifetime ended, a release of storage that was no live block of its
//! allocator, or bits no write had reached, has found something about the
//! program: a [`Finding`], recorded at the instruction that did it. The run
//! does not end there. It goes on with the machine's own semantics (the write
//! lands where the machine puts it; a read of ended or never-written storage
//! is unknown), so every other fact it makes is still made, and the findings
//! are a product of the run like those facts.
//!
//! A finding carries its evidence: the bytes the access reached, the object
//! it was judged against (the extent the host states and where it comes
//! from, or the heap block with the call that made it and the ones that
//! ended it), and its witness, the decisions of the path that found it.

use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{Hash, Hasher},
    ptr,
    sync::{Arc, Mutex},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::analysis::execution::{
    facts::{ExtentSource, StorageObject},
    marks::UnwrittenUseKind,
    memory::AllocatorFamily,
    value::{Address, Root},
};

/// What a run found wrong with the program at one instruction, with its
/// evidence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Finding {
    /// The body, as the execution numbers the bodies it entered
    /// ([`StaticFacts::body_methods`](crate::analysis::execution::StaticFacts::body_methods)).
    pub body: usize,
    /// The block of the instruction.
    pub block: usize,
    /// The instruction's index in its block.
    pub instruction: usize,
    /// What it found.
    pub kind: FindingKind,
    /// The bytes the access reached: its first and how many; a release's
    /// address, reaching none. `None` for a use of never-written bits.
    pub access: Option<(Address, u64)>,
    /// The object the access or release was judged against, where one is.
    pub object: Option<FoundObject>,
    /// How the path that found it got there.
    pub witness: Witness,
}

/// The object a finding was judged against.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FoundObject {
    /// An object outside the heap, as the host states it, in `root`.
    Stated {
        /// The root its offsets are in.
        root: Root,
        /// Its extent and where the extent comes from.
        object: StorageObject,
    },
    /// A heap block.
    Block {
        /// The block's root.
        root: Root,
        /// The bytes its allocator made it with, on every path.
        size: u64,
        /// The most bytes it holds on any path, which a past-extent finding
        /// is judged against; `None` where a path's allocator did not state
        /// its size.
        extent: Option<u64>,
        /// The allocator that made it, where every path agrees on one.
        family: Option<AllocatorFamily>,
        /// The call that made it, `(block, instruction)` in the outermost
        /// activation, where every path agrees on one.
        allocated_at: Option<(usize, usize)>,
        /// The stated deallocations that ended it on some path, each
        /// `(block, instruction)` in the outermost activation.
        ended_at: BTreeSet<(usize, usize)>,
    },
}

/// One decision the path took that the program alone did not make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Decision {
    /// At an undecided branch ending `block`, the path took `successor`.
    Branch {
        /// The body, as the execution numbers its bodies.
        body: usize,
        /// The branching block.
        block: usize,
        /// The successor taken.
        successor: usize,
    },
    /// At a modelled call with alternatives, the path took `alternative`.
    Alternative {
        /// The body, as the execution numbers its bodies.
        body: usize,
        /// The call, `(block, instruction)`.
        site: (usize, usize),
        /// The alternative taken.
        alternative: usize,
    },
    /// An instruction raised and the path entered `handler`.
    Raise {
        /// The body, as the execution numbers its bodies.
        body: usize,
        /// The raising instruction, `(block, instruction)`.
        site: (usize, usize),
        /// The handler entered.
        handler: usize,
    },
}

/// The witness of a finding: the decisions of the path that found it, first
/// to last, and whether the path is a join of several, whose decisions are
/// those of one of them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Witness {
    /// The decisions, first to last.
    pub decisions: Vec<Decision>,
    /// Whether the path joined others on the way.
    pub joined: bool,
}

/// The decisions a path took, as it carries them: a shared chain each fork
/// extends without copying, read out into a [`Witness`] only where a finding
/// needs one.
#[derive(Debug, Clone, Default)]
pub struct Decisions {
    /// The last decision, linked to the ones before it.
    last: Option<Arc<Link>>,
    /// Whether the path joined others on the way.
    joined: bool,
}

/// One decision of a chain, with the chain before it.
#[derive(Debug)]
struct Link {
    /// The decision.
    decision: Decision,
    /// The decisions before it.
    previous: Option<Arc<Link>>,
}

impl Decisions {
    /// No decision: the path that entered the function.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last: None,
            joined: false,
        }
    }

    /// These decisions followed by `decision`.
    #[must_use]
    pub fn then(&self, decision: Decision) -> Self {
        Self {
            last: Some(Arc::new(Link {
                decision,
                previous: self.last.clone(),
            })),
            joined: self.joined,
        }
    }

    /// These decisions, of a path that joined others.
    #[must_use]
    pub fn joined(&self) -> Self {
        Self {
            last: self.last.clone(),
            joined: true,
        }
    }

    /// The witness these decisions are.
    #[must_use]
    pub fn witness(&self) -> Witness {
        let mut decisions = Vec::new();
        let mut link = self.last.as_deref();
        while let Some(current) = link {
            decisions.push(current.decision);
            link = current.previous.as_deref();
        }
        decisions.reverse();
        Witness {
            decisions,
            joined: self.joined,
        }
    }
}

/// Decisions describe how a path got somewhere, not where it is: two paths
/// compare equal and hash alike whatever decisions they carry.
impl PartialEq for Decisions {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Decisions {}

impl Hash for Decisions {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

/// What a [`Finding`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FindingKind {
    /// Bits no write had reached since their object began are used, as the
    /// use says.
    Unwritten(UnwrittenUseKind),
    /// An access ran past the extent of the object it reached, where the
    /// source says the extent comes from: an exact one is the object's own,
    /// a recovered one is inferred from how the code uses the storage.
    PastExtent(ExtentSource),
    /// An access reached heap storage whose lifetime had ended.
    Ended(Ending),
    /// A release handed storage that was no live block of the allocator
    /// releasing it.
    Release(ReleaseFault),
    /// A return went back through an address the run could not tell: a write
    /// it could not place may have overwritten the one the call left (the
    /// stack cell, or the link register's saved copy). The run goes on at the
    /// address the call left, the one the program returns to unless that
    /// write changed it.
    UnknownReturn,
    /// A store reached storage the image states is never written, as the
    /// path's protection changes leave it. The machine faults the store: the
    /// path ends there, unless a handler receives the raise.
    ReadOnlyWrite,
}

/// How surely the storage an access reached had ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Ending {
    /// It had ended on every path that reached the access.
    Every,
    /// It had ended on some of them.
    Some,
    /// A release the run could not resolve may have ended it.
    Possibly,
}

/// What was wrong with a release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ReleaseFault {
    /// The block had already ended: a second release.
    Repeated,
    /// The pointer lies inside a block, not at its start.
    Interior,
    /// The pointer names no heap storage.
    NotHeap,
    /// The block was made by another allocator family.
    OtherFamily,
}

/// Where a finding was made and what it found: the identity of a finding,
/// whatever path made it.
type FindingKey = (usize, usize, usize, FindingKind);

/// The findings of one run, on every path, shared by the paths, their
/// memories and the activations that record them. Two handles are one
/// record when they share it. One instruction finds one kind once: the first
/// path that finds it gives its evidence.
#[derive(Debug, Clone, Default)]
pub struct Findings(Arc<Mutex<BTreeMap<FindingKey, Finding>>>);

impl Findings {
    /// Records `found`, unless its instruction already found its kind.
    pub fn record(&self, found: Finding) {
        if let Ok(mut findings) = self.0.lock() {
            findings
                .entry((found.body, found.block, found.instruction, found.kind))
                .or_insert(found);
        }
    }

    /// Whether instruction `instruction` of block `block` of body `body`
    /// has found `kind` already, so a later record of it would add nothing.
    #[must_use]
    pub fn has(&self, body: usize, block: usize, instruction: usize, kind: FindingKind) -> bool {
        self.0
            .lock()
            .is_ok_and(|findings| findings.contains_key(&(body, block, instruction, kind)))
    }

    /// Every finding recorded, in order of instruction and kind.
    #[must_use]
    pub fn all(&self) -> Vec<Finding> {
        self.0
            .lock()
            .map(|findings| findings.values().cloned().collect())
            .unwrap_or_default()
    }
}

/// One record is one record: every path of a run shares the run's.
impl PartialEq for Findings {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Findings {}

/// A record hashes as the record it is, whatever it holds.
impl Hash for Findings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decision chain reads out first to last, a fork of it shares what
    /// came before without changing it, and a join marks the witness joined.
    #[test]
    fn decisions_read_out_first_to_last_and_forks_share_them() {
        let branch = |successor| Decision::Branch {
            body: 0,
            block: 1,
            successor,
        };
        let entry = Decisions::new();
        let first = entry.then(branch(2));
        let left = first.then(branch(3));
        let right = first.then(branch(4));
        assert_eq!(entry.witness(), Witness::default());
        assert_eq!(left.witness().decisions, [branch(2), branch(3)]);
        assert_eq!(right.witness().decisions, [branch(2), branch(4)]);
        assert_eq!(first.witness().decisions, [branch(2)]);
        let joined = left.joined();
        assert!(joined.witness().joined);
        assert_eq!(joined.witness().decisions, left.witness().decisions);
        assert!(joined.then(branch(5)).witness().joined);
    }
}
