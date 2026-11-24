//! Inclusion-based (Andersen) points-to analysis core.
//!
//! This is the solver: it takes a set of inclusion [`Constraint`]s over opaque
//! abstract [`Loc`]ations and computes their least fixpoint - each location's
//! points-to set. Constraint *extraction* from the SSA (and any interprocedural
//! summaries that feed it) layer on top; keeping the solver standalone lets it
//! be reasoned about and tested in isolation.
//!
//! [`analyze_function`] is the intraprocedural extractor: SSA value ids are
//! pointer locations; the function's frame, the image and each allocation are
//! [`Object`]s whose bytes are cells of their own ([`Constraint::Points`]),
//! moved through by address arithmetic ([`Constraint::Offset`]); and a managed
//! front end's locals, arguments and struct fields get stable synthetic
//! locations so two addresses of the same cell coincide. A host states what
//! the functions a call reaches do ([`CallModels`]); an object whose address
//! leaves the function by any other call, a return or the image may hold
//! anything ([`Constraint::Escape`]), and of the frame, the locals from the
//! least address that leaves up: a register spilled below them holds what the
//! function stored.
//!
//! The analysis is a **may**-analysis: every result over-approximates, and each
//! of its limits falls back toward *more* aliasing, never less. Field-cell
//! exhaustion falls back to the whole object; an object a pointer walks,
//! indexes without a bound, copies onto itself or accesses at overlapping
//! bytes is one cell. An exhausted step budget
//! ([`Budget::PointsToStepsPerConstraint`]) stops the solve before the fixpoint,
//! when some sets are still missing members they would have gained - a partial
//! relation that would answer "no alias" for pointers it had not finished, so
//! the result answers every location with [`UNKNOWN_LOC`] instead, and records
//! the [`Exhaustion`].
//!
//! # Reading a result
//!
//! An empty or absent points-to set means **unconstrained**, not "points
//! nowhere". A definition this module does not model - the result of a call no
//! model describes, a `Select` - is seeded with [`UNKNOWN_LOC`], which aliases
//! everything, and what it holds is unknown too. Implement "could these
//! alias?" as
//!
//! ```text
//! a.contains(UNKNOWN_LOC) || b.contains(UNKNOWN_LOC) || !a.is_disjoint(b)
//! ```
//!
//! A bare set intersection is **not** a sound alias query: it answers `NoAlias`
//! for any pointer the extractor did not understand.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    num::NonZeroU64,
    sync::{Arc, LazyLock, Mutex},
};

use log::warn;

use crate::{
    analysis::{
        address::{AddressBase, AliasKey},
        budget::{Budget, Exhaustion, WorkBudget},
        context::CallModels,
        execution::{Allocation, Delivery, LibraryModel},
    },
    ir::{
        CallData, ConstValue, InstrId, SsaFunction, SsaOp, SsaVarId,
        ops::{CallLocation, CallMemory, FrameOrigin, PointeeExtent},
    },
    pointer::PointerSize,
    target::Target,
};

/// An abstract memory location - an SSA value id, an allocation site, a stack
/// slot, a global. Opaque to the solver, which only relates locations by
/// inclusion.
pub type Loc = u32;

/// One inclusion constraint over abstract locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    /// `ptr ∋ &target` - `target` is a member of `pts(ptr)`.
    AddressOf {
        /// The pointer whose points-to set gains a member.
        ptr: Loc,
        /// The location whose address is taken.
        target: Loc,
    },
    /// `dst = src` - `pts(src) ⊆ pts(dst)`.
    Copy {
        /// Destination, which absorbs the source's points-to set.
        dst: Loc,
        /// Source of the copied points-to set.
        src: Loc,
    },
    /// `dst = *ptr` - for every `o ∈ pts(ptr)`, `pts(o) ⊆ pts(dst)`.
    Load {
        /// Destination of the loaded value.
        dst: Loc,
        /// Pointer being dereferenced.
        ptr: Loc,
    },
    /// `*ptr = src` - for every `o ∈ pts(ptr)`, `pts(src) ⊆ pts(o)`.
    Store {
        /// Pointer being written through.
        ptr: Loc,
        /// Value whose points-to set flows into the pointee.
        src: Loc,
    },
    /// `ptr ∋ &object[offset]` - `ptr` points at byte `offset` of `object`, a
    /// cell of its own ([`PointsTo::cell`]).
    Points {
        /// The pointer whose points-to set gains the cell.
        ptr: Loc,
        /// The object it points into.
        object: Object,
        /// The byte it points at.
        offset: i64,
    },
    /// `*dst = *src` for every byte: each cell of what `src` points into is
    /// copied into the cell as far into what `dst` points into (a `memcpy`).
    CopyCells {
        /// The pointer copied to.
        dst: Loc,
        /// The pointer copied from.
        src: Loc,
    },
    /// What `ptr` points at escapes the function: handed to a call, returned.
    /// An escaped object's cells may hold anything ([`UNKNOWN_LOC`]), and
    /// what they hold escapes too. Of the frame, what escapes is the local a
    /// pointer into it points at: the bytes from there to the next local whose
    /// address escapes, or to the frame's end; together, every byte from the
    /// least such address up. A byte below them, a register the function
    /// spills, is the function's own.
    Escape {
        /// The pointer that escapes.
        ptr: Loc,
    },
    /// An access through `ptr` moves `bytes` bytes, `None` where it does not
    /// say: two accesses to one object whose bytes overlap without being the
    /// same make the object one cell, as does one of no stated width, so a
    /// write is seen by every read of a byte it reaches.
    Width {
        /// The pointer accessed through.
        ptr: Loc,
        /// How many bytes the access moves.
        bytes: Option<u32>,
    },
    /// A call reaches the bytes `reached` names through `ptr` while it runs:
    /// where it `writes`, each cell there may hold anything once it returns;
    /// where it `reads`, what each holds escapes to it; where it `keeps` what
    /// it reaches, every byte from there up escapes ([`Constraint::Escape`]).
    /// A cell is reached where its first byte is, as an escape reaches the
    /// frame.
    Touch {
        /// The pointer the call is handed, or one the bytes it names are
        /// measured from.
        ptr: Loc,
        /// Which bytes, from where `ptr` points.
        reached: Reached,
        /// What the call does there.
        reach: Reach,
    },
    /// `dst = src + by` - for every cell `object[offset] ∈ pts(src)`,
    /// `object[offset + by] ∈ pts(dst)`, and every other pointee of `src` as
    /// it is ([`Shift`]).
    Offset {
        /// The pointer moved to.
        dst: Loc,
        /// The pointer moved from.
        src: Loc,
        /// How far.
        by: Shift,
    },
}

/// A memory object a pointer may point into, whose bytes are cells of their
/// own ([`Constraint::Points`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Object {
    /// The function's own frame: the bytes below its entry stack pointer,
    /// by their (negative) offset from it.
    Frame,
    /// The caller's bytes from the function's entry stack pointer up, by
    /// their offset from it: what the caller may have let go of, escaped
    /// from the outset. One space of offsets with [`Object::Frame`]
    /// ([`Object::at`]).
    Caller,
    /// The image, by address.
    Image,
    /// What one allocation returns, by the instruction that allocates it: a
    /// call to an allocator, or stack memory allocated as the function runs.
    Allocation(InstrId),
}

impl Object {
    /// The object byte `offset` of this one's space of offsets lies in: of
    /// the frame and the caller's bytes, which share the entry stack
    /// pointer's, the caller's from it up and the frame's below; any other
    /// object is itself.
    #[must_use]
    pub const fn at(self, offset: i64) -> Self {
        match self {
            Self::Frame | Self::Caller if offset >= 0 => Self::Caller,
            Self::Frame | Self::Caller => Self::Frame,
            Self::Image | Self::Allocation(_) => self,
        }
    }

    /// Whether every byte of it escaped whatever the function does: the
    /// image, which any other function may read and write, and the caller's
    /// bytes.
    #[must_use]
    pub const fn always_escaped(self) -> bool {
        matches!(self, Self::Image | Self::Caller)
    }
}

/// How far an [`Constraint::Offset`] moves a pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    /// By a known number of bytes.
    Bytes(i64),
    /// By an amount nothing bounds: anywhere in the object. The object's
    /// bytes are then one cell ([`PointsTo::cell`] answers `None` for its
    /// offset), the image's alone excepted, which is too many objects to be
    /// one: a pointer into it moved by an unknown amount may point anywhere
    /// ([`UNKNOWN_LOC`]).
    Unknown,
}

/// Which bytes a call reaches through a pointer ([`Constraint::Touch`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reached {
    /// Any byte of the object the pointer points into.
    Object,
    /// From `at` bytes past where the pointer points.
    From {
        /// How far past where the pointer points the bytes begin.
        at: i64,
        /// How many bytes from there; to the object's end where `None`.
        bytes: Option<NonZeroU64>,
    },
}

/// What a call does to the bytes it reaches ([`Constraint::Touch`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reach {
    /// It may write them.
    pub writes: bool,
    /// It may read them.
    pub reads: bool,
    /// It may keep their address past its return.
    pub keeps: bool,
}

/// The points-to relation - each location's points-to set.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PointsTo {
    /// The budget the solve ran out of, when it did. Such a relation is not a
    /// fixpoint, and answers every location with [`UNKNOWN_LOC`].
    exhaustion: Option<Exhaustion>,
    sets: BTreeMap<Loc, BTreeSet<Loc>>,
    /// Each cell location's object and byte, by its distance from
    /// [`CELL_LOC_BASE`]; `None` for the byte of a collapsed object.
    cells: Vec<(Object, Option<i64>)>,
    /// The objects whose every byte escaped ([`Constraint::Escape`]).
    escaped: BTreeSet<Object>,
    /// The least byte of the frame whose address escaped, where only some of
    /// it did: every byte from there up has escaped.
    frame_escaped_from: Option<i64>,
}

impl PointsTo {
    /// The empty relation - every location points nowhere known.
    ///
    /// `const` so a consumer that has no relation to offer can borrow a
    /// `'static` one instead of owning an instance, which is what lets an alias
    /// oracle borrow its relation from [`FunctionAnalyses`] in the common case
    /// and still have an empty fallback.
    ///
    /// [`FunctionAnalyses`]: crate::analysis::cache::FunctionAnalyses
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            exhaustion: None,
            sets: BTreeMap::new(),
            cells: Vec::new(),
            escaped: BTreeSet::new(),
            frame_escaped_from: None,
        }
    }

    /// Whether byte `offset` of `object` (`None` for any of its bytes) has
    /// escaped the function ([`Constraint::Escape`]): anything may write it,
    /// any call among them. The image and the caller's bytes always have
    /// ([`Object::always_escaped`]); of the frame, the bytes from the least
    /// one whose address escaped up. Every byte has, of a relation whose
    /// solve ran out of budget.
    #[must_use]
    pub fn has_escaped(&self, object: Object, offset: Option<i64>) -> bool {
        self.exhaustion.is_some()
            || object.always_escaped()
            || self.escaped.contains(&object)
            || (object == Object::Frame
                && self
                    .frame_escaped_from
                    .is_some_and(|from| offset.is_none_or(|offset| offset >= from)))
    }

    /// The cells a pointer to the cell `loc` reaches beside it: every cell of
    /// its object, of the frame those from its byte up (a local and the ones
    /// above it, as escape reads them) and the caller's above them all, the
    /// object's one cell included.
    /// Empty where `loc` is no cell.
    fn region(&self, loc: Loc) -> Vec<Loc> {
        let Some((object, from)) = self.cell(loc) else {
            return Vec::new();
        };
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, (other, offset))| {
                (*other == object
                    && (object != Object::Frame
                        || from
                            .zip(*offset)
                            .is_none_or(|(from, offset)| offset >= from)))
                    || (object == Object::Frame && *other == Object::Caller)
            })
            .filter_map(|(index, _)| {
                u32::try_from(index)
                    .ok()
                    .and_then(|index| CELL_LOC_BASE.checked_add(index))
            })
            .collect()
    }

    /// The object and byte the cell location `loc` stands for
    /// ([`Constraint::Points`]), the byte `None` where the object is one cell
    /// ([`Shift::Unknown`]); `None` where `loc` is no cell.
    #[must_use]
    pub fn cell(&self, loc: Loc) -> Option<(Object, Option<i64>)> {
        let index = loc.checked_sub(CELL_LOC_BASE)?;
        self.cells.get(usize::try_from(index).ok()?).copied()
    }

    /// Returns the points-to set of `loc`, empty when the location points
    /// nowhere known.
    ///
    /// For a relation whose solve ran out of budget, every location's answer is
    /// `{`[`UNKNOWN_LOC`]`}`: the solve stopped before the fixpoint, so no set it
    /// holds is known to be complete.
    #[must_use]
    pub fn get(&self, loc: Loc) -> &BTreeSet<Loc> {
        static EMPTY: BTreeSet<Loc> = BTreeSet::new();
        static ANYWHERE: LazyLock<BTreeSet<Loc>> = LazyLock::new(|| BTreeSet::from([UNKNOWN_LOC]));
        if self.exhaustion.is_some() {
            return &ANYWHERE;
        }
        self.sets.get(&loc).unwrap_or(&EMPTY)
    }

    /// Iterates each location with a non-empty points-to set and its pointees.
    ///
    /// Empty for a relation whose solve ran out of budget: it lists nothing it
    /// cannot vouch for, and [`get`](Self::get) answers every location.
    pub fn iter(&self) -> impl Iterator<Item = (Loc, &BTreeSet<Loc>)> {
        let listed = if self.exhaustion.is_some() {
            None
        } else {
            Some(&self.sets)
        };
        listed.into_iter().flatten().map(|(loc, set)| (*loc, set))
    }

    /// Returns the budget the solve ran out of, when it did.
    #[must_use]
    pub const fn exhaustion(&self) -> Option<Exhaustion> {
        self.exhaustion
    }

    /// Returns the transitive points-to closure reachable from `seeds`.
    ///
    /// Starting from each seed location, follows points-to edges to a fixpoint,
    /// collecting every abstract location a chain of dereferences could reach.
    /// The seeds themselves are included. Used to compute what memory a callee
    /// can touch through the pointer arguments it is handed.
    ///
    /// # Arguments
    ///
    /// * `seeds` - The starting locations; each is added to the result and its
    ///   points-to edges are followed transitively.
    ///
    /// # Returns
    ///
    /// The set of all locations reachable from `seeds`, seeds included.
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
            // A pointer into an object reaches the rest of what it points
            // into, and what that holds.
            for beside in self.region(loc) {
                if reached.insert(beside) {
                    stack.push(beside);
                }
            }
            for &pointee in self.get(loc) {
                if reached.insert(pointee) {
                    stack.push(pointee);
                }
            }
        }
        reached
    }

    /// The footprint of an access through the value `pointer`, from `first`
    /// bytes past where it points to `end`, on a target whose addresses are
    /// `ptr_size` wide. `end` is `None` where the access's extent is unknown
    /// or an index moves it: it may then touch any byte of the object its
    /// first byte lies in, which an index does not leave.
    ///
    /// [`Footprint::Anywhere`] where the relation places the value nowhere
    /// (a number, a value nothing defines), where it points at a location
    /// that is no object's byte, and where an offset leaves the range of the
    /// target's addresses, which the relation does not wrap.
    ///
    /// # Arguments
    ///
    /// * `pointer` - The value the access goes through.
    /// * `first` - Its first byte, from where `pointer` points.
    /// * `end` - The byte past its last, where its extent is known.
    /// * `ptr_size` - The target's pointer width.
    #[must_use]
    pub fn footprint(
        &self,
        pointer: SsaVarId,
        first: i64,
        end: Option<i64>,
        ptr_size: PointerSize,
    ) -> Footprint {
        let pointees = self.get(pointer.as_u32());
        if pointees.is_empty() {
            return Footprint::Anywhere;
        }
        let mut spans = Vec::new();
        let mut escaped = false;
        for &pointee in pointees {
            if pointee == UNKNOWN_LOC {
                escaped = true;
                continue;
            }
            let Some((object, at)) = self.cell(pointee) else {
                return Footprint::Anywhere;
            };
            // The image is addressed unsigned, every other object from its
            // start or its entry stack pointer, signed.
            let addressable = |offset: i64| match object {
                Object::Image => u64::try_from(offset)
                    .is_ok_and(|offset| ptr_size.mask_unsigned(offset) == offset),
                Object::Frame | Object::Caller | Object::Allocation(_) => {
                    ptr_size.mask_signed(offset) == offset
                }
            };
            let Some(at) = at else {
                spans.push((object, None));
                continue;
            };
            let Some(first) = at.checked_add(first).filter(|first| addressable(*first)) else {
                return Footprint::Anywhere;
            };
            // Bytes of the frame's space of offsets on either side of the
            // entry stack pointer are the frame's and the caller's.
            let below = object.at(first);
            let Some(end) = end else {
                spans.push((below, None));
                continue;
            };
            let Some(end) = at
                .checked_add(end)
                .filter(|end| *end <= first || addressable(end.saturating_sub(1)))
            else {
                return Footprint::Anywhere;
            };
            let above = object.at(end.saturating_sub(1).max(first));
            if below == above {
                spans.push((below, Some((first, end))));
            } else {
                spans.push((below, Some((first, 0))));
                spans.push((above, Some((0, end))));
            }
        }
        self.placed(spans, escaped)
    }

    /// The footprint of an access over `spans`, and, where `escaped`, over
    /// memory the function lets go of, with whether a span holds a byte that
    /// escaped ([`Self::has_escaped`]): of the frame, the last byte of a span
    /// is the highest, and the frame escapes from a byte up.
    ///
    /// # Arguments
    ///
    /// * `spans` - Each object and the bytes of it, `None` for any of them.
    /// * `escaped` - Whether the access may touch memory let go of.
    #[must_use]
    pub fn placed(&self, spans: Vec<(Object, Option<(i64, i64)>)>, escaped: bool) -> Footprint {
        let exposed = spans.iter().any(|&(object, bytes)| {
            self.has_escaped(object, bytes.map(|(_, end)| end.saturating_sub(1)))
        });
        Footprint::Placed {
            spans,
            escaped,
            exposed,
        }
    }
}

/// The memory an access may touch, as a [`PointsTo`] relation places it
/// ([`PointsTo::footprint`]): bytes of the objects it names, and the memory
/// the function lets go of.
///
/// Objects are apart from one another. What a pointer of unknown pointees
/// ([`UNKNOWN_LOC`]) may reach is every byte that escaped the function and
/// every byte of no object the function knows, which leaves out what the
/// function keeps to itself: the frame below its least escaped local, and
/// each allocation that never escapes ([`PointsTo::has_escaped`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Footprint {
    /// Any memory: nothing places the access.
    Anywhere,
    /// Bytes of the objects the relation names, and, where `escaped`, any
    /// byte that escaped the function or lies in no object it knows
    /// ([`PointsTo::placed`]).
    Placed {
        /// Each object the access may touch, with the bytes `[first, end)` of
        /// it the access may touch, `None` for any of them.
        spans: Vec<(Object, Option<(i64, i64)>)>,
        /// Whether the access may touch memory the relation knows only as
        /// escaped.
        escaped: bool,
        /// Whether one of `spans` holds a byte that escaped, which memory let
        /// go of may be.
        exposed: bool,
    },
}

impl Footprint {
    /// Memory the function lets go of: any byte that escaped it, and any
    /// byte of no object it knows.
    pub const ESCAPED: Self = Self::Placed {
        spans: Vec::new(),
        escaped: true,
        exposed: false,
    };

    /// The bytes `[first, end)` of the image, `None` for any of them: an
    /// access at an address the program states. Every byte of the image
    /// escaped.
    #[must_use]
    pub fn image(bytes: Option<(i64, i64)>) -> Self {
        Self::Placed {
            spans: vec![(Object::Image, bytes)],
            escaped: false,
            exposed: true,
        }
    }

    /// Whether an access over `self` and one over `other` may touch one
    /// byte.
    ///
    /// # Arguments
    ///
    /// * `other` - The other access's footprint, placed by the same relation.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        let (
            Self::Placed {
                spans: mine,
                escaped: mine_escaped,
                exposed: mine_exposed,
            },
            Self::Placed {
                spans: theirs,
                escaped: theirs_escaped,
                exposed: theirs_exposed,
            },
        ) = (self, other)
        else {
            return true;
        };
        let shared = mine.iter().any(|&(object, bytes)| {
            theirs.iter().any(|&(other, other_bytes)| {
                object == other
                    && bytes.zip(other_bytes).is_none_or(
                        |((first, end), (other_first, other_end))| {
                            first < other_end && other_first < end
                        },
                    )
            })
        });
        (*mine_escaped && (*theirs_escaped || *theirs_exposed))
            || (*theirs_escaped && *mine_exposed)
            || shared
    }
}

/// The mod-ref question for a call, answered from points-to: which alias cells
/// can a call write?
///
/// A call cannot modify the cell behind a pointer whose address provably never
/// reaches the callee - it is neither passed (directly or transitively) as an
/// argument nor stored into memory that escapes. Built once per function, it
/// lets a forward analysis keep what it knows of such cells across a call
/// instead of forgetting every cell at every call.
///
/// Always sound: where escape cannot be proven - no points-to information for
/// a cell's base, or a base that reaches escaped memory - the cell is
/// clobbered, exactly as forgetting everything would.
///
/// It shares the relation it answers from, so one oracle per function can be
/// kept beside the relation ([`FunctionAnalyses::call_escape`]) and asked by
/// every solve of that function.
///
/// [`FunctionAnalyses::call_escape`]: crate::analysis::cache::FunctionAnalyses::call_escape
#[derive(Debug, Clone)]
pub struct CallEscape {
    /// The function's points-to relation.
    points_to: Arc<PointsTo>,
    /// Locations whose address escapes the function whatever the call -
    /// anything passed to any call, and its points-to closure. A cell whose
    /// base reaches one of these is clobberable by *any* call.
    globally_escaped: BTreeSet<Loc>,
    /// The least byte of each object the calls' inputs reach a cell of: every
    /// byte of it from there up a call may reach ([`i64::MIN`] for all).
    reached_from: BTreeMap<Object, i64>,
}

impl Default for CallEscape {
    /// The oracle built from no information: every call clobbers every cell.
    fn default() -> Self {
        Self {
            points_to: Arc::new(PointsTo::empty()),
            globally_escaped: BTreeSet::new(),
            reached_from: BTreeMap::new(),
        }
    }
}

impl CallEscape {
    /// Builds the oracle from a function's points-to relation and the
    /// locations handed to some call.
    ///
    /// The seeds must cover **every** call in the function: each call's ordinary
    /// and implicit inputs, and an indirect call's target too. That lets one set
    /// answer for all of them - a single call's reachable closure is a subset
    /// of the closure over every call's arguments, so its union with the global
    /// set *is* the global set, and no call needs a closure of its own.
    #[must_use]
    pub fn new<I>(points_to: Arc<PointsTo>, call_input_values: I) -> Self
    where
        I: IntoIterator<Item = Loc>,
    {
        let globally_escaped = points_to.reachable_closure(call_input_values);
        let mut reached_from: BTreeMap<Object, i64> = BTreeMap::new();
        for (object, offset) in globally_escaped
            .iter()
            .filter_map(|loc| points_to.cell(*loc))
        {
            let from = match (object, offset) {
                (Object::Frame, Some(offset)) => offset,
                _ => i64::MIN,
            };
            let least = reached_from.entry(object).or_insert(from);
            *least = (*least).min(from);
        }
        Self {
            points_to,
            globally_escaped,
            reached_from,
        }
    }

    /// Builds the oracle for `ssa`, seeded with every value it hands a call -
    /// each call's ordinary and implicit inputs and each indirect target - and
    /// every value it stores to memory.
    ///
    /// A stored value escapes as surely as a passed one: an address written to
    /// a static field, a field or an element is there for any callee to read
    /// back and write through. Seeding the arguments alone let a call keep a
    /// cell whose address it could reach through a global.
    #[must_use]
    pub fn of<T: Target>(points_to: Arc<PointsTo>, ssa: &SsaFunction<T>) -> Self {
        let mut values: Vec<Loc> = Vec::new();
        for block in ssa.blocks() {
            for instruction in block.instructions() {
                match instruction.op() {
                    SsaOp::Call(call) => {
                        values.extend(call.args.iter().map(|arg| arg.as_u32()));
                        values.extend(
                            call.implicit_inputs
                                .iter()
                                .map(|input| input.value.as_u32()),
                        );
                        values.extend(call.callee.fptr().map(|fptr| fptr.as_u32()));
                    }
                    SsaOp::StoreStaticField { value, .. }
                    | SsaOp::StoreField { value, .. }
                    | SsaOp::StoreElement { value, .. }
                    | SsaOp::StoreIndirect { value, .. } => values.push(value.as_u32()),
                    _ => {}
                }
            }
        }
        values.sort_unstable();
        values.dedup();
        Self::new(points_to, values)
    }

    /// Returns `true` when the alias cells `a` and `b` may lie in one object.
    ///
    /// Two cells off different pointer values are in different objects when
    /// the values point into disjoint sets of them. Anything the relation does
    /// not vouch for - a base that is not a value, a value with no points-to
    /// set, one that may point anywhere - may share, which is the sound answer.
    ///
    /// # Arguments
    ///
    /// * `a` - One cell.
    /// * `b` - The other.
    #[must_use]
    pub fn may_share_object(&self, a: &AliasKey, b: &AliasKey) -> bool {
        let (AddressBase::Value(left), AddressBase::Value(right)) = (a.base, b.base) else {
            return true;
        };
        if left == right {
            return true;
        }
        let (left, right) = (
            self.points_to.get(left.as_u32()),
            self.points_to.get(right.as_u32()),
        );
        let vouched = |set: &BTreeSet<Loc>| !set.is_empty() && !set.contains(&UNKNOWN_LOC);
        !(vouched(left) && vouched(right) && left.is_disjoint(right))
    }

    /// Returns `true` when a call can clobber the alias cell `key`.
    ///
    /// The cell survives only when its base's points-to set is wholly disjoint
    /// from what the function's calls can reach. A base with no points-to
    /// information is treated as reachable: nothing proves it stays local.
    ///
    /// An absolute cell has no base value; it is answered as its constant's
    /// points-to set reads, which is the single unknown location. A slot of
    /// the function's own frame is its own location, reachable exactly when
    /// its address escaped.
    #[must_use]
    pub fn clobbers(&self, key: &AliasKey) -> bool {
        let base = match key.base {
            AddressBase::Value(base) => base,
            AddressBase::Local(index) => {
                return self
                    .globally_escaped
                    .contains(&LOCAL_LOC_BASE.saturating_add(u32::from(index)));
            }
            AddressBase::Argument(index) => {
                return self
                    .globally_escaped
                    .contains(&ARG_LOC_BASE.saturating_add(u32::from(index)));
            }
            AddressBase::Symbol | AddressBase::Number => {
                return self.globally_escaped.contains(&UNKNOWN_LOC);
            }
        };
        let pointees = self.points_to.get(base.as_u32());
        if pointees.is_empty() {
            return true;
        }
        // The byte the key names past each cell its base points at, or any
        // of the object's where an index moves it.
        let moved = key
            .index
            .is_none()
            .then(|| {
                key.offset_bits
                    .checked_div(8)
                    .filter(|_| key.offset_bits % 8 == 0)
            })
            .flatten();
        pointees
            .iter()
            .any(|pointee| match self.points_to.cell(*pointee) {
                Some((object, offset)) => {
                    let byte = offset
                        .zip(moved)
                        .and_then(|(offset, moved)| offset.checked_add(moved));
                    // A byte of the frame's space of offsets moved across the
                    // entry stack pointer is the caller's, or the frame's.
                    let object = byte.map_or(object, |byte| object.at(byte));
                    self.points_to.has_escaped(object, byte) || self.reached(object, byte)
                }
                None => self.globally_escaped.contains(pointee),
            })
    }

    /// Whether a call may reach byte `byte` of `object` (`None` for any of
    /// its bytes) through what the function hands its calls: a cell of the
    /// object the calls' inputs reach, of the frame one at or below the byte.
    fn reached(&self, object: Object, byte: Option<i64>) -> bool {
        self.reached_from
            .get(&object)
            .is_some_and(|from| byte.is_none_or(|byte| byte >= *from))
    }
}

/// The load / store constraints, indexed by the pointer location they
/// dereference.
///
/// Written once while loading the constraint set and never mutated afterwards.
/// Kept out of [`SolverGraph`] precisely so the worklist loop can hold these
/// borrowed while it mutates the solver state - the two are disjoint, and
/// merging them would force the loop to clone an edge list on every pop just to
/// satisfy the borrow checker.
#[derive(Default)]
struct DerefConstraints {
    /// Load constraints indexed by pointer: `ptr → {dst}` means `dst = *ptr`.
    loads: BTreeMap<Loc, BTreeSet<Loc>>,
    /// Store constraints indexed by pointer: `ptr → {src}` means `*ptr = src`.
    stores: BTreeMap<Loc, BTreeSet<Loc>>,
    /// Offset constraints indexed by the pointer moved: `src → {(dst, by)}`
    /// means `dst = src + by`.
    offsets: BTreeMap<Loc, Vec<(Loc, Shift)>>,
    /// The widths of the accesses through each pointer ([`Constraint::Width`]).
    widths: BTreeMap<Loc, BTreeSet<Option<u32>>>,
    /// The pointers that escape ([`Constraint::Escape`]).
    escapes: BTreeSet<Loc>,
    /// What calls reach through each pointer ([`Constraint::Touch`]).
    touches: BTreeMap<Loc, BTreeSet<Touch>>,
    /// Block copies, indexed by either pointer: `loc → {(dst, src)}`
    /// ([`Constraint::CopyCells`]).
    copies: BTreeMap<Loc, BTreeSet<(Loc, Loc)>>,
}

/// How many distinct bytes of one object the cells name before the object is
/// one cell.
const MAX_OBJECT_BYTES: usize = 4096;

/// How many distinct bytes of one object one pointer may point at before
/// the object is one cell: a pointer a loop walks reaches a new byte each
/// trip, where a record's pointer names one of a few.
const MAX_POINTER_BYTES: usize = 16;

/// The cells a solve names ([`Constraint::Points`]), minted as constraints
/// and moves reach them.
#[derive(Default)]
struct Cells {
    /// Each cell's location, by object and byte.
    by_name: BTreeMap<(Object, Option<i64>), Loc>,
    /// Each cell's object and byte, by its distance from [`CELL_LOC_BASE`].
    names: Vec<(Object, Option<i64>)>,
    /// Each object's cells.
    of: BTreeMap<Object, Vec<Loc>>,
    /// The objects that are one cell.
    collapsed: BTreeSet<Object>,
    /// The bytes each object's accesses move, as `(first byte, length)`.
    accessed: BTreeMap<Object, BTreeSet<(i64, u32)>>,
    /// The objects whose addresses escaped the function; the image always,
    /// which any other function may read and write.
    escaped: BTreeSet<Object>,
    /// The least byte of the frame whose address escaped, where only some of
    /// it did: every byte from there up has escaped ([`Self::escape`]).
    frame_escaped_from: Option<i64>,
    /// The bytes of each object a call reaches ([`Self::touch`]).
    touched: BTreeMap<Object, Vec<Touched>>,
    /// Block copies between objects: each cell of the first object at a byte
    /// is copied into the second's cell that many bytes further, `None`
    /// where it is copied into the second's one cell.
    copied: BTreeSet<(Object, Option<i64>, Object)>,
    /// The cells named since the copies were last applied ([`Self::drain`]).
    fresh: Vec<Loc>,
}

/// What a call reaches through one pointer ([`Constraint::Touch`]): the bytes
/// and what it does there, as the constraint states them.
type Touch = (Reached, Reach);

/// Bytes of one object a call reaches ([`Constraint::Touch`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Touched {
    /// The first byte.
    first: i64,
    /// The byte past the last, `None` for the object's end.
    end: Option<i64>,
    /// What the call does there.
    reach: Reach,
}

impl Touched {
    /// Whether the cell whose first byte is `offset` (`None` for the cell
    /// of a whole object) lies in these bytes.
    fn reaches(&self, offset: Option<i64>) -> bool {
        offset.is_none_or(|offset| self.first <= offset && self.end.is_none_or(|end| offset < end))
    }

    /// Makes the reached cell `cell` hold anything where the call writes
    /// it, and what it holds escape where the call reads it.
    fn apply(&self, cell: Loc, graph: &mut SolverGraph, work: &mut Worklist) {
        if self.reach.writes && graph.add_pointee(cell, UNKNOWN_LOC) {
            work.push(cell);
        }
        if self.reach.reads {
            graph.add_copy_edge(cell, UNKNOWN_LOC, work);
        }
    }
}

impl Cells {
    /// The cell of byte `offset` of `object`, `None` for the whole of it,
    /// minted on first sight; [`UNKNOWN_LOC`] once the cell band is full.
    ///
    /// A byte of the frame's space of offsets is the caller's or the frame's
    /// as it lies ([`Object::at`]). Naming the whole object, or more than
    /// [`MAX_OBJECT_BYTES`] of its bytes, makes it one cell: every cell of it holds what the others do
    /// ([`Self::collapse`]), and every byte of it is that cell from then on.
    fn cell(
        &mut self,
        object: Object,
        offset: Option<i64>,
        graph: &mut SolverGraph,
        work: &mut Worklist,
    ) -> Loc {
        let object = offset.map_or(object, |offset| object.at(offset));
        let offset = offset.filter(|_| !self.collapsed.contains(&object));
        if let Some(loc) = self.by_name.get(&(object, offset)) {
            return *loc;
        }
        let Some(loc) = u32::try_from(self.names.len())
            .ok()
            .and_then(|index| CELL_LOC_BASE.checked_add(index))
            .filter(|loc| *loc < SYNTHETIC_LOC_BASE)
        else {
            return UNKNOWN_LOC;
        };
        self.by_name.insert((object, offset), loc);
        self.names.push((object, offset));
        let cells = self.of.entry(object).or_default();
        cells.push(loc);
        let too_many = cells.len() > MAX_OBJECT_BYTES;
        if self.has_escaped(object, offset) {
            graph.tie_to_unknown(loc, work);
        }
        for touched in self.touched.get(&object).into_iter().flatten() {
            if touched.reaches(offset) {
                touched.apply(loc, graph, work);
            }
        }
        // A copy out of the object reaches the new cell too, once drained.
        self.fresh.push(loc);
        if offset.is_none() {
            self.collapse(object, loc, graph, work);
        } else if too_many {
            return self.cell(object, None, graph, work);
        }
        loc
    }

    /// Applies the copies out of each object to its cells named since the last
    /// drain ([`Self::copy`]), and to those the copies name in turn.
    fn drain(&mut self, graph: &mut SolverGraph, work: &mut Worklist) {
        while let Some(loc) = self.fresh.pop() {
            let Some((object, offset)) = self.name(loc) else {
                continue;
            };
            let copies: Vec<(Option<i64>, Object)> = self
                .copied
                .iter()
                .filter(|(from, _, _)| *from == object)
                .map(|(_, delta, to)| (*delta, *to))
                .collect();
            for (delta, to) in copies {
                let target = match (delta, offset) {
                    (Some(delta), Some(offset)) => {
                        self.cell(to, offset.checked_add(delta), graph, work)
                    }
                    _ => self.cell(to, None, graph, work),
                };
                graph.add_copy_edge(loc, target, work);
            }
        }
    }

    /// The object and byte the cell `loc` names; `None` where it is no cell.
    fn name(&self, loc: Loc) -> Option<(Object, Option<i64>)> {
        loc.checked_sub(CELL_LOC_BASE)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.names.get(index))
            .copied()
    }

    /// Copies what `src` holds into `dst` byte for byte, where both are cells
    /// ([`Constraint::CopyCells`]): each cell of `src`'s object into the cell
    /// as far from `dst`, now and as cells are named. A copy from what may be
    /// anywhere writes anything; one to it hands what is copied away.
    fn copy(&mut self, dst: Loc, src: Loc, graph: &mut SolverGraph, work: &mut Worklist) {
        match (self.name(src), self.name(dst)) {
            (Some((from, src_offset)), Some((to, dst_offset))) => {
                let delta = src_offset
                    .zip(dst_offset)
                    .and_then(|(src, dst)| dst.checked_sub(src));
                // A copy within one object to another place in it would name
                // a byte further on for every byte it reaches: the object is
                // one cell instead, which the copy leaves as it was.
                if from == to && delta != Some(0) {
                    self.cell(from, None, graph, work);
                    return;
                }
                if !self.copied.insert((from, delta, to)) {
                    return;
                }
                for cell in self.of.get(&from).cloned().unwrap_or_default() {
                    let offset = self.name(cell).and_then(|(_, offset)| offset);
                    let target = match (delta, offset) {
                        (Some(delta), Some(offset)) => {
                            self.cell(to, offset.checked_add(delta), graph, work)
                        }
                        _ => self.cell(to, None, graph, work),
                    };
                    graph.add_copy_edge(cell, target, work);
                }
            }
            (None, Some(_)) if src == UNKNOWN_LOC => {
                if graph.add_pointee(dst, UNKNOWN_LOC) {
                    work.push(dst);
                }
            }
            (Some((from, _)), None) if dst == UNKNOWN_LOC => {
                for cell in self.of.get(&from).cloned().unwrap_or_default() {
                    graph.add_copy_edge(cell, UNKNOWN_LOC, work);
                }
            }
            _ => {}
        }
    }

    /// Whether byte `offset` of `object`, `None` for the whole of it, has
    /// escaped ([`Self::escape`]).
    fn has_escaped(&self, object: Object, offset: Option<i64>) -> bool {
        object.always_escaped()
            || self.escaped.contains(&object)
            || (object == Object::Frame
                && self
                    .frame_escaped_from
                    .is_some_and(|from| offset.is_none_or(|offset| offset >= from)))
    }

    /// Marks what the cell `pointee` points into escaped: each cell of it may
    /// hold anything, and what it holds escapes ([`SolverGraph::tie_to_unknown`]).
    /// Of an object, that is the whole of it; of the frame, the local the
    /// cell begins, whose address escaped, and every local above it another
    /// escape may begin ([`Constraint::Escape`]), unless the frame is one
    /// cell.
    fn escape(&mut self, pointee: Loc, graph: &mut SolverGraph, work: &mut Worklist) {
        let Some((object, offset)) = self.name(pointee) else {
            return;
        };
        self.escape_from(object, offset, graph, work);
    }

    /// Marks `object` escaped from its byte `offset` up ([`Self::escape`]):
    /// of the frame, the bytes from there, every byte where `offset` is
    /// `None` or the frame is one cell; of any other object, the whole of it.
    fn escape_from(
        &mut self,
        object: Object,
        offset: Option<i64>,
        graph: &mut SolverGraph,
        work: &mut Worklist,
    ) {
        let offset = offset.or((object == Object::Frame).then_some(i64::MIN));
        if object.always_escaped() {
            return;
        }
        let from = match (object, offset) {
            (Object::Frame, Some(offset)) if !self.collapsed.contains(&Object::Frame) => {
                if self.escaped.contains(&object)
                    || self.frame_escaped_from.is_some_and(|from| from <= offset)
                {
                    return;
                }
                self.frame_escaped_from = Some(offset);
                Some(offset)
            }
            _ => {
                if !self.escaped.insert(object) {
                    return;
                }
                None
            }
        };
        for cell in self.of.get(&object).cloned().unwrap_or_default() {
            let reached = match (from, self.name(cell)) {
                (Some(from), Some((_, Some(offset)))) => offset >= from,
                _ => true,
            };
            if reached {
                graph.tie_to_unknown(cell, work);
            }
        }
    }

    /// Records that a call reaches the bytes `reached` names past the cell
    /// `pointee`, as `reach` says ([`Constraint::Touch`]): each cell there
    /// named so far, and each named later ([`Self::cell`]). The cells of an
    /// object that always escaped are tied to what may be anywhere already.
    fn touch(
        &mut self,
        pointee: Loc,
        reached: Reached,
        reach: Reach,
        graph: &mut SolverGraph,
        work: &mut Worklist,
    ) {
        let Some((object, offset)) = self.name(pointee) else {
            return;
        };
        let (at, bytes) = match reached {
            Reached::Object => (None, None),
            Reached::From { at, bytes } => (Some(at), bytes.map(NonZeroU64::get)),
        };
        let first = offset
            .zip(at)
            .and_then(|(offset, at)| offset.checked_add(at));
        // The bytes lie where their first does in the frame's space of
        // offsets; anywhere in that space, the frame's own are the ones the
        // call can change, the caller's having escaped.
        let object = match first {
            Some(first) => object.at(first),
            None if object == Object::Caller => Object::Frame,
            None => object,
        };
        if object.always_escaped() {
            return;
        }
        let touched = Touched {
            first: first.unwrap_or(i64::MIN),
            end: first.zip(bytes).and_then(|(first, bytes)| {
                i64::try_from(bytes)
                    .ok()
                    .and_then(|bytes| first.checked_add(bytes))
            }),
            reach,
        };
        if reach.keeps {
            self.escape_from(object, first, graph, work);
        }
        let recorded = self.touched.entry(object).or_default();
        if recorded.contains(&touched) {
            return;
        }
        recorded.push(touched);
        for cell in self.of.get(&object).cloned().unwrap_or_default() {
            if touched.reaches(self.name(cell).and_then(|(_, offset)| offset)) {
                touched.apply(cell, graph, work);
            }
        }
    }

    /// Makes `object` one cell, `whole`: each cell of it named so far and
    /// `whole` hold what the other holds.
    fn collapse(
        &mut self,
        object: Object,
        whole: Loc,
        graph: &mut SolverGraph,
        work: &mut Worklist,
    ) {
        self.collapsed.insert(object);
        let cells = self.of.get(&object).cloned().unwrap_or_default();
        for cell in cells.into_iter().filter(|cell| *cell != whole) {
            graph.add_copy_edge(cell, whole, work);
            graph.add_copy_edge(whole, cell, work);
        }
    }

    /// Records an access of `bytes` bytes at the cell `pointee`, making its
    /// object one cell where the access says no width or overlaps another of
    /// the object's accesses without being the same bytes. One of the frame
    /// that runs into the caller's bytes lets the frame go from its first.
    fn access(
        &mut self,
        pointee: Loc,
        bytes: Option<u32>,
        graph: &mut SolverGraph,
        work: &mut Worklist,
    ) {
        let Some(&(object, Some(offset))) = pointee
            .checked_sub(CELL_LOC_BASE)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.names.get(index))
        else {
            return;
        };
        let Some(bytes) = bytes.filter(|bytes| *bytes > 0) else {
            self.cell(object, None, graph, work);
            return;
        };
        let end = offset.saturating_add(i64::from(bytes));
        // An access of the frame that runs into the caller's bytes reaches
        // what the caller may have let go of.
        if object == Object::Frame && end > 0 {
            self.escape_from(object, Some(offset), graph, work);
        }
        let accessed = self.accessed.entry(object).or_default();
        let overlaps = accessed.iter().any(|&(first, length)| {
            (first, length) != (offset, bytes)
                && first < end
                && offset < first.saturating_add(i64::from(length))
        });
        accessed.insert((offset, bytes));
        if overlaps {
            self.cell(object, None, graph, work);
        }
    }

    /// Makes the object of the cell `moved` one cell where the pointer `dst`
    /// now points at more than [`MAX_POINTER_BYTES`] of its bytes: a walk.
    fn bound_walk(&mut self, dst: Loc, moved: Loc, graph: &mut SolverGraph, work: &mut Worklist) {
        let Some(&(object, Some(_))) = moved
            .checked_sub(CELL_LOC_BASE)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.names.get(index))
        else {
            return;
        };
        let held = graph.sets.get(&dst).map_or(0, |pointees| {
            pointees
                .iter()
                .filter(|pointee| {
                    pointee
                        .checked_sub(CELL_LOC_BASE)
                        .and_then(|index| usize::try_from(index).ok())
                        .and_then(|index| self.names.get(index))
                        .is_some_and(|(held, _)| *held == object)
                })
                .count()
        });
        if held > MAX_POINTER_BYTES {
            let whole = self.cell(object, None, graph, work);
            if graph.add_pointee(dst, whole) {
                work.push(dst);
            }
        }
    }

    /// Where `pointee` is moved `by`: a cell to its object's byte that far
    /// on, any other location as it is. Moved by an amount nothing bounds, a
    /// cell is anywhere in its object: of escaped memory, what may be
    /// anywhere ([`UNKNOWN_LOC`]); of any other, the object as one cell; the
    /// entry stack pointer's byte, the function's own frame as one cell.
    fn moved(
        &mut self,
        pointee: Loc,
        by: Shift,
        graph: &mut SolverGraph,
        work: &mut Worklist,
    ) -> Loc {
        let Some(&(object, offset)) = pointee
            .checked_sub(CELL_LOC_BASE)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.names.get(index))
        else {
            return pointee;
        };
        match (by, offset) {
            (_, None) => pointee,
            (Shift::Bytes(bytes), Some(offset)) => {
                let moved = offset.checked_add(bytes);
                self.cell(object, moved, graph, work)
            }
            // The entry stack pointer moved by an amount nothing bounds (an
            // alignment) points into the function's own frame, where it
            // allocates: the caller's bytes are reached by stated offsets.
            (Shift::Unknown, Some(0)) if object == Object::Caller => {
                self.cell(Object::Frame, None, graph, work)
            }
            // A byte of escaped memory moved anywhere in its object is any
            // such byte: what may be anywhere, which reaches each already.
            (Shift::Unknown, Some(offset)) if self.has_escaped(object, Some(offset)) => UNKNOWN_LOC,
            (Shift::Unknown, Some(_)) => self.cell(object, None, graph, work),
        }
    }
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
    /// `copy_edges` - a different field - while this mutates the points-to
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

    /// Makes the escaped cell `cell` hold anything, and what it holds what
    /// may be anywhere holds.
    fn tie_to_unknown(&mut self, cell: Loc, work: &mut Worklist) {
        if self.add_pointee(cell, UNKNOWN_LOC) {
            work.push(cell);
        }
        self.add_copy_edge(cell, UNKNOWN_LOC, work);
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
///
/// A step budget proportional to the constraint count caps pathological cubic
/// runs: `budget`'s
/// [`PointsToStepsPerConstraint`](Budget::PointsToStepsPerConstraint) times one
/// more than the constraint count, so an empty or tiny constraint set still has
/// room to converge.
///
/// # Arguments
///
/// * `constraints` - The inclusion constraints to solve; base `AddressOf` facts
///   seed the points-to sets and the rest form the copy / load / store edges.
/// * `budget` - The work budget the solve is bounded by.
///
/// # Returns
///
/// The [`PointsTo`] relation at the least fixpoint, or, when the step budget was
/// exceeded, a saturated one that answers every location with [`UNKNOWN_LOC`]
/// and records the [`Exhaustion`] (see [`PointsTo::get`]).
#[must_use]
pub fn solve(constraints: &[Constraint], budget: &WorkBudget) -> PointsTo {
    let mut graph = SolverGraph::default();
    let mut deref = DerefConstraints::default();
    let mut work = Worklist::default();
    let mut cells = Cells::default();
    // What may be anywhere holds what may be anything: a load through an
    // unknown pointer reads an unknown value.
    if graph.add_pointee(UNKNOWN_LOC, UNKNOWN_LOC) {
        work.push(UNKNOWN_LOC);
    }

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
            Constraint::Points {
                ptr,
                object,
                offset,
            } => {
                let cell = cells.cell(object, Some(offset), &mut graph, &mut work);
                if graph.add_pointee(ptr, cell) {
                    work.push(ptr);
                }
            }
            Constraint::Offset { dst, src, by } => {
                deref.offsets.entry(src).or_default().push((dst, by));
                if graph.sets.contains_key(&src) {
                    work.push(src);
                }
            }
            Constraint::Width { ptr, bytes } => {
                deref.widths.entry(ptr).or_default().insert(bytes);
                if graph.sets.contains_key(&ptr) {
                    work.push(ptr);
                }
            }
            Constraint::Escape { ptr } => {
                deref.escapes.insert(ptr);
                if graph.sets.contains_key(&ptr) {
                    work.push(ptr);
                }
            }
            Constraint::Touch {
                ptr,
                reached,
                reach,
            } => {
                deref
                    .touches
                    .entry(ptr)
                    .or_default()
                    .insert((reached, reach));
                if graph.sets.contains_key(&ptr) {
                    work.push(ptr);
                }
            }
            Constraint::CopyCells { dst, src } => {
                for pointer in [dst, src] {
                    deref.copies.entry(pointer).or_default().insert((dst, src));
                    if graph.sets.contains_key(&pointer) {
                        work.push(pointer);
                    }
                }
            }
        }
    }
    // What may be anywhere has escaped: a store through an unknown pointer
    // hands its value to whoever holds that pointer.
    deref.escapes.insert(UNKNOWN_LOC);
    cells.drain(&mut graph, &mut work);

    // Safety budget. The difference-propagation worklist converges by monotone
    // set growth, but a crafted function with dense aliasing can drive
    // worst-case cubic work and pin a CPU core. Bound the work proportional to
    // the constraint count so normal functions always converge first. On
    // exceed the sets are not a fixpoint - some are missing members they would
    // have gained - so the relation is saturated rather than returned partial:
    // a partial may-relation answers "no alias" for exactly the pointers it had
    // not finished.
    let safety_bound = budget
        .limit(Budget::PointsToStepsPerConstraint)
        .saturating_mul(constraints.len().saturating_add(1));
    let mut exhaustion = None;
    let mut steps = 0usize;
    while let Some(loc) = work.pop() {
        // A pop is work of its own, whatever it propagates.
        steps = steps.saturating_add(1);
        // The budget counts *work*, not pops. A single pop propagates along
        // every outgoing copy edge and, for each load or store through `loc`,
        // does `pointees x destinations` work - so bounding pops left the cubic
        // term unbounded, and the guard could not fire before the analysis had
        // already done the work it was meant to prevent.
        if steps > safety_bound {
            warn!(
                "points-to analysis exceeded its step budget ({} constraints, bound {safety_bound}); every location may point anywhere",
                constraints.len()
            );
            exhaustion = Some(Exhaustion {
                budget: Budget::PointsToStepsPerConstraint,
                limit: safety_bound,
                demand: steps,
            });
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
        // Each block copy through `loc`, between every pair of cells its two
        // pointers may point at.
        if let Some(copies) = deref.copies.get(&loc) {
            for &(dst, src) in copies {
                let targets = graph.sets.get(&dst).cloned().unwrap_or_default();
                let sources = graph.sets.get(&src).cloned().unwrap_or_default();
                steps = steps.saturating_add(targets.len().saturating_mul(sources.len()));
                for &target in &targets {
                    for &source in &sources {
                        cells.copy(target, source, &mut graph, &mut work);
                    }
                }
            }
        }
        cells.drain(&mut graph, &mut work);
        // What an escaping pointer points at escapes.
        if deref.escapes.contains(&loc) {
            let pointees = graph.sets.get(&loc).cloned().unwrap_or_default();
            steps = steps.saturating_add(pointees.len());
            for pointee in pointees {
                cells.escape(pointee, &mut graph, &mut work);
            }
        }
        // What each call reaches through `loc`, of every cell it may point at.
        if let Some(touches) = deref.touches.get(&loc) {
            let pointees = graph.sets.get(&loc).cloned().unwrap_or_default();
            steps = steps.saturating_add(pointees.len().saturating_mul(touches.len()));
            for pointee in pointees {
                for &(reached, reach) in touches {
                    cells.touch(pointee, reached, reach, &mut graph, &mut work);
                }
            }
        }
        // Each access through `loc` at its width, on every cell it may reach.
        if let Some(widths) = deref.widths.get(&loc) {
            let pointees = graph.sets.get(&loc).cloned().unwrap_or_default();
            steps = steps.saturating_add(pointees.len().saturating_mul(widths.len()));
            for pointee in pointees {
                for &bytes in widths {
                    cells.access(pointee, bytes, &mut graph, &mut work);
                }
            }
        }
        // For each move `dst = loc + by`, every pointee moved that far is one
        // of `dst`'s.
        if let Some(moves) = deref.offsets.get(&loc) {
            let pointees = graph.sets.get(&loc).cloned().unwrap_or_default();
            steps = steps.saturating_add(pointees.len().saturating_mul(moves.len()));
            for pointee in pointees {
                for &(dst, by) in moves {
                    let moved = cells.moved(pointee, by, &mut graph, &mut work);
                    if graph.add_pointee(dst, moved) {
                        work.push(dst);
                        cells.bound_walk(dst, moved, &mut graph, &mut work);
                    }
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

    PointsTo {
        exhaustion,
        sets: graph.sets,
        cells: cells.names,
        escaped: cells.escaped,
        frame_escaped_from: cells.frame_escaped_from,
    }
}

/// The constraints of a call that lists every access it makes
/// ([`CallEffects::accesses`]), pushed onto `constraints`: an argument it does
/// not keep ([`Capture::keeps`]) reaches it only through the accesses it
/// states of it ([`CallLocation::Pointee`]), as far as each reaches at this
/// call ([`PointeeExtent::at_call`]), and only while it runs
/// ([`Constraint::Touch`]); every other operand escapes, as one passed to a
/// call that says nothing does. What it defines may point anywhere.
///
/// [`CallEffects::accesses`]: crate::ir::ops::CallEffects::accesses
/// [`Capture::keeps`]: crate::ir::ops::Capture::keeps
fn stated_call<T: Target>(
    constraints: &mut Vec<Constraint>,
    ir: &SsaFunction<T>,
    op: &SsaOp<T>,
    call: &CallData<T>,
) {
    let Some(effects) = call.effects.as_ref() else {
        return;
    };
    let accesses = effects.accesses().unwrap_or_default();
    let confined: Vec<(u32, SsaVarId)> = (0u32..)
        .zip(call.args.iter().copied())
        .filter(|(position, _)| !effects.captures.keeps(*position))
        .collect();
    // An operand escapes where it is used more often than it is confined.
    let mut uses: BTreeMap<SsaVarId, usize> = BTreeMap::new();
    op.for_each_use(|used| {
        let count = uses.entry(used).or_default();
        *count = count.saturating_add(1);
    });
    for (_, arg) in &confined {
        if let Some(count) = uses.get_mut(arg) {
            *count = count.saturating_sub(1);
        }
    }
    for (value, _) in uses.iter().filter(|(_, count)| **count > 0) {
        constraints.push(Constraint::Escape {
            ptr: value.as_u32(),
        });
    }
    for (position, arg) in confined {
        for access in accesses {
            let CallLocation::Pointee { arg: named, extent } = access.location else {
                continue;
            };
            if named != position {
                continue;
            }
            let bytes = match extent.at_call(&call.args, |count| ir.const_u64(count)) {
                // No bytes are none reached.
                PointeeExtent::Bytes(bytes) => match NonZeroU64::new(bytes) {
                    Some(bytes) => Some(bytes),
                    None => continue,
                },
                PointeeExtent::InArgument(_) | PointeeExtent::Unbounded => None,
            };
            constraints.push(Constraint::Touch {
                ptr: arg.as_u32(),
                reached: Reached::From { at: 0, bytes },
                reach: Reach {
                    writes: access.access.writes(),
                    reads: access.access.reads(),
                    keeps: false,
                },
            });
        }
    }
    for dest in op.defs() {
        constraints.push(Constraint::AddressOf {
            ptr: dest.as_u32(),
            target: UNKNOWN_LOC,
        });
    }
}

/// The constraints of the bytes of the caller's frame `call` states it
/// reaches ([`CallLocation::Frame`]), measured from the entry stack pointer
/// `stack`: what it writes there holds anything once it returns, and what
/// it reads escapes to it. Bytes measured from the frame's alignment are at
/// no offset from entry the relation knows, so the whole frame is reached.
fn frame_reaches<T: Target>(
    constraints: &mut Vec<Constraint>,
    call: &CallData<T>,
    stack: SsaVarId,
) {
    let Some(CallMemory::Only(accesses)) = call.effects.as_ref().map(|effects| &effects.memory)
    else {
        return;
    };
    for access in accesses {
        let CallLocation::Frame {
            origin,
            offset,
            bytes,
        } = access.location
        else {
            continue;
        };
        let Some(bytes) = NonZeroU64::new(bytes) else {
            continue;
        };
        constraints.push(Constraint::Touch {
            ptr: stack.as_u32(),
            reached: match origin {
                FrameOrigin::Entry => Reached::From {
                    at: offset,
                    bytes: Some(bytes),
                },
                FrameOrigin::Alignment => Reached::Object,
            },
            reach: Reach {
                writes: access.access.writes(),
                reads: access.access.reads(),
                keeps: false,
            },
        });
    }
}

/// The constraints of a call to a function `model` describes, pushed onto
/// `constraints`; `false`, pushing nothing, where the model does not say what
/// the call does with what it is handed.
///
/// A fill, a string copy, a length, a release or a read writes no pointer and
/// keeps none of its arguments; a block copy copies the bytes it is handed,
/// pointers included ([`Constraint::CopyCells`]); an allocation returns an
/// object of its own, named by the call ([`Object::Allocation`]), a
/// reallocation with the old object's bytes copied in; a function that
/// returns one of its arguments returns that pointer.
fn modelled_call<T: Target>(
    constraints: &mut Vec<Constraint>,
    call: &crate::ir::CallData<T>,
    site: Option<InstrId>,
    model: &LibraryModel,
) -> bool {
    let argument = |index: usize| call.args.get(index).map(|arg| arg.as_u32());
    let result = call.dest.map(|dest| dest.as_u32());
    match model {
        LibraryModel::Fill
        | LibraryModel::CopyString
        | LibraryModel::AppendString
        | LibraryModel::Length
        | LibraryModel::Free(_)
        | LibraryModel::NoWrites
        | LibraryModel::Identity { .. }
        | LibraryModel::DestroyHeap { .. } => {}
        // The block goes to the cell the out argument points at, which no
        // constraint here names without a value holding the block's
        // address; its result is a status. As an unmodeled call, its
        // arguments' pointees may hold anything.
        LibraryModel::Allocate(Allocation {
            delivery: Delivery::Out { .. },
            ..
        }) => return false,
        // It may store pointers in what its arguments reach and in data the
        // program let escape, which no constraint here names: as an
        // unmodeled call, its arguments' pointees may hold anything.
        LibraryModel::WritesData(_) => return false,
        LibraryModel::Copy => {
            if let (Some(dst), Some(src)) = (argument(0), argument(1)) {
                constraints.push(Constraint::CopyCells { dst, src });
            }
        }
        LibraryModel::Allocate(_)
        | LibraryModel::Duplicate(_)
        | LibraryModel::Reallocate(_)
        | LibraryModel::CreateHeap { .. } => {
            let (Some(result), Some(site)) = (result, site) else {
                return false;
            };
            constraints.push(Constraint::Points {
                ptr: result,
                object: Object::Allocation(site),
                offset: 0,
            });
            if let (LibraryModel::Reallocate(_), Some(old)) = (model, argument(0)) {
                constraints.push(Constraint::CopyCells {
                    dst: result,
                    src: old,
                });
            }
        }
        LibraryModel::NoReturn | LibraryModel::ProtectCode { .. } => return false,
        // A thrown object goes to the runtime, which hands it to whichever
        // handler takes it: the object a handler's `__cxa_begin_catch` returns
        // is no value of its call, so these are read as unmodeled calls.
        LibraryModel::Throw { .. }
        | LibraryModel::Rethrow
        | LibraryModel::Resume
        | LibraryModel::BeginCatch
        | LibraryModel::EndCatch { .. } => return false,
    }
    if let (Some(result), Some(returned)) = (result, model.returned_argument().and_then(argument)) {
        constraints.push(Constraint::Copy {
            dst: result,
            src: returned,
        });
    }
    true
}

/// Lower bound of the cell locations ([`Constraint::Points`]), which the
/// solver mints as it names objects' bytes: above every dense SSA value id,
/// below the synthetic band.
pub const CELL_LOC_BASE: Loc = 0x8000_0000;

/// Lower bound of the synthetic object-location range. SSA value ids are dense
/// from zero, so any location at or above this is a synthetic stack/argument
/// cell rather than a pointer-valued SSA id.
const SYNTHETIC_LOC_BASE: Loc = 0xF000_0000;
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
/// understand - an under-approximation, and an unsound alias proof.
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
/// SSA value ids are pointer locations. The entry stack pointer is the frame's
/// first byte, a symbol's address the image's byte at it, and a sum with a
/// constant the same object that many bytes on ([`Object`]); every access of
/// every operation loads or stores its cell at its width. A managed front
/// end's locals and arguments get stable synthetic locations so two
/// addresses of the same slot coincide. The analysis
/// is **field-sensitive**: `&object.field` resolves to a distinct abstract cell
/// per `(object, field)`, so two different fields of the same object do not
/// alias. A field whose host reports no member index via
/// [`Target::field_member_index`] (or once the field-cell range is exhausted)
/// falls back to the sound field-insensitive whole-object alias.
///
/// Phi nodes are walked alongside instructions because they live in a separate
/// vector; missing them would give every merge-defined pointer - loop-carried
/// pointers and if/else results - an empty (and unsound) points-to set. Any op
/// the extractor does not model has its defs seeded with [`UNKNOWN_LOC`].
///
/// # Arguments
///
/// * `ir` - The SSA function to extract constraints from.
/// * `budget` - The work budget the solve is bounded by ([`solve`]).
/// * `models` - What the functions the code calls do, as the host states it
///   ([`CallModels`]): a call no model describes may do anything with what
///   it is handed.
///
/// # Returns
///
/// The solved [`PointsTo`] relation for the function's SSA value ids and
/// synthetic object cells; see [`solve`] for what exhaustion does to it.
/// Constraints the function was solved to before are not solved again
/// ([`SsaFunction::solved_points_to`]).
#[must_use]
pub fn analyze_function<T: Target>(
    ir: &SsaFunction<T>,
    budget: &WorkBudget,
    models: Option<&dyn CallModels>,
) -> Arc<PointsTo> {
    let mut constraints: Vec<Constraint> = Vec::new();
    let mut field_cells = FieldCells::default();
    // The stack pointer the function is entered with is its frame's first
    // byte.
    if let Some(stack) = ir.entry_stack_pointer() {
        constraints.push(Constraint::Points {
            ptr: stack.as_u32(),
            object: Object::Frame,
            offset: 0,
        });
    }
    // A call that returns twice resumes code no edge leads to, which reads
    // the frame as a later call leaves it: every byte of the frame escapes.
    if let Some(stack) = ir.entry_stack_pointer()
        && ir.iter_instructions().any(|(_, _, instruction)| {
            instruction
                .op()
                .as_call()
                .and_then(|call| call.effects.as_ref())
                .is_some_and(|effects| effects.returns_twice)
        })
    {
        constraints.push(Constraint::Touch {
            ptr: stack.as_u32(),
            reached: Reached::Object,
            reach: Reach {
                writes: true,
                reads: true,
                keeps: true,
            },
        });
    }
    for (block_index, block) in ir.blocks().iter().enumerate() {
        // A phi is a merge: its result may hold any of its operands' values.
        // Phis live in a separate vector from instructions, so iterating only
        // `instructions()` silently gave every merge-defined pointer - which is
        // every loop-carried pointer, and every pointer out of an if/else - an
        // empty points-to set.
        for phi in block.phi_nodes() {
            for operand in phi.operands() {
                constraints.push(Constraint::Copy {
                    dst: phi.result().as_u32(),
                    src: operand.value().as_u32(),
                });
            }
        }
        for (instruction_index, instruction) in block.instructions().iter().enumerate() {
            let op = instruction.op();
            // The bytes of the frame a call states it reaches, whatever else
            // is known of it.
            if let SsaOp::Call(call) = op
                && let Some(stack) = ir.entry_stack_pointer()
            {
                frame_reaches(&mut constraints, call, stack);
            }
            // A call whose function the host models does what the model says,
            // and nothing else with what it is handed.
            if let SsaOp::Call(call) = op
                && let Some(model) =
                    models.and_then(|models| models.model(block_index, instruction_index))
                && modelled_call(&mut constraints, call, instruction.id(), &model)
            {
                // The model says what its result is, and nothing of what the
                // call returns beside it.
                for result in &call.results {
                    constraints.push(Constraint::AddressOf {
                        ptr: result.value.as_u32(),
                        target: UNKNOWN_LOC,
                    });
                }
                continue;
            }
            // Every access the operation makes, at its width: a write stores
            // its value, or an unknown one where it states none or also
            // reads (an exchange, a block copy), and a read loads into its
            // value, or into each result of an operation that also writes.
            let mut loaded: Vec<SsaVarId> = Vec::new();
            let mut accessed = false;
            op.for_each_addressed_access(|access| {
                accessed = true;
                let ptr = access.addr.as_u32();
                let bytes = access
                    .width_bits
                    .filter(|bits| bits.is_multiple_of(8))
                    .map(|bits| bits / 8);
                constraints.push(Constraint::Width { ptr, bytes });
                if access.writes {
                    let src = access
                        .value
                        .filter(|_| !access.reads)
                        .map_or(UNKNOWN_LOC, SsaVarId::as_u32);
                    constraints.push(Constraint::Store { ptr, src });
                }
                if access.reads {
                    let read: Vec<SsaVarId> = match access.value.filter(|_| !access.writes) {
                        Some(value) => vec![value],
                        None => op.defs().collect(),
                    };
                    for dst in read {
                        constraints.push(Constraint::Load {
                            dst: dst.as_u32(),
                            ptr,
                        });
                        loaded.push(dst);
                    }
                }
            });
            match op {
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
                // An address computed from a base points into the base's
                // object, its displacement on, or anywhere in it where it
                // adds an index.
                SsaOp::PtrAdd {
                    dest,
                    base,
                    index,
                    offset,
                    ..
                } => constraints.push(Constraint::Offset {
                    dst: dest.as_u32(),
                    src: base.as_u32(),
                    by: index.map_or(Shift::Bytes(*offset), |_| Shift::Unknown),
                }),
                // A sum with a constant is a pointer moved that far; a sum of
                // two values may be either moved by the other.
                SsaOp::Add {
                    dest, left, right, ..
                }
                | SsaOp::AddOvf {
                    dest, left, right, ..
                } => {
                    let moved = |src: SsaVarId, by: Shift| Constraint::Offset {
                        dst: dest.as_u32(),
                        src: src.as_u32(),
                        by,
                    };
                    match (ir.const_i64(*left), ir.const_i64(*right)) {
                        (_, Some(bytes)) => constraints.push(moved(*left, Shift::Bytes(bytes))),
                        (Some(bytes), None) => constraints.push(moved(*right, Shift::Bytes(bytes))),
                        (None, None) => {
                            constraints.push(moved(*left, Shift::Unknown));
                            constraints.push(moved(*right, Shift::Unknown));
                        }
                    }
                }
                // A difference is the left pointer moved back by the right.
                SsaOp::Sub {
                    dest, left, right, ..
                }
                | SsaOp::SubOvf {
                    dest, left, right, ..
                } => {
                    let by = ir
                        .const_i64(*right)
                        .and_then(i64::checked_neg)
                        .map_or(Shift::Unknown, Shift::Bytes);
                    constraints.push(Constraint::Offset {
                        dst: dest.as_u32(),
                        src: left.as_u32(),
                        by,
                    });
                }
                // Any other arithmetic on a pointer leaves it somewhere in its
                // object (an alignment mask, a tag); on integers, nothing.
                SsaOp::Mul { .. }
                | SsaOp::MulOvf { .. }
                | SsaOp::Div { .. }
                | SsaOp::Rem { .. }
                | SsaOp::Neg { .. }
                | SsaOp::And { .. }
                | SsaOp::Or { .. }
                | SsaOp::Xor { .. }
                | SsaOp::Not { .. }
                | SsaOp::Shl { .. }
                | SsaOp::Shr { .. }
                | SsaOp::Rol { .. }
                | SsaOp::Ror { .. } => {
                    for dest in op.defs() {
                        for src in op.uses() {
                            constraints.push(Constraint::Offset {
                                dst: dest.as_u32(),
                                src: src.as_u32(),
                                by: Shift::Unknown,
                            });
                        }
                    }
                }
                // Stack memory a function allocates as it runs (`alloca`) is an
                // object of its own, named by the instruction that allocates it.
                SsaOp::LocalAlloc { dest, .. } => match instruction.id() {
                    Some(site) => constraints.push(Constraint::Points {
                        ptr: dest.as_u32(),
                        object: Object::Allocation(site),
                        offset: 0,
                    }),
                    None => constraints.push(Constraint::AddressOf {
                        ptr: dest.as_u32(),
                        target: UNKNOWN_LOC,
                    }),
                },
                // A branch compares and goes; it hands nothing on.
                SsaOp::Jump { .. }
                | SsaOp::Branch { .. }
                | SsaOp::BranchCmp { .. }
                | SsaOp::BranchFlags { .. }
                | SsaOp::Switch { .. } => {}
                // A comparison is no pointer, nor a number.
                SsaOp::Ceq { .. }
                | SsaOp::Clt { .. }
                | SsaOp::Cgt { .. }
                | SsaOp::BoolAnd { .. }
                | SsaOp::BoolOr { .. }
                | SsaOp::BoolXor { .. }
                | SsaOp::BoolNot { .. } => {}
                // A symbol's address is that byte of the image; a number points
                // nowhere known, and nothing derives an object from it.
                SsaOp::Const { dest, value } => {
                    if let ConstValue::Symbol(symbol) = value {
                        match T::symbol_address(symbol) {
                            Some(address) => constraints.push(Constraint::Points {
                                ptr: dest.as_u32(),
                                object: Object::Image,
                                offset: address,
                            }),
                            None => constraints.push(Constraint::AddressOf {
                                ptr: dest.as_u32(),
                                target: UNKNOWN_LOC,
                            }),
                        }
                    }
                }
                // Field-sensitive: `&object.field` resolves to a distinct
                // synthetic cell per `(object, field)`. The address value first
                // takes the object's pointees (so it stays anchored to the same
                // object), then additionally points at the field cell - keeping
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
                // A call that lists what it touches reaches what it does not
                // keep only as it says.
                SsaOp::Call(call)
                    if call
                        .effects
                        .as_ref()
                        .is_some_and(|effects| effects.accesses().is_some()) =>
                {
                    stated_call(&mut constraints, ir, op, call);
                }
                // Anything not modelled above - a call no model describes, a
                // return, `Select` - may hand what it is given anywhere, and
                // defines a value whose pointees this module cannot determine.
                // Recording `UNKNOWN_LOC` keeps "unmodelled" distinguishable from
                // "points nowhere"; the latter would let a consumer prove NoAlias
                // against everything.
                other => {
                    // A load or store says all it does with its operands
                    // through its accesses; a call or anything else may hand
                    // them on.
                    let described = accessed && !matches!(other, SsaOp::Call(_));
                    for src in other.uses().into_iter().filter(|_| !described) {
                        constraints.push(Constraint::Escape { ptr: src.as_u32() });
                    }
                    for dest in other.defs().filter(|dest| !loaded.contains(dest)) {
                        constraints.push(Constraint::AddressOf {
                            ptr: dest.as_u32(),
                            target: UNKNOWN_LOC,
                        });
                    }
                }
            }
        }
    }
    ir.solved_points_to().solve(constraints, budget)
}

/// The relations a function's points-to constraints were last solved to,
/// kept with the function ([`SsaFunction::solved_points_to`]) so the same
/// constraints are not solved again.
///
/// A relation is what its constraints and budget solve to, whatever the
/// function: one is reused only where both are equal to a kept solve's, so a
/// reuse is that solve. The pipeline runs a pass again after any other pass
/// changes the function, and each consumer of the relation asks for it of the
/// function as it then stands, so most solves are of constraints solved
/// before: the state a function was last solved in is the one asked of
/// again, so the latest [`Self::KEPT`] solve is kept, in one store a function
/// and its copies share. A solve of an earlier state would hold memory for
/// as long as the function lives, and is seldom asked of again.
#[derive(Default)]
pub struct SolvedPointsTo {
    /// The latest solves, the most recent first: one store for a function
    /// and every copy of it ([`Clone`]).
    kept: Arc<Mutex<Vec<Arc<Solved>>>>,
}

/// One kept solve ([`SolvedPointsTo`]).
struct Solved {
    /// The constraints solved.
    constraints: Vec<Constraint>,
    /// The budget they were solved within.
    budget: WorkBudget,
    /// What they solved to.
    relation: Arc<PointsTo>,
}

impl SolvedPointsTo {
    /// How many solves are kept.
    pub const KEPT: usize = 1;

    /// The relation `constraints` solve to within `budget`: a kept solve's
    /// where they are its constraints and budget, solved and kept otherwise.
    ///
    /// # Arguments
    ///
    /// * `constraints` - The constraints to solve.
    /// * `budget` - The budget to solve them within.
    #[must_use]
    pub fn solve(&self, constraints: Vec<Constraint>, budget: &WorkBudget) -> Arc<PointsTo> {
        let found = self.kept.lock().ok().and_then(|kept| {
            kept.iter()
                .find(|solved| solved.budget == *budget && solved.constraints == constraints)
                .map(|solved| Arc::clone(&solved.relation))
        });
        if let Some(relation) = found {
            return relation;
        }
        let relation = Arc::new(solve(&constraints, budget));
        if let Ok(mut kept) = self.kept.lock() {
            kept.insert(
                0,
                Arc::new(Solved {
                    constraints,
                    budget: *budget,
                    relation: Arc::clone(&relation),
                }),
            );
            kept.truncate(Self::KEPT);
        }
        relation
    }
}

impl Clone for SolvedPointsTo {
    /// The same store: a copy of a function, which a pass runs on and the
    /// pipeline keeps or drops, reuses the solves of the function's other
    /// copies, and they its.
    fn clone(&self) -> Self {
        Self {
            kept: Arc::clone(&self.kept),
        }
    }
}

impl fmt::Debug for SolvedPointsTo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kept = self.kept.lock().map_or(0, |kept| kept.len());
        f.debug_struct("SolvedPointsTo")
            .field("kept", &kept)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{
            cache::FunctionAnalyses,
            context::AnalysisContext,
            execution::{AllocatorFamily, Conditions, Operand},
        },
        ir::{
            AccessMode, CallAccess, CallData, CallEffects, CallInputKey, CallTermination, Callee,
            Capture, SsaBlock, SsaInstruction, SsaVarId,
            ops::AccessQualifiers,
            phi::{PhiNode, PhiOperand},
            variable::{DefSite, VariableOrigin, VariableRole},
        },
        testing::{MACHINE, MOCK_FIELD_UNRESOLVED, MockTarget, MockType},
    };

    /// Declares a pointer-typed variable defined at `(block, instruction)`.
    fn ptr_var(
        ir: &mut SsaFunction<MockTarget>,
        origin: VariableOrigin,
        block: usize,
        instruction: usize,
    ) -> SsaVarId {
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
    /// `instructions()` gave every merge-defined pointer an empty set - which is
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
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ir.add_block(b3);
        ir.recompute_uses();

        let result = analyze_function(&ir, &WorkBudget::DEFAULT, None);
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
        // An operation the extractor does not model, on an address.
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Popcount {
            dest: sum,
            src: a,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ir.add_block(block);
        ir.recompute_uses();

        let result = analyze_function(&ir, &WorkBudget::DEFAULT, None);
        assert!(
            result.get(sum.as_u32()).contains(&UNKNOWN_LOC),
            "an unmodelled pointer definition must be unconstrained, not empty"
        );
    }

    /// A copy edge carries the source's taken address into the destination's set.
    #[test]
    fn copy_propagates_address_of() {
        // p = &x; q = p  =>  q -> x
        let pts = solve(
            &[
                Constraint::AddressOf { ptr: 1, target: 10 },
                Constraint::Copy { dst: 2, src: 1 },
            ],
            &WorkBudget::DEFAULT,
        );
        assert!(pts.get(2).contains(&10));
    }

    /// A load resolves the pointer one level and inherits its pointee's pointees.
    #[test]
    fn load_dereferences_pointer() {
        // p -> a, a -> b; q = *p  =>  q -> b
        let pts = solve(
            &[
                Constraint::AddressOf { ptr: 1, target: 20 }, // p -> a(20)
                Constraint::AddressOf {
                    ptr: 20,
                    target: 30,
                }, // a -> b(30)
                Constraint::Load { dst: 2, ptr: 1 },          // q = *p
            ],
            &WorkBudget::DEFAULT,
        );
        assert!(pts.get(2).contains(&30));
    }

    /// A store flows the source's pointees into every pointee of the pointer.
    #[test]
    fn store_writes_through_pointer() {
        // p -> a, s -> b; *p = s  =>  a -> b
        let pts = solve(
            &[
                Constraint::AddressOf { ptr: 1, target: 20 }, // p -> a(20)
                Constraint::AddressOf { ptr: 3, target: 30 }, // s -> b(30)
                Constraint::Store { ptr: 1, src: 3 },         // *p = s
            ],
            &WorkBudget::DEFAULT,
        );
        assert!(pts.get(20).contains(&30));
    }

    /// A 64-link copy chain propagates the seeded address all the way to the tail.
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
        let pts = solve(&constraints, &WorkBudget::DEFAULT);
        assert!(pts.get(64).contains(&500));
        assert!(pts.get(32).contains(&500));
    }

    /// A solve that runs out of budget answers every location "anywhere",
    /// never the partial set it had reached, and says why.
    ///
    /// A may-relation cut off before its fixpoint is missing members some sets
    /// would have gained, so a partial answer is a false "no alias" for exactly
    /// the pointers it had not finished.
    #[test]
    fn an_exhausted_solve_saturates_rather_than_truncating() {
        let mut constraints = vec![Constraint::AddressOf {
            ptr: 0,
            target: 777,
        }];
        for dst in 1u32..=64 {
            constraints.push(Constraint::Copy {
                dst,
                src: dst.saturating_sub(1),
            });
        }
        let starved = WorkBudget::DEFAULT.with_limit(Budget::PointsToStepsPerConstraint, 0);
        let pts = solve(&constraints, &starved);
        let exhaustion = pts.exhaustion().expect("a zero budget cannot converge");
        assert_eq!(exhaustion.budget, Budget::PointsToStepsPerConstraint);
        assert_eq!(pts.get(64), &BTreeSet::from([UNKNOWN_LOC]));
        assert_eq!(
            pts.get(9999),
            &BTreeSet::from([UNKNOWN_LOC]),
            "any location"
        );
        assert_eq!(
            pts.iter().count(),
            0,
            "it lists nothing it cannot vouch for"
        );
        assert_eq!(solve(&constraints, &WorkBudget::DEFAULT).exhaustion(), None);
    }

    /// A large but legitimate chain stays under the step budget and is not truncated.
    #[test]
    fn large_chain_stays_within_step_budget_without_truncation() {
        // The step budget is `64 * len + 64`. A legitimately large function
        // (here a 2000-link chain, 2001 constraints, needing ~2000 worklist
        // steps) is well under that bound, so the safety stop must NOT truncate
        // it - the tail still resolves the seeded address. This guards the budget
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
        let pts = solve(&constraints, &WorkBudget::DEFAULT);
        assert!(
            pts.get(2000).contains(&777),
            "tail must resolve the address"
        );
        assert!(pts.get(1000).contains(&777));
    }

    /// The objects and bytes of `loc`'s pointees, a non-cell as `None`.
    fn cells_of(relation: &PointsTo, loc: Loc) -> Vec<Option<(Object, Option<i64>)>> {
        relation
            .get(loc)
            .iter()
            .map(|pointee| relation.cell(*pointee))
            .collect()
    }

    /// **Bytes of one object are cells of their own.** A pointer stored at
    /// the frame's byte 8 is read back through another address of byte 8,
    /// and not through byte 0: two fields of one record.
    #[test]
    fn bytes_of_one_object_are_cells_of_their_own() {
        let (frame, at_8, again_8, table, read, other) = (1, 2, 3, 4, 5, 6);
        let relation = solve(
            &[
                Constraint::Points {
                    ptr: frame,
                    object: Object::Frame,
                    offset: -128,
                },
                Constraint::Offset {
                    dst: at_8,
                    src: frame,
                    by: Shift::Bytes(8),
                },
                Constraint::Points {
                    ptr: table,
                    object: Object::Image,
                    offset: 0x1000,
                },
                Constraint::Store {
                    ptr: at_8,
                    src: table,
                },
                Constraint::Offset {
                    dst: again_8,
                    src: frame,
                    by: Shift::Bytes(8),
                },
                Constraint::Load {
                    dst: read,
                    ptr: again_8,
                },
                Constraint::Load {
                    dst: other,
                    ptr: frame,
                },
            ],
            &WorkBudget::DEFAULT,
        );
        assert_eq!(
            cells_of(&relation, read),
            vec![Some((Object::Image, Some(0x1000)))]
        );
        assert!(
            relation.get(other).is_empty(),
            "byte 0 holds nothing stored"
        );
    }

    /// **An object moved by an unknown amount is one cell.** A store through
    /// the frame moved by an index nothing bounds may write any byte of it,
    /// so a read of byte 8 sees it; the image moved so may be anywhere.
    #[test]
    fn an_object_moved_by_an_unknown_amount_is_one_cell() {
        let (frame, anywhere, value, read, image, somewhere, at_8) = (1, 2, 3, 4, 5, 6, 7);
        let relation = solve(
            &[
                Constraint::Points {
                    ptr: frame,
                    object: Object::Frame,
                    offset: -128,
                },
                Constraint::Offset {
                    dst: anywhere,
                    src: frame,
                    by: Shift::Unknown,
                },
                Constraint::Points {
                    ptr: value,
                    object: Object::Image,
                    offset: 0x2000,
                },
                Constraint::Store {
                    ptr: anywhere,
                    src: value,
                },
                Constraint::Offset {
                    dst: at_8,
                    src: frame,
                    by: Shift::Bytes(8),
                },
                Constraint::Load {
                    dst: read,
                    ptr: at_8,
                },
                Constraint::Points {
                    ptr: image,
                    object: Object::Image,
                    offset: 0x3000,
                },
                Constraint::Offset {
                    dst: somewhere,
                    src: image,
                    by: Shift::Unknown,
                },
            ],
            &WorkBudget::DEFAULT,
        );
        assert_eq!(
            cells_of(&relation, read),
            vec![Some((Object::Image, Some(0x2000)))]
        );
        assert_eq!(relation.get(somewhere), &BTreeSet::from([UNKNOWN_LOC]));
    }

    /// **A walk collapses its object and converges.** A pointer a loop moves
    /// four bytes a trip names a new byte each round until the object is one
    /// cell; a load through an unknown pointer reads an unknown value.
    #[test]
    fn a_walk_collapses_its_object_and_converges() {
        let (start, walked, next, read, unknown_read) = (1, 2, 3, 4, 5);
        let relation = solve(
            &[
                Constraint::Points {
                    ptr: start,
                    object: Object::Allocation(InstrId::from_raw(1).expect("nonzero")),
                    offset: 0,
                },
                Constraint::Copy {
                    dst: walked,
                    src: start,
                },
                Constraint::Offset {
                    dst: next,
                    src: walked,
                    by: Shift::Bytes(4),
                },
                Constraint::Copy {
                    dst: walked,
                    src: next,
                },
                Constraint::Load {
                    dst: read,
                    ptr: walked,
                },
                Constraint::Load {
                    dst: unknown_read,
                    ptr: UNKNOWN_LOC,
                },
            ],
            &WorkBudget::DEFAULT,
        );
        assert!(relation.exhaustion().is_none(), "converged");
        assert!(
            relation.get(walked).iter().any(|pointee| relation
                .cell(*pointee)
                .is_some_and(|(_, byte)| byte.is_none())),
            "the walked object is one cell"
        );
        assert_eq!(relation.get(unknown_read), &BTreeSet::from([UNKNOWN_LOC]));
    }

    /// A cyclic store-then-load through a self-referential pointer reaches a fixpoint.
    #[test]
    fn store_load_through_pointer_cycle_converges() {
        // A store-then-load cycle through a self-referential pointer must reach a
        // fixpoint: p -> a; *p = q where q -> a; r = *p  =>  pts(a) ⊇ {a} and
        // r -> a. The store grows pts(a) (a now points at itself), which feeds
        // back through the load - a genuine cyclic dependency.
        let pts = solve(
            &[
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
            ],
            &WorkBudget::DEFAULT,
        );
        // a now points at itself (the store wrote q's pointee a into a).
        assert!(pts.get(100).contains(&100));
        // r loaded a's contents, which include a.
        assert!(pts.get(3).contains(&100));
    }

    /// A three-link copy chain resolves the address while an unrelated location stays empty.
    #[test]
    fn transitive_copy_chain_converges() {
        // a = &x; b = a; c = b  =>  c -> x
        let pts = solve(
            &[
                Constraint::AddressOf { ptr: 1, target: 99 },
                Constraint::Copy { dst: 2, src: 1 },
                Constraint::Copy { dst: 3, src: 2 },
            ],
            &WorkBudget::DEFAULT,
        );
        assert!(pts.get(3).contains(&99));
        assert!(pts.get(4).is_empty());
    }

    /// Extraction turns `LoadLocalAddr` plus a copy into both pointers aliasing the local.
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
        let pts = analyze_function(&ir, &WorkBudget::DEFAULT, None);
        let local5 = LOCAL_LOC_BASE.saturating_add(5);
        assert!(pts.get(p.as_u32()).contains(&local5));
        assert!(pts.get(q.as_u32()).contains(&local5));
    }

    /// Distinct fields of one object hold disjoint field cells yet both resolve through the object.
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

        let pts = analyze_function(&ir, &WorkBudget::DEFAULT, None);
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

    /// A field with no recovered member index aliases the whole object, minting no field cell.
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

        let pts = analyze_function(&ir, &WorkBudget::DEFAULT, None);
        let local7 = LOCAL_LOC_BASE.saturating_add(7);
        let set = pts.get(field_dest.as_u32());
        // The fallback aliases the whole object: only the object's pointee, no
        // synthesized field cell.
        assert!(set.contains(&local7));
        assert!(set.iter().all(|loc| *loc < FIELD_LOC_BASE));
    }

    /// The cell of a pointer to one place, keyed by the pointer value.
    fn cell_of(base: u32) -> AliasKey {
        AliasKey {
            base: AddressBase::Value(SsaVarId::from_index(base as usize)),
            index: None,
            stride_bytes: 0,
            offset_bits: 0,
        }
    }

    /// **A call reaches the bytes of the local it is handed, not the
    /// function's own.** A shell routine hands `setstackmark` its `stackmark`
    /// at frame byte `48`, which the callee fills three words of: a call may
    /// write the mark's words at `48` and `56`, read off the entry stack
    /// pointer, and a byte above it; the register the routine spilled at
    /// `20`, below, it may not. The control: handed the frame's first byte,
    /// a call reaches the spill too.
    #[test]
    fn a_call_reaches_the_bytes_of_the_local_it_is_handed() {
        let clobbered = |handed: i64| {
            let (sp, mark) = (1, 2);
            let relation = solve(
                &[
                    Constraint::Points {
                        ptr: sp,
                        object: Object::Frame,
                        offset: -128,
                    },
                    Constraint::Offset {
                        dst: mark,
                        src: sp,
                        by: Shift::Bytes(handed),
                    },
                    Constraint::Escape { ptr: mark },
                ],
                &WorkBudget::DEFAULT,
            );
            let escape = CallEscape::new(Arc::new(relation), [mark]);
            [48, 56, 100, 20].map(|byte: i64| {
                escape.clobbers(&AliasKey {
                    offset_bits: byte * 8,
                    ..cell_of(sp)
                })
            })
        };
        assert_eq!(clobbered(48), [true, true, true, false]);
        assert_eq!(clobbered(0), [true; 4], "the control");
    }

    /// **A call keeps the cells its arguments cannot reach.** Value 1 points
    /// at 100 and value 9 at 200; only 9 is handed to a call, so a call may
    /// write 200 and cannot write 100. A base nothing is known about proves
    /// nothing, and is clobbered.
    #[test]
    fn a_call_clobbers_only_what_escapes_to_it() {
        let points_to = solve(
            &[
                Constraint::AddressOf {
                    ptr: 1,
                    target: 100,
                },
                Constraint::AddressOf {
                    ptr: 9,
                    target: 200,
                },
            ],
            &WorkBudget::DEFAULT,
        );
        let escape = CallEscape::new(Arc::new(points_to), [9u32]);
        assert!(!escape.clobbers(&cell_of(1)), "100 never reaches a call");
        assert!(escape.clobbers(&cell_of(9)), "200 is handed to one");
        assert!(
            escape.clobbers(&cell_of(5)),
            "an unknown base proves nothing"
        );
        assert!(
            CallEscape::default().clobbers(&cell_of(1)),
            "control: with no points-to at all, every call clobbers every cell"
        );
    }

    /// **Overlapping accesses make their object one cell.** A word stored at
    /// byte 0 and a byte read at byte 2 touch the same bytes at different
    /// starts, so the read sees the store; two accesses of the same bytes
    /// stay a cell of their own. The control: the read at byte 4, which no
    /// store touches, sees nothing.
    #[test]
    fn overlapping_accesses_make_their_object_one_cell() {
        let read_at = |at: i64, bytes: u32| {
            let (frame, inner, value, read) = (1, 2, 3, 4);
            let relation = solve(
                &[
                    Constraint::Points {
                        ptr: frame,
                        object: Object::Frame,
                        offset: -128,
                    },
                    Constraint::Points {
                        ptr: value,
                        object: Object::Image,
                        offset: 0x1000,
                    },
                    Constraint::Width {
                        ptr: frame,
                        bytes: Some(4),
                    },
                    Constraint::Store {
                        ptr: frame,
                        src: value,
                    },
                    Constraint::Offset {
                        dst: inner,
                        src: frame,
                        by: Shift::Bytes(at),
                    },
                    Constraint::Width {
                        ptr: inner,
                        bytes: Some(bytes),
                    },
                    Constraint::Load {
                        dst: read,
                        ptr: inner,
                    },
                ],
                &WorkBudget::DEFAULT,
            );
            relation.get(read).is_empty()
        };
        assert!(!read_at(2, 1), "a byte inside the stored word sees it");
        assert!(!read_at(0, 4), "the same word sees it");
        assert!(read_at(4, 4), "the control: the next word does not");
    }

    /// **What escapes may be written by anyone.** A frame byte whose address
    /// is handed away holds anything after; one never handed away holds only
    /// what the function stored.
    #[test]
    fn what_escapes_may_be_written_by_anyone() {
        let read = |escapes: bool| {
            let (frame, read) = (1, 2);
            let mut constraints = vec![
                Constraint::Points {
                    ptr: frame,
                    object: Object::Frame,
                    offset: -128,
                },
                Constraint::Load {
                    dst: read,
                    ptr: frame,
                },
            ];
            if escapes {
                constraints.push(Constraint::Escape { ptr: frame });
            }
            solve(&constraints, &WorkBudget::DEFAULT)
                .get(read)
                .contains(&UNKNOWN_LOC)
        };
        assert!(read(true));
        assert!(!read(false), "the control: kept to itself");
    }

    /// **Of the frame, what escapes is the locals whose addresses do.** A
    /// shell routine keeps the global offset table's address in a frame
    /// byte (`-108`), hands its `jmploc` (`-80`) to `handler`, and anything may
    /// then write through what it reads back from the `jmploc`: the kept
    /// byte holds the table's address alone. The controls: a byte above the
    /// escaped local is part of what escapes, and so is every byte from a
    /// lower escape up.
    #[test]
    fn of_the_frame_what_escapes_is_the_locals_whose_addresses_do() {
        let read = |kept: i64, lowest: Option<i64>| {
            let (spill, local, table, read, anywhere, other, low) = (1, 2, 3, 4, 5, 6, 7);
            let mut constraints = vec![
                Constraint::Points {
                    ptr: spill,
                    object: Object::Frame,
                    offset: kept,
                },
                Constraint::Points {
                    ptr: local,
                    object: Object::Frame,
                    offset: -80,
                },
                Constraint::Points {
                    ptr: table,
                    object: Object::Image,
                    offset: 0x2_fd54,
                },
                Constraint::Store {
                    ptr: spill,
                    src: table,
                },
                Constraint::Escape { ptr: local },
                Constraint::Load {
                    dst: anywhere,
                    ptr: local,
                },
                Constraint::Points {
                    ptr: other,
                    object: Object::Image,
                    offset: 0x1000,
                },
                Constraint::Store {
                    ptr: anywhere,
                    src: other,
                },
                Constraint::Load {
                    dst: read,
                    ptr: spill,
                },
            ];
            if let Some(offset) = lowest {
                constraints.push(Constraint::Points {
                    ptr: low,
                    object: Object::Frame,
                    offset,
                });
                constraints.push(Constraint::Escape { ptr: low });
            }
            let relation = solve(&constraints, &WorkBudget::DEFAULT);
            let held = relation.get(read);
            !held.is_empty() && !held.contains(&UNKNOWN_LOC)
        };
        assert!(read(-108, None), "a byte below the escaped local is kept");
        assert!(!read(-68, None), "the control: a byte above it escaped");
        assert!(
            !read(-108, Some(-120)),
            "the control: a lower escape reaches it"
        );
    }

    /// **Escaped memory moved anywhere is anywhere.** A shell routine keeps
    /// a table's address in a frame byte (`-108`) below a local whose address
    /// escapes (`-80`), and a value it reads through a pointer that may be
    /// anywhere, which may be that local's address, is moved by an amount
    /// nothing bounds (a `ctype` lookup's index): it may point anywhere, and
    /// the kept byte still holds the table's address alone.
    #[test]
    fn escaped_memory_moved_anywhere_is_anywhere() {
        let (spill, local, table, read, anywhere, loaded, moved) = (1, 2, 3, 4, 5, 6, 7);
        let relation = solve(
            &[
                Constraint::Points {
                    ptr: spill,
                    object: Object::Frame,
                    offset: -108,
                },
                Constraint::Points {
                    ptr: local,
                    object: Object::Frame,
                    offset: -80,
                },
                Constraint::Points {
                    ptr: table,
                    object: Object::Image,
                    offset: 0x1_1224,
                },
                Constraint::Store {
                    ptr: spill,
                    src: table,
                },
                Constraint::Escape { ptr: local },
                Constraint::Store {
                    ptr: local,
                    src: local,
                },
                Constraint::Load {
                    dst: anywhere,
                    ptr: local,
                },
                Constraint::Load {
                    dst: loaded,
                    ptr: anywhere,
                },
                Constraint::Offset {
                    dst: moved,
                    src: loaded,
                    by: Shift::Unknown,
                },
                Constraint::Load {
                    dst: read,
                    ptr: spill,
                },
            ],
            &WorkBudget::DEFAULT,
        );
        assert!(relation.get(moved).contains(&UNKNOWN_LOC));
        let held = relation.get(read);
        assert!(!held.is_empty() && !held.contains(&UNKNOWN_LOC));
    }

    /// **A block copy copies each byte's pointer to the byte as far on.** A
    /// table's address at byte 8 of a frame record, copied into an
    /// allocation at byte 64, is at its byte 72; its byte 64 holds nothing
    /// copied. A copy within one object to another place in it makes the
    /// object one cell (the frame here, copied onto itself 16 bytes on).
    #[test]
    fn a_block_copy_copies_each_bytes_pointer_as_far_on() {
        let (from, from_8, to, to_8, table, read, other) = (1, 2, 3, 4, 5, 6, 7);
        let relation = solve(
            &[
                Constraint::Points {
                    ptr: from,
                    object: Object::Frame,
                    offset: -128,
                },
                Constraint::Offset {
                    dst: from_8,
                    src: from,
                    by: Shift::Bytes(8),
                },
                Constraint::Points {
                    ptr: table,
                    object: Object::Image,
                    offset: 0x1000,
                },
                Constraint::Store {
                    ptr: from_8,
                    src: table,
                },
                Constraint::Points {
                    ptr: to,
                    object: Object::Allocation(InstrId::from_raw(2).expect("nonzero")),
                    offset: 64,
                },
                Constraint::CopyCells { dst: to, src: from },
                Constraint::Offset {
                    dst: to_8,
                    src: to,
                    by: Shift::Bytes(8),
                },
                Constraint::Load {
                    dst: read,
                    ptr: to_8,
                },
                Constraint::Load {
                    dst: other,
                    ptr: to,
                },
            ],
            &WorkBudget::DEFAULT,
        );
        assert_eq!(
            cells_of(&relation, read),
            vec![Some((Object::Image, Some(0x1000)))]
        );
        assert!(
            relation.get(other).is_empty(),
            "nothing was copied to byte 64"
        );

        let (on, word, onto) = (8, 9, 10);
        let onto_itself = solve(
            &[
                Constraint::Points {
                    ptr: from,
                    object: Object::Frame,
                    offset: -128,
                },
                Constraint::Points {
                    ptr: word,
                    object: Object::Image,
                    offset: 0x2000,
                },
                Constraint::Store {
                    ptr: from,
                    src: word,
                },
                Constraint::Offset {
                    dst: on,
                    src: from,
                    by: Shift::Bytes(16),
                },
                Constraint::CopyCells { dst: on, src: from },
                Constraint::Load { dst: onto, ptr: on },
            ],
            &WorkBudget::DEFAULT,
        );
        assert!(
            onto_itself.exhaustion().is_none(),
            "the copy onto itself ends"
        );
        assert!(
            cells_of(&onto_itself, onto).contains(&Some((Object::Image, Some(0x2000)))),
            "the byte copied 16 bytes on holds what byte 0 did"
        );
    }

    /// The host's models for one test: what the call at each position does.
    #[derive(Debug)]
    struct Models(Vec<((usize, usize), LibraryModel)>);

    impl CallModels for Models {
        fn model(&self, block: usize, instruction: usize) -> Option<LibraryModel> {
            self.0
                .iter()
                .find(|(at, _)| *at == (block, instruction))
                .map(|(_, model)| *model)
        }
    }

    /// **A modelled call keeps what it is handed.** A record in the frame
    /// holds a table's address at its first byte and is then filled, as dash
    /// clears a `synstack` record with `memset`: a fill writes no pointer and
    /// keeps none of its arguments, so the record's first byte still holds
    /// the table's address and nothing else. The control: a call no model
    /// describes takes the record's address, which escapes, and its bytes may
    /// hold anything.
    /// A frame record that holds a table's address at its first byte and is
    /// then filled by the call at `(0, 3)`, and the value read back from its
    /// first byte.
    fn filled_record() -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let stack = ir.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::Ptr);
        if let Some(variable) = ir.variable_mut(stack) {
            variable.set_role(VariableRole::EntryStackPointer);
        }
        let record = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
        let table = ptr_var(&mut ir, VariableOrigin::Local(1), 0, 1);
        let read = ptr_var(&mut ir, VariableOrigin::Local(2), 0, 4);
        let mut block = SsaBlock::with_capacity(0, 0, 6);
        block.add_instruction(instr(SsaOp::PtrAdd {
            dest: record,
            base: stack,
            index: None,
            stride: 1,
            offset: -16,
            result_type: MockType::Ptr,
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: table,
            value: ConstValue::Symbol(0x1000),
        }));
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr: record,
            value: table,
            value_type: MockType::Ptr,
            access: AccessQualifiers::FLAT,
        }));
        block.add_instruction(instr(SsaOp::from(CallData::new(
            None,
            Callee::Direct(7),
            vec![record],
        ))));
        block.add_instruction(instr(SsaOp::LoadIndirect {
            dest: read,
            addr: record,
            value_type: MockType::Ptr,
            access: AccessQualifiers::FLAT,
        }));
        block.add_instruction(instr(SsaOp::returns(None)));
        ir.add_block(block);
        ir.recompute_uses();

        (ir, read)
    }

    #[test]
    fn a_modelled_call_keeps_what_it_is_handed() {
        let (ir, read) = filled_record();
        let fill = Models(vec![((0, 3), LibraryModel::Fill)]);
        let modelled = analyze_function(&ir, &WorkBudget::DEFAULT, Some(&fill));
        assert_eq!(
            cells_of(&modelled, read.as_u32()),
            vec![Some((Object::Image, Some(0x1000)))]
        );
        let unmodelled = analyze_function(&ir, &WorkBudget::DEFAULT, None);
        assert!(
            unmodelled.get(read.as_u32()).contains(&UNKNOWN_LOC),
            "the control: a call no model describes may write the record"
        );
    }

    /// **A model says nothing of what a call returns beside its result.** A
    /// fill's model describes the pointer it returns and the bytes it writes;
    /// a value the call returns beside that may point anywhere, as one no
    /// constraint names would read as pointing nowhere.
    #[test]
    fn a_modelled_calls_values_beside_its_result_point_anywhere() {
        let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let beside = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
        let mut block = SsaBlock::with_capacity(0, 0, 2);
        block.add_instruction(instr(SsaOp::from(
            CallData::new(None, Callee::Direct(7), vec![])
                .with_result(CallInputKey::new(2), beside),
        )));
        block.add_instruction(instr(SsaOp::returns(None)));
        ir.add_block(block);
        ir.recompute_uses();
        let fill = Models(vec![((0, 0), LibraryModel::Fill)]);
        let modelled = analyze_function(&ir, &WorkBudget::DEFAULT, Some(&fill));
        assert!(modelled.get(beside.as_u32()).contains(&UNKNOWN_LOC));
    }

    /// **The models a context states reach the points-to relation its
    /// analyses cache.** The record filled through a modelled fill keeps the
    /// table's address where the context carries the model
    /// ([`AnalysisContext::with_models`]), and may hold anything where it
    /// carries none.
    #[test]
    fn a_contexts_models_reach_the_cached_relation() {
        let (ir, read) = filled_record();
        let fill = Models(vec![((0, 3), LibraryModel::Fill)]);
        let context = AnalysisContext::new(MACHINE).with_models(&fill);
        assert!(context.models().is_some());
        let modelled = FunctionAnalyses::new(&ir, &context);
        assert_eq!(
            cells_of(modelled.points_to(), read.as_u32()),
            vec![Some((Object::Image, Some(0x1000)))]
        );
        let plain = AnalysisContext::new(MACHINE);
        let unmodelled = FunctionAnalyses::new(&ir, &plain);
        assert!(
            unmodelled
                .points_to()
                .get(read.as_u32())
                .contains(&UNKNOWN_LOC)
        );
    }

    /// **An allocator delivering through a cell points no result at its
    /// block.** `posix_memalign(&p, 16, n)` returns a status and writes the
    /// block to `p`: its result points at nothing, and as a call no
    /// constraint states the store of, the cell it is handed may hold
    /// anything. The control: `malloc(n)`, which returns the block, points
    /// its result at the allocation.
    #[test]
    fn an_allocator_delivering_through_a_cell_points_no_result_at_its_block() {
        let build = || {
            let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
            let stack = ir.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::Ptr);
            if let Some(variable) = ir.variable_mut(stack) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            let cell = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
            let result = ptr_var(&mut ir, VariableOrigin::Local(1), 0, 1);
            let mut block = SsaBlock::with_capacity(0, 0, 3);
            block.add_instruction(instr(SsaOp::PtrAdd {
                dest: cell,
                base: stack,
                index: None,
                stride: 1,
                offset: -16,
                result_type: MockType::Ptr,
            }));
            block.add_instruction(instr(SsaOp::from(CallData::new(
                Some(result),
                Callee::Direct(7),
                vec![cell],
            ))));
            block.add_instruction(instr(SsaOp::returns(None)));
            ir.add_block(block);
            ir.recompute_uses();
            (ir, result)
        };
        let allocation = |delivery| {
            LibraryModel::Allocate(Allocation {
                family: AllocatorFamily(1),
                size: Operand::Argument(2),
                count: None,
                alignment: None,
                zeroed: false,
                flags: None,
                delivery,
                requires: Conditions::NONE,
                pages: None,
                protection: None,
                heap: None,
            })
        };
        let (ir, result) = build();
        let out = Models(vec![(
            (0, 1),
            allocation(Delivery::Out {
                out: 0,
                failure: 12,
            }),
        )]);
        let delivered = analyze_function(&ir, &WorkBudget::DEFAULT, Some(&out));
        assert!(
            cells_of(&delivered, result.as_u32())
                .iter()
                .all(|cell| !matches!(cell, Some((Object::Allocation(_), _)))),
            "a status points at no block"
        );
        let returned = Models(vec![((0, 1), allocation(Delivery::Returned))]);
        let made = analyze_function(&ir, &WorkBudget::DEFAULT, Some(&returned));
        assert!(
            cells_of(&made, result.as_u32())
                .iter()
                .any(|cell| matches!(cell, Some((Object::Allocation(_), _)))),
            "the control: a returned block is what the result points at"
        );
    }

    /// **An address stored to memory escapes, and so does a slot's.** Local
    /// 3's address is written to a static field and a call follows: the call
    /// can read the field and write through it, so it clobbers local 3's cell.
    /// The control drops the store: nothing hands the call the address, and
    /// the cell survives it.
    #[test]
    fn an_address_stored_to_memory_escapes_to_every_call() {
        let build = |store: bool| {
            let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
            let address = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
            let mut block = SsaBlock::new(0);
            block.add_instruction(instr(SsaOp::LoadLocalAddr {
                dest: address,
                local_index: 3,
            }));
            if store {
                block.add_instruction(instr(SsaOp::StoreStaticField {
                    access: AccessQualifiers::FLAT,
                    field: 1,
                    value: address,
                }));
            }
            block.add_instruction(instr(SsaOp::from(CallData::new(
                None,
                Callee::Direct(7),
                Vec::new(),
            ))));
            ir.add_block(block);
            ir.recompute_uses();
            ir
        };
        let slot = AliasKey {
            base: AddressBase::Local(3),
            index: None,
            stride_bytes: 0,
            offset_bits: 0,
        };
        let stored = build(true);
        let points_to = analyze_function(&stored, &WorkBudget::DEFAULT, None);
        assert!(
            CallEscape::of(points_to, &stored).clobbers(&slot),
            "the call can reach local 3 through the field"
        );
        let kept = build(false);
        let points_to = analyze_function(&kept, &WorkBudget::DEFAULT, None);
        assert!(
            !CallEscape::of(points_to, &kept).clobbers(&slot),
            "control: nothing hands the call local 3's address"
        );
    }

    /// An implicit pointer reaches the same cells as an ordinary argument;
    /// the unpassed control remains private despite sharing the call's block.
    #[test]
    fn implicit_call_inputs_escape_and_unpassed_controls_remain_private() {
        let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let hidden = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
        let private = ptr_var(&mut ir, VariableOrigin::Local(1), 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::LoadLocalAddr {
            dest: hidden,
            local_index: 3,
        }));
        block.add_instruction(instr(SsaOp::LoadLocalAddr {
            dest: private,
            local_index: 4,
        }));
        block.add_instruction(instr(
            CallData::new(None, Callee::Direct(7), vec![])
                .with_implicit_input(CallInputKey::new(17), hidden)
                .into(),
        ));
        block.add_instruction(instr(SsaOp::returns(None)));
        ir.add_block(block);
        ir.recompute_uses();
        let points_to = analyze_function(&ir, &WorkBudget::DEFAULT, None);
        let escape = CallEscape::of(points_to, &ir);
        assert!(escape.clobbers(&cell_of(hidden.as_u32())));
        assert!(!escape.clobbers(&cell_of(private.as_u32())));
    }

    /// A cell as [`PointsTo::cell`] names it.
    type Cell = Option<(Object, Option<i64>)>;

    /// What the cells a solve reads back hold: whether `read` may be
    /// anything, and the cells it points at.
    fn holds(relation: &PointsTo, read: Loc) -> (bool, Vec<Cell>) {
        (
            relation.get(read).contains(&UNKNOWN_LOC),
            cells_of(relation, read)
                .into_iter()
                .filter(Option::is_some)
                .collect(),
        )
    }

    /// **A call reaches the bytes it states, and no others.** Frame bytes
    /// `-16` and `-8` each hold the image's `0x1000`; a call writes 4 bytes
    /// from byte `-16` and keeps nothing: byte `-16` may then hold anything,
    /// byte `-8` holds what it did, and no byte of the frame escaped. A cell
    /// named only after the call is reached all the same (byte `-14`, through
    /// a move the solve makes later). The control: a call reaching to the
    /// object's end reaches byte `-8` too.
    #[test]
    fn a_call_reaches_the_bytes_it_states_and_no_others() {
        let solved = |bytes: Option<NonZeroU64>| {
            let (frame, at8, value, inner) = (1, 2, 3, 4);
            let (read0, read8, read2) = (10, 11, 12);
            let relation = solve(
                &[
                    Constraint::Points {
                        ptr: frame,
                        object: Object::Frame,
                        offset: -16,
                    },
                    Constraint::Points {
                        ptr: at8,
                        object: Object::Frame,
                        offset: -8,
                    },
                    Constraint::Points {
                        ptr: value,
                        object: Object::Image,
                        offset: 0x1000,
                    },
                    Constraint::Store {
                        ptr: frame,
                        src: value,
                    },
                    Constraint::Store {
                        ptr: at8,
                        src: value,
                    },
                    Constraint::Touch {
                        ptr: frame,
                        reached: Reached::From { at: 0, bytes },
                        reach: Reach {
                            writes: true,
                            reads: false,
                            keeps: false,
                        },
                    },
                    Constraint::Offset {
                        dst: inner,
                        src: frame,
                        by: Shift::Bytes(2),
                    },
                    Constraint::Load {
                        dst: read0,
                        ptr: frame,
                    },
                    Constraint::Load {
                        dst: read8,
                        ptr: at8,
                    },
                    Constraint::Load {
                        dst: read2,
                        ptr: inner,
                    },
                ],
                &WorkBudget::DEFAULT,
            );
            (
                relation.get(read0).contains(&UNKNOWN_LOC),
                relation.get(read8).contains(&UNKNOWN_LOC),
                relation.get(read2).contains(&UNKNOWN_LOC),
                relation.has_escaped(Object::Frame, Some(-16)),
            )
        };
        assert_eq!(solved(NonZeroU64::new(4)), (true, false, true, false));
        assert_eq!(
            solved(None),
            (true, true, true, false),
            "the control: to the object's end"
        );
    }

    /// **What a call reads escapes to it; what it keeps escapes from there
    /// up.** Frame byte `-32` holds the address of frame byte `-16`. A call
    /// that reads byte `-32` learns that address, so byte `-16` escapes; one
    /// that only writes it learns nothing. One that keeps what it reaches
    /// lets byte `-32` and everything above it escape, and one that keeps
    /// the whole frame lets every byte go, byte `-40` below it included.
    #[test]
    fn what_a_call_reads_escapes_and_what_it_keeps_escapes_from_there_up() {
        let solved = |at: Option<i64>, reach: Reach| {
            let (frame, local, read) = (1, 2, 3);
            let relation = solve(
                &[
                    Constraint::Points {
                        ptr: frame,
                        object: Object::Frame,
                        offset: -32,
                    },
                    Constraint::Points {
                        ptr: local,
                        object: Object::Frame,
                        offset: -16,
                    },
                    Constraint::Store {
                        ptr: frame,
                        src: local,
                    },
                    Constraint::Touch {
                        ptr: frame,
                        reached: at.map_or(Reached::Object, |at| Reached::From {
                            at,
                            bytes: NonZeroU64::new(4),
                        }),
                        reach,
                    },
                    Constraint::Load {
                        dst: read,
                        ptr: frame,
                    },
                ],
                &WorkBudget::DEFAULT,
            );
            [-40, -32, -16].map(|byte| relation.has_escaped(Object::Frame, Some(byte)))
        };
        let reach = |writes, reads, keeps| Reach {
            writes,
            reads,
            keeps,
        };
        assert_eq!(
            solved(Some(0), reach(false, true, false)),
            [false, false, true]
        );
        assert_eq!(
            solved(Some(0), reach(true, false, false)),
            [false, false, false],
            "the control: a write learns nothing"
        );
        assert_eq!(
            solved(Some(0), reach(true, true, true)),
            [false, true, true]
        );
        assert_eq!(solved(None, reach(true, true, true)), [true, true, true]);
    }

    /// One function for the call-effect tests: a record at frame byte `-16`
    /// and a slot at `-8` each hold the image's `0x1000`, a call with
    /// `effects` is handed the record, and each is read back after it.
    ///
    /// Returns the function and the two values read back.
    fn handed_record(
        effects: CallEffects<MockTarget>,
    ) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        handed_counted_record(effects, None)
    }

    /// [`handed_record`], the call passed `count` after the record, where
    /// one is given.
    fn handed_counted_record(
        effects: CallEffects<MockTarget>,
        count: Option<i32>,
    ) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut ir: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let stack = ir.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::Ptr);
        if let Some(variable) = ir.variable_mut(stack) {
            variable.set_role(VariableRole::EntryStackPointer);
        }
        let record = ptr_var(&mut ir, VariableOrigin::Local(0), 0, 0);
        let slot = ptr_var(&mut ir, VariableOrigin::Local(1), 0, 1);
        let table = ptr_var(&mut ir, VariableOrigin::Local(2), 0, 2);
        let from_record = ptr_var(&mut ir, VariableOrigin::Local(3), 0, 6);
        let from_slot = ptr_var(&mut ir, VariableOrigin::Local(4), 0, 7);
        let mut block = SsaBlock::with_capacity(0, 0, 9);
        for (dest, offset) in [(record, -16), (slot, -8)] {
            block.add_instruction(instr(SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset,
                result_type: MockType::Ptr,
            }));
        }
        block.add_instruction(instr(SsaOp::Const {
            dest: table,
            value: ConstValue::Symbol(0x1000),
        }));
        for addr in [record, slot] {
            block.add_instruction(instr(SsaOp::StoreIndirect {
                addr,
                value: table,
                value_type: MockType::I64,
                access: AccessQualifiers::FLAT,
            }));
        }
        let mut args = vec![record];
        if let Some(bytes) = count {
            let counted = ptr_var(&mut ir, VariableOrigin::Local(5), 0, 8);
            block.add_instruction(instr(SsaOp::Const {
                dest: counted,
                value: ConstValue::I32(bytes),
            }));
            args.push(counted);
        }
        block.add_instruction(instr(SsaOp::from(
            CallData::new(None, Callee::Direct(7), args).with_effects(effects),
        )));
        for (dest, addr) in [(from_record, record), (from_slot, slot)] {
            block.add_instruction(instr(SsaOp::LoadIndirect {
                dest,
                addr,
                value_type: MockType::I64,
                access: AccessQualifiers::FLAT,
            }));
        }
        block.add_instruction(instr(SsaOp::returns(None)));
        ir.add_block(block);
        ir.recompute_uses();
        (ir, from_record, from_slot)
    }

    /// The effects of a call that reads and writes escaped memory and
    /// `accesses`, keeping what `captures` says, returning twice where
    /// `returns_twice` does.
    fn stated(
        accesses: Vec<CallAccess<MockTarget>>,
        captures: Capture,
        returns_twice: bool,
    ) -> CallEffects<MockTarget> {
        let mut memory = vec![CallAccess {
            location: CallLocation::Escaped,
            access: AccessMode::Update,
        }];
        memory.extend(accesses);
        CallEffects {
            memory: CallMemory::Only(memory),
            world: true,
            calls_back: true,
            captures,
            termination: CallTermination::Returns,
            returns_twice,
            may_throw: false,
        }
    }

    /// **A call reaches an argument it does not keep only as it says.** A
    /// shell routine hands a 12-byte record in its frame to a call that
    /// updates the record and keeps nothing: the record may hold anything
    /// after, the slot above it holds what it did, and no byte of the frame
    /// escaped. The control: a call that may keep the record lets it and
    /// every byte above it go, the slot included.
    #[test]
    fn a_call_reaches_an_argument_it_does_not_keep_only_as_it_says() {
        let record = CallAccess {
            location: CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::Bytes(4),
            },
            access: AccessMode::Update,
        };
        let solved = |captures: Capture| {
            let (ir, from_record, from_slot) =
                handed_record(stated(vec![record.clone()], captures, false));
            let relation = analyze_function(&ir, &WorkBudget::DEFAULT, None);
            (
                holds(&relation, from_record.as_u32()).0,
                holds(&relation, from_slot.as_u32()),
                relation.has_escaped(Object::Frame, Some(-16)),
            )
        };
        assert_eq!(
            solved(Capture::AnyBut(vec![0])),
            (
                true,
                (false, vec![Some((Object::Image, Some(0x1000)))]),
                false
            )
        );
        let (_, (slot_anything, _), escaped) = solved(Capture::Any);
        assert!(
            slot_anything && escaped,
            "the control: the call may keep it"
        );
    }

    /// **A call reaches as many bytes as the constant its count argument
    /// holds.** `memcpy(dst, src, n)` writes `n` bytes from `dst`: a call
    /// updating four bytes of the record, as its second argument counts them,
    /// leaves the slot above it holding what it did. The controls: sixteen
    /// bytes reach the slot, and so does a count read as no constant, which
    /// bounds nothing.
    #[test]
    fn a_call_reaches_as_many_bytes_as_its_constant_count() {
        let counted = CallAccess {
            location: CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::InArgument(1),
            },
            access: AccessMode::Update,
        };
        let slot_anything = |count: Option<i32>| {
            let (ir, _, from_slot) = handed_counted_record(
                stated(vec![counted.clone()], Capture::AnyBut(vec![0, 1]), false),
                count,
            );
            let relation = analyze_function(&ir, &WorkBudget::DEFAULT, None);
            holds(&relation, from_slot.as_u32()).0
        };
        assert!(!slot_anything(Some(4)), "four bytes reach the record alone");
        assert!(slot_anything(Some(16)), "sixteen reach the slot");
    }

    /// **A call writes the bytes of the frame it names.** A call that keeps
    /// nothing it is handed writes 4 bytes at frame byte `-8`, where its
    /// arguments lie: the slot there may hold anything after it, the record
    /// below holds what it did. The control: bytes at `-4` miss the slot.
    #[test]
    fn a_call_writes_the_bytes_of_the_frame_it_names() {
        let solved = |offset: i64| {
            let (ir, from_record, from_slot) = handed_record(stated(
                vec![CallAccess {
                    location: CallLocation::Frame {
                        origin: FrameOrigin::Entry,
                        offset,
                        bytes: 4,
                    },
                    access: AccessMode::Write,
                }],
                Capture::Nothing,
                false,
            ));
            let relation = analyze_function(&ir, &WorkBudget::DEFAULT, None);
            (
                holds(&relation, from_record.as_u32()).0,
                holds(&relation, from_slot.as_u32()).0,
                relation.has_escaped(Object::Frame, None),
            )
        };
        assert_eq!(solved(-8), (false, true, false));
        assert_eq!(
            solved(-4),
            (false, false, false),
            "the control: other bytes"
        );
    }

    /// **A call that returns twice lets every byte of the frame go.** It
    /// resumes code no edge leads to, which reads the frame as a later call
    /// leaves it: the slot, which nothing hands away, may hold anything. The
    /// control: the same call returning once.
    #[test]
    fn a_call_that_returns_twice_lets_every_byte_of_the_frame_go() {
        let solved = |returns_twice: bool| {
            let (ir, _, from_slot) =
                handed_record(stated(Vec::new(), Capture::Nothing, returns_twice));
            let relation = analyze_function(&ir, &WorkBudget::DEFAULT, None);
            (
                holds(&relation, from_slot.as_u32()).0,
                relation.has_escaped(Object::Frame, Some(-16)),
            )
        };
        assert_eq!(solved(true), (true, true));
        assert_eq!(
            solved(false),
            (false, false),
            "the control: it returns once"
        );
    }

    /// **Footprints overlap where their objects' bytes do.** Two words of the
    /// frame apart, two allocations, the frame against memory let go of while
    /// no byte of it is: apart. The same word, any byte of one allocation,
    /// and memory let go of against a frame byte that escaped: together; and
    /// a footprint of any memory overlaps everything.
    #[test]
    fn footprints_overlap_where_their_objects_bytes_do() {
        let site = |n: u32| InstrId::from_raw(n).map(Object::Allocation);
        let (Some(first), Some(second)) = (site(1), site(2)) else {
            panic!("allocation sites");
        };
        let relation = |escaped_from: Option<i64>| {
            let (frame, local) = (1, 2);
            let mut constraints = vec![
                Constraint::Points {
                    ptr: frame,
                    object: Object::Frame,
                    offset: -16,
                },
                Constraint::Points {
                    ptr: local,
                    object: Object::Frame,
                    offset: escaped_from.unwrap_or(0),
                },
            ];
            if escaped_from.is_some() {
                constraints.push(Constraint::Escape { ptr: local });
            }
            solve(&constraints, &WorkBudget::DEFAULT)
        };
        let kept = relation(None);
        let placed = |spans: Vec<(Object, Option<(i64, i64)>)>| kept.placed(spans, false);
        let word = |at: i64| placed(vec![(Object::Frame, Some((at, at + 4)))]);
        assert!(!word(-16).overlaps(&word(-8)));
        assert!(word(-16).overlaps(&word(-14)));
        assert!(
            !placed(vec![(first, Some((0, 4)))]).overlaps(&placed(vec![(second, Some((0, 4)))]))
        );
        assert!(placed(vec![(first, None)]).overlaps(&placed(vec![(first, Some((8, 12)))])));
        assert!(!Footprint::ESCAPED.overlaps(&word(-16)));
        assert!(!Footprint::ESCAPED.overlaps(&placed(vec![(first, None)])));
        assert!(Footprint::ESCAPED.overlaps(&Footprint::image(Some((0x1000, 0x1004)))));
        assert!(Footprint::ESCAPED.overlaps(&Footprint::ESCAPED));
        assert!(Footprint::Anywhere.overlaps(&word(-16)));
        let let_go = relation(Some(-20));
        let word = |at: i64| let_go.placed(vec![(Object::Frame, Some((at, at + 4)))], false);
        assert!(
            Footprint::ESCAPED.overlaps(&word(-16)),
            "the control: a byte from the escaped local up"
        );
        assert!(!Footprint::ESCAPED.overlaps(&word(-24)), "one below it");
    }

    /// **A footprint past the target's addresses is any memory.** The
    /// relation does not wrap an offset at the pointer's width, so a byte it
    /// names past that range may be one it names by another number: on a
    /// 32-bit target, frame byte `0xffff_fff8` is byte `-8`. The control: the
    /// same byte on a 64-bit target is placed, among the caller's bytes.
    #[test]
    fn a_footprint_past_the_targets_addresses_is_any_memory() {
        let (frame, moved) = (1, 2);
        let relation = solve(
            &[
                Constraint::Points {
                    ptr: frame,
                    object: Object::Frame,
                    offset: 0,
                },
                Constraint::Offset {
                    dst: moved,
                    src: frame,
                    by: Shift::Bytes(0xffff_fff8),
                },
            ],
            &WorkBudget::DEFAULT,
        );
        let pointer = SsaVarId::from_index(moved as usize);
        assert_eq!(
            relation.footprint(pointer, 0, Some(4), PointerSize::Bit32),
            Footprint::Anywhere
        );
        assert_eq!(
            relation.footprint(pointer, 0, Some(4), PointerSize::Bit64),
            Footprint::Placed {
                spans: vec![(Object::Caller, Some((0xffff_fff8, 0xffff_fffc)))],
                escaped: false,
                exposed: true,
            },
            "the control: within a 64-bit target's, a byte of the caller's"
        );
    }

    /// **The caller's bytes are anyone's.** A pointer to frame byte `-8`
    /// names, 16 bytes on, byte `8`: the caller's, which any call may write
    /// though nothing hands one the frame. The control: 4 bytes on, byte
    /// `-4` is the function's own and no call reaches it.
    #[test]
    fn the_callers_bytes_are_anyones() {
        let frame = 1;
        let points_to = solve(
            &[Constraint::Points {
                ptr: frame,
                object: Object::Frame,
                offset: -8,
            }],
            &WorkBudget::DEFAULT,
        );
        let escape = CallEscape::new(Arc::new(points_to), std::iter::empty());
        let at = |bytes: i64| AliasKey {
            offset_bits: bytes * 8,
            ..cell_of(frame)
        };
        assert!(escape.clobbers(&at(16)), "byte 8 is the caller's");
        assert!(!escape.clobbers(&at(4)), "the control: byte -4 is its own");
    }

    /// **A pointer into the frame reaches the caller's bytes above it.** A
    /// frame byte's address handed on reaches, besides the frame's bytes
    /// from it up, the caller's bytes above the entry stack pointer and what
    /// they hold. The control: nothing below it.
    #[test]
    fn a_pointer_into_the_frame_reaches_the_callers_bytes_above_it() {
        let (local, below, caller) = (1, 2, 3);
        let points_to = solve(
            &[
                Constraint::Points {
                    ptr: local,
                    object: Object::Frame,
                    offset: -16,
                },
                Constraint::Points {
                    ptr: below,
                    object: Object::Frame,
                    offset: -32,
                },
                Constraint::Points {
                    ptr: caller,
                    object: Object::Frame,
                    offset: 8,
                },
            ],
            &WorkBudget::DEFAULT,
        );
        let reached: BTreeSet<Option<(Object, Option<i64>)>> = points_to
            .reachable_closure([local])
            .into_iter()
            .map(|loc| points_to.cell(loc))
            .collect();
        assert!(reached.contains(&Some((Object::Caller, Some(8)))));
        assert!(
            !reached.contains(&Some((Object::Frame, Some(-32)))),
            "the control: a byte below"
        );
    }

    /// **An access running into the caller's bytes lets the frame go from
    /// it.** An 8-byte read at frame byte `-4` reads 4 of the caller's
    /// bytes, which the caller may have let go of: the frame escapes from
    /// byte `-4` up. The control: a 4-byte read there stays the function's.
    #[test]
    fn an_access_running_into_the_callers_bytes_lets_the_frame_go_from_it() {
        let escaped = |bytes: u32| {
            let frame = 1;
            solve(
                &[
                    Constraint::Points {
                        ptr: frame,
                        object: Object::Frame,
                        offset: -4,
                    },
                    Constraint::Width {
                        ptr: frame,
                        bytes: Some(bytes),
                    },
                ],
                &WorkBudget::DEFAULT,
            )
            .has_escaped(Object::Frame, Some(-4))
        };
        assert!(escaped(8));
        assert!(!escaped(4), "the control: it ends at the entry");
    }

    /// **The entry stack pointer aligned points into the frame.** `sp & -16`
    /// moves the entry stack pointer by an amount nothing bounds, into the
    /// function's own frame, which is then one cell and escaped no more than
    /// it was. The control: a caller's byte above the entry moved so (a walk
    /// of the stack arguments) may be anywhere, the caller's bytes having
    /// escaped.
    #[test]
    fn the_entry_stack_pointer_aligned_points_into_the_frame() {
        let moved = |from: i64| {
            let (entry, aligned) = (1, 2);
            let relation = solve(
                &[
                    Constraint::Points {
                        ptr: entry,
                        object: Object::Frame,
                        offset: from,
                    },
                    Constraint::Offset {
                        dst: aligned,
                        src: entry,
                        by: Shift::Unknown,
                    },
                ],
                &WorkBudget::DEFAULT,
            );
            (
                cells_of(&relation, aligned),
                relation.has_escaped(Object::Frame, None),
            )
        };
        assert_eq!(moved(0), (vec![Some((Object::Frame, None))], false));
        assert_eq!(
            moved(4),
            (vec![None], false),
            "the control: a caller's byte"
        );
    }

    /// **An indexed access lies in the object of its first byte.** An index
    /// into a local array 40 bytes below the entry stack pointer, of no
    /// stated extent, may touch any byte of the frame, the array's own
    /// elements among them, though the stack pointer points at the caller's
    /// first byte. The control: an index from 8 bytes above it, a walk of the
    /// stack arguments, lies in the caller's bytes.
    #[test]
    fn an_indexed_access_lies_in_the_object_of_its_first_byte() {
        let entry = 1;
        let relation = solve(
            &[Constraint::Points {
                ptr: entry,
                object: Object::Frame,
                offset: 0,
            }],
            &WorkBudget::DEFAULT,
        );
        let pointer = SsaVarId::from_index(entry as usize);
        let indexed = |first: i64| relation.footprint(pointer, first, None, PointerSize::Bit64);
        assert_eq!(
            indexed(-40),
            Footprint::Placed {
                spans: vec![(Object::Frame, None)],
                escaped: false,
                exposed: false,
            }
        );
        let element = relation.footprint(pointer, -4, Some(0), PointerSize::Bit64);
        assert!(indexed(-40).overlaps(&element), "its own element");
        assert_eq!(
            indexed(8),
            Footprint::Placed {
                spans: vec![(Object::Caller, None)],
                escaped: false,
                exposed: true,
            },
            "the control: the caller's bytes"
        );
    }

    /// **The same constraints are solved once.** A function asked for its
    /// relation twice as it stands gets the one solve both times, its copy
    /// too; the controls solve again: the function with a store added, whose
    /// constraints differ, and the same constraints within another budget.
    #[test]
    fn the_same_constraints_are_solved_once() {
        let (mut ir, _, _) = handed_record(stated(Vec::new(), Capture::Nothing, false));
        let first = analyze_function(&ir, &WorkBudget::DEFAULT, None);
        assert!(Arc::ptr_eq(
            &first,
            &analyze_function(&ir, &WorkBudget::DEFAULT, None)
        ));
        assert!(Arc::ptr_eq(
            &first,
            &analyze_function(&ir.clone(), &WorkBudget::DEFAULT, None)
        ));
        let tight = WorkBudget::DEFAULT.with_limit(Budget::PointsToStepsPerConstraint, 1);
        assert!(
            !Arc::ptr_eq(&first, &analyze_function(&ir, &tight, None)),
            "the control: another budget"
        );
        let value = ptr_var(&mut ir, VariableOrigin::Local(9), 0, 8);
        if let Some(block) = ir.block_mut(0) {
            block.add_instruction(instr(SsaOp::Const {
                dest: value,
                value: ConstValue::Symbol(0x2000),
            }));
        }
        ir.recompute_uses();
        assert!(
            !Arc::ptr_eq(&first, &analyze_function(&ir, &WorkBudget::DEFAULT, None)),
            "the control: other constraints"
        );
    }
}
