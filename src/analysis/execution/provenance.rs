//! Which object each pointer was formed into, and the accesses that leave it.
//!
//! A pointer is formed into an object where the program makes it a pointer
//! into one: a base indexed by a value the code does not write as a literal,
//! which forms the result into the object the base, displaced by the
//! constant, lands in; and a pointer handed to a call as an argument, which
//! forms the callee's copy into the object it lands in. Only an object whose
//! extent is exact is one a pointer is formed into. A literal forms
//! nothing: machine code reaches an object as an anchor plus a constant (a
//! section anchor, a literal pool word, a base folded with an offset), and
//! the anchor is no object the pointer is into. Nor does a pointer moved by
//! a constant alone, as a stack or frame pointer moves: those land in
//! whatever slot they pass, and are pointers into the frame, not into a slot.
//!
//! Arithmetic carries the object to its result; a store keeps it with the
//! stored bytes, and a load of them hands it back. An access through a
//! pointer formed into an object is judged against that object: one outside
//! it is past its extent, whatever object the access itself lands in. Every
//! access is: a scalar, vector or atomic one through its address, a block
//! operation's through each of its pointers, and a model's through the
//! argument it reaches memory by ([`Pointer`]). The object travels beside
//! the value, as never-written marks do: it is no part of what the value is,
//! and two values that hold the same number are equal whatever objects they
//! were formed into.

use crate::{
    analysis::execution::{
        access::Access,
        memory::Through,
        state::State,
        value::{Address, Value},
    },
    ir::{function::SsaFunction, ops::SsaOp, variable::SsaVarId},
    pointer::PointerSize,
    target::Target,
};

/// A pointer a model reaches memory through: the address it holds, and the
/// start of the object it was formed into, where it was formed into one.
///
/// A model's every access through it is judged against that object, as an
/// instruction's access through a pointer is, so a `memset` handed a pointer
/// formed into one object and moved past it writes past that object's
/// extent, whatever object the write lands in. A pointer the model makes
/// itself, an allocation's block, is formed into nothing: it is an
/// [`Address`], and its accesses are judged where they land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pointer {
    /// The address it holds.
    address: Address,
    /// The start of the object it was formed into.
    object: Option<Address>,
}

impl Pointer {
    /// A pointer holding `address`, formed into the object starting at
    /// `object`, where one is.
    #[must_use]
    pub const fn new(address: Address, object: Option<Address>) -> Self {
        Self { address, object }
    }

    /// The address it holds.
    #[must_use]
    pub const fn address(self) -> Address {
        self.address
    }

    /// The start of the object it was formed into.
    #[must_use]
    pub const fn object(self) -> Option<Address> {
        self.object
    }

    /// This pointer moved by `delta` bytes at the pointer width `size`, into
    /// the object it was formed into still, as a walk along an object
    /// keeps it. `None` where the address leaves its address space.
    #[must_use]
    pub fn moved(self, delta: i64, size: PointerSize) -> Option<Self> {
        Some(Self {
            address: self.address.moved(delta, size)?,
            object: self.object,
        })
    }
}

/// An address the model made itself: formed into nothing.
impl From<Address> for Pointer {
    fn from(address: Address) -> Self {
        Self {
            address,
            object: None,
        }
    }
}

/// Where a pointer was formed: into an object, or from an anchor whose
/// object its use decides.
///
/// An indexed pointer does not know its object where the index is added:
/// machine code adds the constant that reaches an object as often after the
/// index as before it (`anchor + i + 16`). It carries its anchor, the base
/// with every constant displacement and without the index, and the object is
/// the one the anchor starts where the pointer is used; an anchor off any
/// object's start names none. A pointer formed
/// into an object keeps it through constant steps, as a pointer walked one
/// element at a time does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Origin {
    /// Formed into the object starting here.
    Object(Address),
    /// Indexed from this anchor, whose landing names the object.
    Anchor(Address),
}

impl Origin {
    /// The origin of this pointer moved by the constant `delta`: an object
    /// stays, an anchor moves with it. `None` where the anchor leaves the
    /// address space.
    #[must_use]
    pub fn displaced(self, delta: i64) -> Option<Self> {
        match self {
            Self::Object(_) => Some(self),
            Self::Anchor(at) => at.moved(delta, at.pointer_size()).map(Self::Anchor),
        }
    }
}

/// Carries the objects pointers were formed into through one function's
/// instructions.
pub struct Provenance<'a, T: Target> {
    /// The function run.
    ssa: &'a SsaFunction<T>,
    /// How the run reads a value as an address.
    access: Access<'a>,
}

/// References only, whatever the target.
impl<T: Target> Clone for Provenance<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Target> Copy for Provenance<'_, T> {}

impl<'a, T: Target> Provenance<'a, T> {
    /// Carries objects through `ssa`'s instructions, reading addresses as
    /// `access` does.
    pub const fn new(ssa: &'a SsaFunction<T>, access: Access<'a>) -> Self {
        Self { ssa, access }
    }

    /// States, before `op` runs, the objects its accesses go through: for
    /// each way it moves bytes, the object of the one pointer it reads or
    /// writes them through, a gather's or a scatter's base for every lane.
    /// Where two of its pointers serve one way, a block compare's two
    /// blocks, that way names none here: the compare states each block's as
    /// it reads it.
    pub fn before(&self, op: &SsaOp<T>, state: &mut State<T>) {
        // With no object stated, no pointer is formed into one.
        if !state.memory().states_objects() {
            return;
        }
        let (mut reads, mut writes) = (Served::None, Served::None);
        op.for_each_addressed_access_by(
            |_| None,
            |access| {
                if access.qualifiers.address_space.is_some() {
                    return;
                }
                if access.reads {
                    reads.add(access.addr);
                }
                if access.writes {
                    writes.add(access.addr);
                }
            },
        );
        // A gather or a scatter reaches each lane at its base plus the
        // lane's index: through the base, whose object judges every lane.
        match op {
            SsaOp::VectorGather { base, access, .. } if access.address_space.is_none() => {
                reads.add(*base);
            }
            SsaOp::VectorScatter { base, access, .. } if access.address_space.is_none() => {
                writes.add(*base);
            }
            _ => {}
        }
        let through = Through {
            reads: reads
                .one()
                .and_then(|pointer| self.pointer_object(state, pointer)),
            writes: writes
                .one()
                .and_then(|pointer| self.pointer_object(state, pointer)),
        };
        state.memory_mut().set_through(through);
    }

    /// States the origin of what `op` defined, once it has run, and keeps
    /// the object of a pointer it stored.
    pub fn after(&self, op: &SsaOp<T>, state: &mut State<T>) {
        // With no object stated, no pointer is formed into one.
        if !state.memory().states_objects() {
            return;
        }
        match op {
            SsaOp::LoadIndirect {
                dest,
                addr,
                value_type,
                access,
            }
            | SsaOp::AtomicLoad {
                dest,
                addr,
                value_type,
                access,
                ..
            } if access.address_space.is_none() => {
                let width = self.bytes(value_type);
                let object = self
                    .address(state, *addr)
                    .zip(width)
                    .and_then(|(at, width)| state.memory().formed(at, width));
                state.set_formed(*dest, object.map(Origin::Object));
            }
            SsaOp::StoreIndirect {
                addr,
                value,
                value_type,
                access,
            }
            | SsaOp::AtomicStore {
                addr,
                value,
                value_type,
                access,
                ..
            } if access.address_space.is_none() => {
                let width = self.bytes(value_type);
                let at = self.address(state, *addr);
                let object = self.pointer_object(state, *value);
                if let (Some(object), Some(at), Some(width)) = (object, at, width) {
                    state.memory_mut().keep_formed(at, width, object);
                }
            }
            SsaOp::Copy { dest, src }
            | SsaOp::PtrToInt {
                dest, operand: src, ..
            }
            | SsaOp::IntToPtr {
                dest, operand: src, ..
            }
            | SsaOp::Bitcast {
                dest, operand: src, ..
            } => {
                let origin = state.formed(*src);
                state.set_formed(*dest, origin);
            }
            SsaOp::PtrAdd {
                dest,
                base,
                index,
                offset,
                ..
            } => {
                let indexed = index.is_some_and(|index| !self.literal(index));
                let origin = match state.formed(*base) {
                    Some(origin) => origin.displaced(*offset),
                    None if indexed => self
                        .address(state, *base)
                        .and_then(|at| at.moved(*offset, at.pointer_size()))
                        .map(Origin::Anchor),
                    None => None,
                };
                state.set_formed(*dest, origin);
            }
            SsaOp::Add {
                dest, left, right, ..
            } => {
                let constant = |operand| self.constant(state, operand);
                let origin = match (state.formed(*left), state.formed(*right)) {
                    (Some(origin), None) => match constant(*right) {
                        Some(delta) => origin.displaced(delta),
                        None => Some(origin),
                    },
                    (None, Some(origin)) => match constant(*left) {
                        Some(delta) => origin.displaced(delta),
                        None => Some(origin),
                    },
                    (Some(_), Some(_)) => None,
                    (None, None) if !self.literal(*left) && !self.literal(*right) => {
                        // An index added to a base: the base is the operand
                        // that lands in an object, where exactly one does.
                        let landed = |operand| {
                            self.address(state, operand)
                                .filter(|at| self.landing(state, *at).is_some())
                        };
                        match (landed(*left), landed(*right)) {
                            (Some(at), None) | (None, Some(at)) => Some(Origin::Anchor(at)),
                            _ => None,
                        }
                    }
                    (None, None) => None,
                };
                state.set_formed(*dest, origin);
            }
            SsaOp::Sub {
                dest, left, right, ..
            } => {
                // A pointer less an integer stays a pointer of its origin;
                // the distance between two pointers is no pointer.
                let origin = match (state.formed(*left), state.formed(*right)) {
                    (Some(origin), None) => match self.constant(state, *right) {
                        Some(delta) => origin.displaced(delta.wrapping_neg()),
                        None => Some(origin),
                    },
                    _ => None,
                };
                state.set_formed(*dest, origin);
            }
            _ => {}
        }
    }

    /// The start of the object `pointer`'s origin names, where it has one.
    pub fn pointer_object(&self, state: &State<T>, pointer: SsaVarId) -> Option<Address> {
        state
            .formed(pointer)
            .and_then(|origin| self.object(state, origin))
    }

    /// The start of the object an argument handed to a call is a pointer
    /// into: the object its origin names, or the one it lands in.
    pub fn argument(&self, state: &State<T>, argument: SsaVarId) -> Option<Address> {
        match state.formed(argument) {
            Some(origin) => self.object(state, origin),
            None => {
                let at = self.address(state, argument)?;
                self.landing(state, at)
            }
        }
    }

    /// The start of the object `origin` names: the object itself, or the one
    /// an anchor starts. An anchor off any object's start (`base - 1` indexed
    /// from one, a constant folded into the middle of an object) names none:
    /// the index's own start carries the rest of the way, which no anchor
    /// shows, and the pointer's accesses are judged where they land.
    fn object(&self, state: &State<T>, origin: Origin) -> Option<Address> {
        match origin {
            Origin::Object(object) => Some(object),
            Origin::Anchor(at) => self
                .landing(state, at)
                .filter(|object| object.offset() == at.offset()),
        }
    }

    /// The integer `value` holds, where it holds a constant one.
    fn constant(&self, state: &State<T>, value: SsaVarId) -> Option<i64> {
        state.value(value).constant()?.as_i64()
    }

    /// The start of the object the host's table places `at` in, where its
    /// extent is exact. A recovered extent is an end the code's accesses
    /// suggest, one member of an aggregate as often as the whole: no object a
    /// pointer is formed into, whose accesses are judged where they land.
    fn landing(&self, state: &State<T>, at: Address) -> Option<Address> {
        let object = state
            .memory()
            .stated_object(at)
            .filter(|object| object.source.exact())?;
        Some(Address::new(at.root(), object.start, at.pointer_size()))
    }

    /// The address `value` holds, read as the run reads an access's address.
    fn address(&self, state: &State<T>, value: SsaVarId) -> Option<Address> {
        let held: Value<T> = state.value(value);
        self.access.address_of(&held, None)
    }

    /// The bytes an access of `ty` spans in memory.
    fn bytes(&self, ty: &T::Type) -> Option<usize> {
        self.access
            .type_layout::<T>(ty)?
            .storage_bytes(self.access.ptr_size())
    }

    /// Whether `value` is a literal the code writes.
    fn literal(&self, value: SsaVarId) -> bool {
        matches!(
            self.ssa
                .recorded_definition(value)
                .map(|definition| definition.op()),
            Some(SsaOp::Const { .. })
        )
    }
}

/// The pointers an operation moves bytes through one way: none, one, or
/// several.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Served {
    /// No pointer serves this way.
    None,
    /// One pointer does.
    One(SsaVarId),
    /// Two different pointers do.
    Several,
}

impl Served {
    /// Adds `pointer` to the pointers serving this way.
    pub fn add(&mut self, pointer: SsaVarId) {
        *self = match *self {
            Self::None => Self::One(pointer),
            Self::One(one) if one == pointer => Self::One(one),
            Self::One(_) | Self::Several => Self::Several,
        };
    }

    /// The one pointer serving this way, where exactly one does.
    #[must_use]
    pub const fn one(self) -> Option<SsaVarId> {
        match self {
            Self::One(pointer) => Some(pointer),
            Self::None | Self::Several => None,
        }
    }
}
