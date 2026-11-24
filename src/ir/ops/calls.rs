//! What a call does beyond computing its result from its arguments.
//!
//! A call is a contract: the values it takes and returns, the memory and state
//! it reads and may write, and what else it does - to the world outside the
//! program, to control flow, to the program's own code by calling back into
//! it. [`CallEffects`] states that contract, and a call operation carries it
//! ([`SsaOp::Call`] and its siblings), so every pass reads
//! the same answer: Memory SSA versions exactly the memory a call may write,
//! the memory pass forwards values across a call that cannot touch them, and
//! dead-code elimination removes a call that has no effect at all.
//!
//! A call whose effects nobody stated carries none, and is what every call was
//! before: it may read and write any memory, act on the world, run any code,
//! throw, or not return.
//!
//! # One model for every kind of callee
//!
//! A WinAPI function writing through an out pointer, `memcpy`, a script
//! engine's command writing a named variable, a runtime helper that reads
//! `errno` - they differ only in how the memory they touch is named, and
//! [`CallLocation`] has a form for each: memory reached through an argument
//! ([`Pointee`](CallLocation::Pointee)), a named cell
//! ([`Field`](CallLocation::Field): a script's global variable, a managed
//! static, runtime state the host names, such as a last-error slot), and any
//! memory the program has let escape ([`Escaped`](CallLocation::Escaped)).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        ops::{
            def::SsaOp,
            effects::{ControlEffect, MemoryEffectLocation, SsaEffectKind, SsaEffects},
        },
        variable::SsaVarId,
    },
    metadata::MetadataRef,
    target::Target,
};

/// How a call uses a location it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AccessMode {
    /// Reads it and leaves it.
    Read,
    /// Replaces it on every call, without reading it first.
    Write,
    /// Replaces it on some calls and leaves it on others: the value from
    /// before may survive.
    MayWrite,
    /// Reads it, then replaces it.
    Update,
}

impl AccessMode {
    /// Reports whether the call reads the location.
    #[must_use]
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::Update)
    }

    /// Reports whether the call may change the location.
    #[must_use]
    pub const fn writes(self) -> bool {
        !matches!(self, Self::Read)
    }
}

/// How much memory a [`CallLocation::Pointee`] covers, from the address the
/// argument holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PointeeExtent {
    /// A fixed number of bytes: a `RECT *` out parameter covers 16.
    Bytes(u64),
    /// As many bytes as argument `n` says: `memcpy`'s destination covers its
    /// third argument's worth.
    InArgument(u32),
    /// An amount the summary cannot bound: a string, a buffer ended by a
    /// terminator.
    Unbounded,
}

impl PointeeExtent {
    /// This extent at a call passing `args`: one an argument counts is as
    /// many bytes as that argument holds where it is a constant,
    /// `memcpy(dst, src, 208)`'s 208, and stays counted by the argument where
    /// it is none. Any other extent is its own.
    ///
    /// # Arguments
    ///
    /// * `args` - The call's arguments, by position.
    /// * `constant` - The unsigned constant a value is, where it is one.
    #[must_use]
    pub fn at_call(self, args: &[SsaVarId], constant: impl Fn(SsaVarId) -> Option<u64>) -> Self {
        match self {
            Self::InArgument(index) => usize::try_from(index)
                .ok()
                .and_then(|index| args.get(index))
                .and_then(|count| constant(*count))
                .map_or(self, Self::Bytes),
            Self::Bytes(_) | Self::Unbounded => self,
        }
    }
}

/// The point a [`CallLocation::Frame`] is measured from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FrameOrigin {
    /// The stack pointer the function was entered with.
    Entry,
    /// The value the function's one stack alignment leaves
    /// ([`AddressWalk::frame_anchor`](crate::analysis::address::AddressWalk::frame_anchor)): how far that lies
    /// below entry is no number, and every byte below it has an exact offset
    /// from it. A function with no single alignment places nothing here.
    Alignment,
}

/// Memory a call reads or writes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::FieldRef: serde::Serialize",
        deserialize = "T::FieldRef: serde::Deserialize<'de>"
    ))
)]
pub enum CallLocation<T: Target> {
    /// The memory argument `arg` points to, from its address on.
    Pointee {
        /// The argument's position in the call's argument list.
        arg: u32,
        /// How far the memory runs.
        extent: PointeeExtent,
    },
    /// A named cell: one static location, the same wherever it is named.
    Field(T::FieldRef),
    /// Any memory the program has let escape: every location but the
    /// function's own frame cells whose address it has kept to itself.
    Escaped,
    /// Bytes of the calling function's own frame: the stack a callee owns
    /// above the stack pointer it is called with, where its arguments and a
    /// convention's register home area lie. Named by place rather than reached
    /// through an address, so what the call does to them is what this access
    /// says whatever escaped memory it reaches: no object of the caller lies
    /// where its callee's parameters do. Written ([`AccessMode::Write`]), they
    /// hold nothing of the caller's once the call returns.
    Frame {
        /// The point `offset` is measured from.
        origin: FrameOrigin,
        /// The first byte, from `origin`.
        offset: i64,
        /// How many bytes.
        bytes: u64,
    },
}

/// One location a call touches, and how.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::FieldRef: serde::Serialize",
        deserialize = "T::FieldRef: serde::Deserialize<'de>"
    ))
)]
pub struct CallAccess<T: Target> {
    /// The memory.
    pub location: CallLocation<T>,
    /// What the call does to it.
    pub access: AccessMode,
}

/// The memory a call may touch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::FieldRef: serde::Serialize",
        deserialize = "T::FieldRef: serde::Deserialize<'de>"
    ))
)]
pub enum CallMemory<T: Target> {
    /// Any memory at all, read or written.
    Unknown,
    /// Exactly these locations, and nothing else. An empty list is a call
    /// that touches no memory.
    Only(Vec<CallAccess<T>>),
}

/// Whether control comes back from a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CallTermination {
    /// It always returns.
    Returns,
    /// It returns on some calls and ends the program, or the code that made
    /// it, on others: `exit` behind a condition, a script command the user can
    /// abort.
    MayNotReturn,
    /// It never returns: `exit`, `abort`, `ExitProcess`.
    NeverReturns,
}

/// Which of the pointers a call is given it may keep past the call: storing
/// one somewhere, handing it to a thread, or returning it. Whatever a kept
/// pointer points to escapes; a pointer the call does not keep reaches only
/// what the call's own accesses through it state ([`CallLocation::Pointee`]),
/// while it runs, and comes back in nothing the call returns.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Capture {
    /// None of them.
    Nothing,
    /// Any of them.
    Any,
    /// Any but these arguments, by position in the call's argument list,
    /// ascending and never none (that is [`Capture::Any`]): a `va_list` a
    /// callee only reads with `va_arg` is reached while it runs, whatever it
    /// does with its other arguments.
    AnyBut(Vec<u32>),
}

impl Capture {
    /// Whether the call may keep the pointer it is given as argument `arg`.
    #[must_use]
    pub fn keeps(&self, arg: u32) -> bool {
        match self {
            Self::Nothing => false,
            Self::Any => true,
            Self::AnyBut(spared) => spared.binary_search(&arg).is_err(),
        }
    }

    /// Whether the call may keep any pointer it is given.
    #[must_use]
    pub fn keeps_any(&self) -> bool {
        !matches!(self, Self::Nothing)
    }

    /// This capture once the call's arguments move: `moved` gives each
    /// argument's new position, `None` for one the call no longer passes,
    /// which it keeps no more than it is passed it. A call that spared only
    /// arguments it no longer passes may keep any it is passed.
    #[must_use]
    pub fn moved(self, moved: impl Fn(u32) -> Option<u32>) -> Self {
        match self {
            Self::Nothing | Self::Any => self,
            Self::AnyBut(spared) => {
                let mut spared: Vec<u32> = spared.into_iter().filter_map(moved).collect();
                spared.sort_unstable();
                spared.dedup();
                if spared.is_empty() {
                    Self::Any
                } else {
                    Self::AnyBut(spared)
                }
            }
        }
    }

    /// This capture with argument `arg` not kept.
    #[must_use]
    pub fn sparing(self, arg: u32) -> Self {
        match self {
            Self::Nothing => Self::Nothing,
            Self::Any => Self::AnyBut(vec![arg]),
            Self::AnyBut(mut spared) => {
                if let Err(at) = spared.binary_search(&arg) {
                    spared.insert(at, arg);
                }
                Self::AnyBut(spared)
            }
        }
    }
}

/// What a call does beyond computing its result from its arguments.
///
/// See the [module documentation](self).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::FieldRef: serde::Serialize",
        deserialize = "T::FieldRef: serde::Deserialize<'de>"
    ))
)]
pub struct CallEffects<T: Target> {
    /// The memory it reads and may write.
    pub memory: CallMemory<T>,
    /// Whether it acts on the world outside the program - a file, a window,
    /// the registry. Such a call is kept however little of its result the
    /// program reads, but it touches no program memory it does not name.
    pub world: bool,
    /// Whether it may run the program's own code: a plugin, a sort's
    /// comparator, a window procedure. That code may touch anything, so
    /// [`CallEffects::accesses`] answers `None` for such a call whatever
    /// [`Self::memory`] lists.
    pub calls_back: bool,
    /// Which pointers it was given it may keep past the call, which makes
    /// whatever they point to escape.
    pub captures: Capture,
    /// Whether control comes back.
    pub termination: CallTermination,
    /// Whether control can come back more than once (`setjmp`, `vfork`).
    pub returns_twice: bool,
    /// Whether it may throw an exception into the caller.
    pub may_throw: bool,
}

impl<T: Target> CallEffects<T> {
    /// What a call nobody described may do: anything.
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            memory: CallMemory::Unknown,
            world: true,
            calls_back: true,
            captures: Capture::Any,
            termination: CallTermination::MayNotReturn,
            returns_twice: false,
            may_throw: true,
        }
    }

    /// A call that computes its result from its arguments and does nothing
    /// else: `abs`, a script's string length.
    #[must_use]
    pub fn pure() -> Self {
        Self {
            memory: CallMemory::Only(Vec::new()),
            world: false,
            calls_back: false,
            captures: Capture::Nothing,
            termination: CallTermination::Returns,
            returns_twice: false,
            may_throw: false,
        }
    }

    /// These effects for the call once its arguments move: `moved` gives each
    /// argument's new position, `None` for one the call no longer passes.
    ///
    /// Effects name arguments by position, so whatever rewrites a call's
    /// argument list says where each went. Memory reached through an argument
    /// no longer passed is no longer reached, since the callee is not handed
    /// the pointer; an extent counted by one is no longer bounded; and an
    /// argument the call did not keep is the same argument at its new
    /// position.
    #[must_use]
    pub fn moved(mut self, moved: impl Fn(u32) -> Option<u32>) -> Self {
        if let CallMemory::Only(accesses) = &mut self.memory {
            accesses.retain_mut(|access| {
                let CallLocation::Pointee { arg, extent } = &mut access.location else {
                    return true;
                };
                let Some(position) = moved(*arg) else {
                    return false;
                };
                *arg = position;
                if let PointeeExtent::InArgument(index) = extent {
                    *extent =
                        moved(*index).map_or(PointeeExtent::Unbounded, PointeeExtent::InArgument);
                }
                true
            });
        }
        self.captures = self.captures.moved(moved);
        self
    }

    /// Returns these effects with `access` added to the memory touched. Adding
    /// to [`CallMemory::Unknown`] leaves it unknown.
    #[must_use]
    pub fn with_access(mut self, location: CallLocation<T>, access: AccessMode) -> Self {
        if let CallMemory::Only(accesses) = &mut self.memory {
            accesses.push(CallAccess { location, access });
        }
        self
    }

    /// Returns the locations the call touches, when it is known to touch no
    /// others. `None` when its memory is unknown, or when it may run program
    /// code and its list does not already let it read and write all escaped
    /// memory - the most the program's own code can reach from a callback.
    #[must_use]
    pub fn accesses(&self) -> Option<&[CallAccess<T>]> {
        match &self.memory {
            CallMemory::Only(accesses)
                if !self.calls_back
                    || accesses.iter().any(|access| {
                        matches!(access.location, CallLocation::Escaped)
                            && access.access == AccessMode::Update
                    }) =>
            {
                Some(accesses)
            }
            _ => None,
        }
    }

    /// Reports whether the call may change any memory the program can see.
    #[must_use]
    pub fn writes_memory(&self) -> bool {
        self.accesses()
            .is_none_or(|accesses| accesses.iter().any(|a| a.access.writes()))
    }

    /// Reports whether the call may read any memory.
    #[must_use]
    pub fn reads_memory(&self) -> bool {
        self.accesses()
            .is_none_or(|accesses| accesses.iter().any(|a| a.access.reads()))
    }

    /// Reports whether the call does nothing but compute its result: removed
    /// when that is unused, merged with an identical call, moved freely.
    #[must_use]
    pub fn is_pure(&self) -> bool {
        matches!(&self.memory, CallMemory::Only(accesses) if accesses.is_empty())
            && !self.calls_back
            && !self.captures.keeps_any()
            && !self.world
            && matches!(self.termination, CallTermination::Returns)
            && !self.returns_twice
            && !self.may_throw
    }

    /// The summary [`SsaOp::effects`](crate::ir::ops::SsaOp::effects) reports for a
    /// call with these effects.
    ///
    /// A pure call is [`SsaEffectKind::Pure`]. One that only reads memory and
    /// otherwise stays within the program is [`SsaEffectKind::Read`]. Every
    /// other call is [`SsaEffectKind::Call`], which orders it against every
    /// other effect; Memory SSA and the memory pass read the locations
    /// themselves ([`Self::accesses`]), so that is no loss of precision about
    /// memory.
    #[must_use]
    pub fn summary(&self) -> SsaEffects {
        if self.is_pure() {
            return SsaEffects::new(SsaEffectKind::Pure, false);
        }
        let stays_inside = !self.world
            && !self.calls_back
            && !self.captures.keeps_any()
            && matches!(self.termination, CallTermination::Returns)
            && !self.returns_twice;
        let summary = if stays_inside && !self.writes_memory() {
            SsaEffects::new(SsaEffectKind::Read, self.may_throw)
        } else {
            SsaEffects::new(SsaEffectKind::Call, self.may_throw).with_control(ControlEffect::Call)
        };
        if !self.calls_back
            && matches!(&self.memory, CallMemory::Only(accesses) if accesses.is_empty())
        {
            summary.with_memory(MemoryEffectLocation::None)
        } else {
            summary
        }
    }
}

impl<T: Target> CallAccess<T> {
    /// The address argument this access is reached through, for a
    /// [`CallLocation::Pointee`] of a call whose arguments are `args`.
    #[must_use]
    pub fn pointer_argument(&self, args: &[SsaVarId]) -> Option<SsaVarId> {
        match self.location {
            CallLocation::Pointee { arg, .. } => args.get(usize::try_from(arg).ok()?).copied(),
            CallLocation::Field(_) | CallLocation::Escaped | CallLocation::Frame { .. } => None,
        }
    }
}

/// The code a call runs, and how the call reaches it.
///
/// Every call is one operation ([`SsaOp::Call`]); this is what tells calls
/// apart. Anything a pass learns from a named method's *declaration* holds for
/// every callee that names it; anything it learns from that method's *body*
/// holds only where the method certainly runs ([`Self::direct`]), since a
/// virtual or constrained call may run an override with a body of its own.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::MethodRef: serde::Serialize, T::TypeRef: serde::Serialize, \
                     T::SigRef: serde::Serialize",
        deserialize = "T::MethodRef: serde::Deserialize<'de>, T::TypeRef: serde::Deserialize<'de>, \
                       T::SigRef: serde::Deserialize<'de>"
    ))
)]
pub enum Callee<T: Target> {
    /// The named method runs (CIL `call`, a native direct call).
    Direct(T::MethodRef),
    /// The receiver, the call's first argument, selects an implementation of
    /// the named method (CIL `callvirt`).
    Virtual(T::MethodRef),
    /// `constraint_type` selects the implementation of the named method that
    /// runs (CIL `constrained.`). Of an instance method (`constrained.
    /// callvirt`) the receiver, the call's first argument, is the address of
    /// a value of that type: where the type implements the method it is
    /// called directly on that value, and otherwise on the value boxed, so
    /// one call serves value and reference types alike. Of a static virtual
    /// method (`constrained. call`) there is no receiver: the type's own
    /// implementation runs.
    Constrained {
        /// The method named.
        method: T::MethodRef,
        /// The type that selects the implementation.
        constraint_type: T::TypeRef,
        /// Whether the method is an instance method, taking the receiver's
        /// address first; a static virtual method takes none.
        instance: bool,
    },
    /// The code a function pointer holds runs, as `signature` states its
    /// interface (CIL `calli`, a native indirect call).
    Indirect {
        /// The function pointer operand.
        fptr: SsaVarId,
        /// The signature the call is made by.
        signature: T::SigRef,
    },
}

impl<T: Target> Callee<T> {
    /// The method the call names, whichever implementation of it runs; `None`
    /// for an indirect call.
    #[must_use]
    pub fn method(&self) -> Option<&T::MethodRef> {
        match self {
            Self::Direct(method) | Self::Virtual(method) | Self::Constrained { method, .. } => {
                Some(method)
            }
            Self::Indirect { .. } => None,
        }
    }

    /// The method the call names, to retarget it; `None` for an indirect call.
    #[must_use]
    pub fn method_mut(&mut self) -> Option<&mut T::MethodRef> {
        match self {
            Self::Direct(method) | Self::Virtual(method) | Self::Constrained { method, .. } => {
                Some(method)
            }
            Self::Indirect { .. } => None,
        }
    }

    /// The method that certainly runs: the named one of a direct call, `None`
    /// for every other callee, whose code may be an override or a pointer's.
    ///
    /// A pass that binds facts recovered from the callee's body (its
    /// parameters, its return value, its effects as executed) reads this
    /// rather than [`Self::method`].
    #[must_use]
    pub fn direct(&self) -> Option<&T::MethodRef> {
        match self {
            Self::Direct(method) => Some(method),
            Self::Virtual(_) | Self::Constrained { .. } | Self::Indirect { .. } => None,
        }
    }

    /// The function pointer an indirect call reads; `None` for every other
    /// callee.
    #[must_use]
    pub const fn fptr(&self) -> Option<SsaVarId> {
        match self {
            Self::Indirect { fptr, .. } => Some(*fptr),
            Self::Direct(_) | Self::Virtual(_) | Self::Constrained { .. } => None,
        }
    }
}

/// A value a caller put on the stack where the call it makes may read it as
/// an argument, which nothing yet says it does ([`CallData::staged`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StagedValue {
    /// Where it lies, in bytes from where the call's stack arguments begin.
    pub offset: i64,
    /// The value put there.
    pub value: SsaVarId,
}

/// An opaque identity for one implicit input in a host's transfer contract.
///
/// The host defines the namespace and meaning of each key, such as an ABI
/// address base or a runtime context. The framework assigns no architectural
/// meaning to the integer. A key is independent of language argument indices
/// and is interpreted under the same host contract as the callee's body.
///
/// # Examples
///
/// A host-defined address base is passed outside the language signature.
/// The callee's host binds this key to its matching entry input through
/// [`crate::analysis::execution::ParameterBindings::bind_implicit`].
///
/// ```rust
/// use analyssa::{
///     ir::{CallData, CallInputKey, Callee, SsaVarId},
///     testing::MockTarget,
/// };
///
/// let address_base = CallInputKey::new(17);
/// let current_value = SsaVarId::from_index(3);
/// let call = CallData::<MockTarget>::new(None, Callee::Direct(5), vec![])
///     .with_implicit_input(address_base, current_value);
/// assert!(call.args.is_empty());
/// assert_eq!(call.implicit_inputs[0].key, address_base);
/// assert_eq!(call.implicit_inputs[0].value, current_value);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct CallInputKey(u32);

impl CallInputKey {
    /// Returns the opaque key whose identity is `value` in the host's namespace.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the integer identity without assigning it an argument position.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// What a return states for static execution alone, beside what it returns
/// ([`SsaOp::Return`]): the value it
/// transfers control through and the registers it hands its caller. A host
/// done executing a function releases both before it renders it
/// ([`SsaFunction::release_execution_operands`](crate::ir::function::SsaFunction::release_execution_operands)).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReturnTransfer {
    /// The value the return transfers control through, where the
    /// architecture returns through a register: the link register's value
    /// at `ret` (AArch64's `x30`, Arm's `lr`, RISC-V's `ra`, MIPS's `$ra`).
    /// The executor checks it is the return address the function was entered
    /// with. `None` where the return reads its target from the stack, which
    /// the executor checks in memory, or the function's source states none.
    #[cfg_attr(feature = "serde", serde(default))]
    pub target: Option<SsaVarId>,
    /// The machine registers the return leaves its caller, each under a key
    /// the host defines, where the code follows no convention that says
    /// which the caller reads: code recovered at run time hands back every
    /// register it names. Each is an ordinary use. Empty where a convention
    /// states what a caller receives.
    #[cfg_attr(feature = "serde", serde(default))]
    pub handover: Box<[CallInput]>,
}

impl ReturnTransfer {
    /// A return through `target` that hands its caller no register.
    #[must_use]
    pub fn through(target: SsaVarId) -> Box<Self> {
        Box::new(Self {
            target: Some(target),
            handover: Box::default(),
        })
    }

    /// Whether it states nothing: no target and no register.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.target.is_none() && self.handover.is_empty()
    }
}

/// One implicit value read by a control transfer under a named host contract.
///
/// The value is an ordinary SSA use and remains live even though it is absent
/// from the language signature. Its key names the callee-side semantic input;
/// the host justifies the pairing to the callee's entry SSA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CallInput {
    /// The semantic slot supplied by this input, independent of argument order.
    pub key: CallInputKey,
    /// The actual value reaching the transfer.
    pub value: SsaVarId,
}

/// A physical location supplied by one fresh call-clobber definition.
///
/// The index names an existing definition in the operation's output list, so
/// SSA renaming cannot separate a physical key from its defining value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CallOutput {
    /// Physical location under the host's machine contract.
    pub key: CallInputKey,
    /// Position in the call-clobber operation's output definition list.
    pub index: u32,
}

/// One value a call returns beside its result, under the key naming where its
/// callee leaves it ([`CallData::results`]).
///
/// A callee whose callers read a second register it leaves a value in returns
/// two values: NSIS's `GetIntFromParm` leaves `myatoi`'s integer in `eax` and
/// the string's first character in `edx`, and its callers keep each. The key
/// names the place in the host's namespace, as a [`CallInput`]'s does, and the
/// callee's return hands the value back under the same key
/// ([`SsaOp::Return`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CallResult {
    /// Where the callee leaves the value, under the host's machine contract.
    pub key: CallInputKey,
    /// The value the call defines there.
    pub value: SsaVarId,
}

/// A call: what it runs, the values it takes and returns, and what it does
/// beyond computing its result ([`SsaOp::Call`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::MethodRef: serde::Serialize, T::TypeRef: serde::Serialize, \
                     T::SigRef: serde::Serialize, T::FieldRef: serde::Serialize",
        deserialize = "T::MethodRef: serde::Deserialize<'de>, T::TypeRef: serde::Deserialize<'de>, \
                       T::SigRef: serde::Deserialize<'de>, T::FieldRef: serde::Deserialize<'de>"
    ))
)]
pub struct CallData<T: Target> {
    /// The variable receiving the result, if the call has one.
    pub dest: Option<SsaVarId>,
    /// The code the call runs, and how it reaches it.
    pub callee: Callee<T>,
    /// The arguments, the receiver first for a virtual or constrained call.
    pub args: Vec<SsaVarId>,
    /// Host-defined input values consumed outside the language signature.
    ///
    /// Inputs retain their emission order; the last occurrence of a repeated
    /// key supplies that semantic slot. Every listed value is an SSA use.
    /// Receiver selection and language arity depend only on [`Self::args`].
    /// The first two inputs use inline storage, avoiding a separate allocation.
    #[cfg_attr(feature = "serde", serde(default))]
    pub implicit_inputs: SmallVec<[CallInput; 2]>,
    /// The values the call returns beside [`Self::dest`], each under the key
    /// naming where its callee leaves it ([`CallResult`]). Each is a
    /// definition, as `dest` is. Empty for a call that returns one value or
    /// none.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "SmallVec::is_empty")
    )]
    pub results: SmallVec<[CallResult; 1]>,
    /// What the call does beyond computing its result, when the host stated
    /// it; `None` when nobody did, and the call may do anything.
    pub effects: Option<CallEffects<T>>,
    /// Whether the call is a tail call: the caller's frame is released before
    /// the callee runs, and the callee's return is the caller's (CIL `tail.`,
    /// a native jump to a function).
    pub tail: bool,
    /// Values the caller staged on the stack that the call may read as
    /// arguments, which no argument of it is until something the host knows
    /// of the callee places one there ([`Self::promote`]).
    ///
    /// A native caller passes its first arguments in registers and the rest
    /// on the stack once the registers are taken - but an argument too wide
    /// for the registers left skips them (an even register pair, a record
    /// passed in registers or not at all), so a value stored where a stack
    /// argument goes beside an unset register may be one, and may be the
    /// caller's own spill. Each is a use of its value, so it lives as long as
    /// the call does, and no reading of the call's arguments sees it.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub staged: Vec<StagedValue>,
}

impl<T: Target> CallData<T> {
    /// Creates a call of `callee` with ordinary `args` and an optional result in `dest`.
    ///
    /// The call has no implicit inputs, unknown effects and ordinary return
    /// control. Set an input through [`Self::with_implicit_input`], effects
    /// through [`Self::with_effects`] or tail control through [`Self::made_tail`].
    #[must_use]
    pub const fn new(dest: Option<SsaVarId>, callee: Callee<T>, args: Vec<SsaVarId>) -> Self {
        Self {
            dest,
            callee,
            args,
            implicit_inputs: SmallVec::new_const(),
            results: SmallVec::new_const(),
            effects: None,
            tail: false,
            staged: Vec::new(),
        }
    }

    /// Makes the value staged at `offset` the call's next argument, and
    /// returns it; `None` where nothing is staged there.
    pub fn promote(&mut self, offset: i64) -> Option<SsaVarId> {
        let at = self
            .staged
            .iter()
            .position(|staged| staged.offset == offset)?;
        let staged = self.staged.remove(at);
        self.args.push(staged.value);
        Some(staged.value)
    }

    /// Whether the call reads `value` only as a staged one: staged, and
    /// neither an argument, an implicit input nor the function pointer it calls
    /// through. Such a
    /// use says the value is where an argument may be, and nothing of what it
    /// is.
    #[must_use]
    pub fn only_stages(&self, value: SsaVarId) -> bool {
        self.staged.iter().any(|staged| staged.value == value)
            && !self.args.contains(&value)
            && !self
                .implicit_inputs
                .iter()
                .any(|input| input.value == value)
            && !matches!(self.callee, Callee::Indirect { fptr, .. } if fptr == value)
    }

    /// Appends an implicit caller value under its semantic input key.
    ///
    /// A repeated key replaces its previous meaning at the call; every source
    /// value remains an SSA use. This IR construction helper uses ordinary
    /// allocation when the inline capacity is exceeded. Execution admits its
    /// separate value captures through the invocation's resource policy.
    #[must_use]
    pub fn with_implicit_input(mut self, key: CallInputKey, value: SsaVarId) -> Self {
        self.implicit_inputs.push(CallInput { key, value });
        self
    }

    /// Appends a value the call returns beside its result, defined in
    /// `value`, under the key naming where its callee leaves it
    /// ([`Self::results`]).
    #[must_use]
    pub fn with_result(mut self, key: CallInputKey, value: SsaVarId) -> Self {
        self.results.push(CallResult { key, value });
        self
    }

    /// This call with the effects its host states.
    #[must_use]
    pub fn with_effects(mut self, effects: CallEffects<T>) -> Self {
        self.effects = Some(effects);
        self
    }

    /// This call made as a tail call ([`Self::tail`]).
    #[must_use]
    pub fn made_tail(mut self) -> Self {
        self.tail = true;
        self
    }

    /// Redirects the call to `callee` with replacement ordinary `args`.
    ///
    /// Preserves its destination and the values returned beside it, implicit
    /// inputs, staged stack values and tail control. Discards effects because
    /// they describe the original callee.
    #[must_use]
    pub fn redirected(&self, callee: Callee<T>, args: Vec<SsaVarId>) -> Self {
        Self {
            dest: self.dest,
            callee,
            args,
            implicit_inputs: self.implicit_inputs.clone(),
            results: self.results.clone(),
            effects: None,
            tail: self.tail,
            staged: self.staged.clone(),
        }
    }

    /// The receiver a virtual call's runtime type selects an implementation
    /// by: its first argument. `None` for every other callee (a constrained
    /// call's first argument is an address, not an object) and for a virtual
    /// call with no arguments.
    #[must_use]
    pub fn receiver(&self) -> Option<SsaVarId> {
        match self.callee {
            Callee::Virtual(_) => self.args.first().copied(),
            Callee::Direct(_) | Callee::Constrained { .. } | Callee::Indirect { .. } => None,
        }
    }
}

impl<T: Target> From<CallData<T>> for SsaOp<T> {
    fn from(call: CallData<T>) -> Self {
        Self::Call(Box::new(call))
    }
}

impl<T: Target> Emit<T> for AccessMode {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Read => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Write => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::MayWrite => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Update => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for CallAccess<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self { location, access } = self;
        Emit::emit(location, writer)?;
        Emit::emit(access, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for Callee<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Direct(method) => {
                Emit::emit(&0u16, writer)?;
                writer.metadata(MetadataRef::Method(method))
            }
            Self::Virtual(method) => {
                Emit::emit(&1u16, writer)?;
                writer.metadata(MetadataRef::Method(method))
            }
            Self::Constrained {
                method,
                constraint_type,
                instance,
            } => {
                Emit::emit(&2u16, writer)?;
                writer.metadata(MetadataRef::Method(method))?;
                writer.metadata(MetadataRef::TypeRef(constraint_type))?;
                Emit::emit(instance, writer)
            }
            Self::Indirect { fptr, signature } => {
                Emit::emit(&3u16, writer)?;
                Emit::emit(fptr, writer)?;
                writer.metadata(MetadataRef::Signature(signature))
            }
        }
    }
}

impl<T: Target> Emit<T> for CallInputKey {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        Emit::emit(&self.0, writer)
    }
}

impl<T: Target> Emit<T> for CallInput {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        Emit::emit(&self.key, writer)?;
        Emit::emit(&self.value, writer)
    }
}

impl<T: Target> Emit<T> for CallResult {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        Emit::emit(&self.key, writer)?;
        Emit::emit(&self.value, writer)
    }
}

impl<T: Target> Emit<T> for CallOutput {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        Emit::emit(&self.key, writer)?;
        Emit::emit(&self.index, writer)
    }
}

impl<T: Target> Emit<T> for StagedValue {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        Emit::emit(&self.offset, writer)?;
        Emit::emit(&self.value, writer)
    }
}

impl<T: Target> Emit<T> for CallData<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            dest,
            callee,
            args,
            implicit_inputs,
            // Emitted by the operation, under a tag of its own where there are
            // any ([`SsaOp::Call`]): a call returning one value keeps the
            // encoding it had before calls returned several.
            results: _,
            effects,
            tail,
            staged,
        } = self;
        Emit::emit(dest, writer)?;
        Emit::emit(callee, writer)?;
        Emit::emit(args, writer)?;
        Emit::emit(implicit_inputs.as_slice(), writer)?;
        Emit::emit(effects, writer)?;
        Emit::emit(tail, writer)?;
        Emit::emit(staged, writer)
    }
}

impl<T: Target> Emit<T> for CallEffects<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            memory,
            world,
            calls_back,
            captures,
            termination,
            returns_twice,
            may_throw,
        } = self;
        Emit::emit(memory, writer)?;
        Emit::emit(world, writer)?;
        Emit::emit(calls_back, writer)?;
        Emit::emit(captures, writer)?;
        Emit::emit(termination, writer)?;
        Emit::emit(returns_twice, writer)?;
        Emit::emit(may_throw, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for Capture {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Nothing => Emit::emit(&false, writer),
            Self::Any => Emit::emit(&true, writer),
            Self::AnyBut(spared) => {
                Emit::emit(&2u8, writer)?;
                Emit::emit(spared, writer)
            }
        }
    }
}

impl<T: Target> Emit<T> for CallLocation<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Pointee { arg, extent } => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(arg, writer)?;
                Emit::emit(extent, writer)?;
                Ok(())
            }
            Self::Field(field_0) => {
                Emit::emit(&1u16, writer)?;
                writer.metadata(MetadataRef::Field(field_0))?;
                Ok(())
            }
            Self::Escaped => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Frame {
                origin,
                offset,
                bytes,
            } => {
                let tag: u16 = match origin {
                    FrameOrigin::Entry => 3,
                    FrameOrigin::Alignment => 4,
                };
                Emit::emit(&tag, writer)?;
                Emit::emit(offset, writer)?;
                Emit::emit(bytes, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for CallMemory<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Unknown => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Only(field_0) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for CallTermination {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Returns => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::MayNotReturn => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::NeverReturns => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for PointeeExtent {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Bytes(field_0) => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::InArgument(field_0) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Unbounded => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockTarget;

    type Effects = CallEffects<MockTarget>;

    #[test]
    fn an_unstated_call_may_do_anything() {
        let unknown = Effects::unknown();
        assert!(unknown.accesses().is_none());
        assert!(unknown.reads_memory() && unknown.writes_memory());
        assert!(!unknown.is_pure());
        assert_eq!(unknown.summary().kind, SsaEffectKind::Call);
        assert!(unknown.summary().may_throw);
    }

    /// **A call spares the arguments it is told it does not keep.** Sparing
    /// an argument of a call that may keep any keeps every other, spared
    /// arguments stay in order however they are added, and a call that keeps
    /// nothing keeps nothing spared or not.
    #[test]
    fn a_call_spares_the_arguments_it_does_not_keep() {
        let spared = Capture::Any.sparing(3).sparing(1).sparing(3);
        assert_eq!(spared, Capture::AnyBut(vec![1, 3]));
        assert!(!spared.keeps(1) && !spared.keeps(3));
        assert!(spared.keeps(0) && spared.keeps(2));
        assert!(spared.keeps_any());
        assert_eq!(Capture::Nothing.sparing(0), Capture::Nothing);
        assert!(!Capture::Nothing.keeps(0) && !Capture::Nothing.keeps_any());
    }

    /// **Effects follow the arguments they name.** A call whose list loses its
    /// first argument and swaps the others reaches through each where it now
    /// is: the pointee of the dropped argument goes, the one an extent counted
    /// by a dropped argument is no longer bounded, and an argument the call
    /// does not keep is still not kept at its new position.
    #[test]
    fn effects_follow_the_arguments_they_name() {
        let pointee = |arg, extent| CallAccess {
            location: CallLocation::<MockTarget>::Pointee { arg, extent },
            access: AccessMode::Read,
        };
        let mut effects = Effects::pure()
            .with_access(
                CallLocation::Pointee {
                    arg: 0,
                    extent: PointeeExtent::Bytes(4),
                },
                AccessMode::Read,
            )
            .with_access(
                CallLocation::Pointee {
                    arg: 1,
                    extent: PointeeExtent::InArgument(0),
                },
                AccessMode::Read,
            )
            .with_access(
                CallLocation::Pointee {
                    arg: 2,
                    extent: PointeeExtent::InArgument(1),
                },
                AccessMode::Read,
            );
        effects.captures = Capture::Any.sparing(1).sparing(0);
        // 0 is dropped, 1 and 2 swap.
        let moved = effects.moved(|arg| match arg {
            1 => Some(1),
            2 => Some(0),
            _ => None,
        });
        assert_eq!(
            moved.memory,
            CallMemory::Only(vec![
                pointee(1, PointeeExtent::Unbounded),
                pointee(0, PointeeExtent::InArgument(1)),
            ])
        );
        assert_eq!(moved.captures, Capture::AnyBut(vec![1]));
    }

    /// **A call that spared only arguments it no longer passes keeps any.**
    /// Dropping every spared argument leaves the one spelling of a call that
    /// may keep any pointer it is given, so two copies of one call, one of
    /// them never told of the dropped argument, state one capture.
    #[test]
    fn a_call_sparing_only_dropped_arguments_keeps_any() {
        let moved = Capture::Any
            .sparing(2)
            .moved(|arg| (arg < 2).then_some(arg));
        assert_eq!(moved, Capture::Any);
        assert!(moved.keeps(0) && moved.keeps(1));
    }

    #[test]
    fn a_pure_call_is_pure() {
        let pure = Effects::pure();
        assert!(pure.is_pure());
        assert_eq!(pure.summary(), SsaEffects::new(SsaEffectKind::Pure, false));
    }

    /// Capturing a pointer is observable even when no data-memory access is listed.
    #[test]
    fn capturing_call_is_not_pure_or_a_read() {
        let mut effects = Effects::pure();
        effects.captures = Capture::Any;
        assert!(!effects.is_pure());
        assert_eq!(effects.summary().kind, SsaEffectKind::Call);
    }

    /// A throwing capture remains a call independently of the pure fast-path predicate.
    #[test]
    fn throwing_capture_is_observable_beyond_read_effects() {
        let mut effects = Effects::pure();
        effects.captures = Capture::Any;
        effects.may_throw = true;
        assert_eq!(effects.summary().kind, SsaEffectKind::Call);
        assert!(effects.summary().may_throw);
    }

    /// Non-memory effects do not invent data-memory reads or writes.
    #[test]
    fn no_memory_call_summary_keeps_non_memory_effects_separate() {
        for case in 0..4 {
            let mut effects = Effects::pure();
            match case {
                0 => effects.captures = Capture::Any,
                1 => effects.world = true,
                2 => effects.returns_twice = true,
                _ => effects.may_throw = true,
            }
            let summary = effects.summary();
            assert!(
                !summary.reads_memory() && !summary.writes_memory(),
                "case {case}"
            );
        }
    }

    #[test]
    fn a_call_that_only_reads_is_a_read() {
        let strlen = Effects::pure().with_access(
            CallLocation::Pointee {
                arg: 0,
                extent: PointeeExtent::Unbounded,
            },
            AccessMode::Read,
        );
        assert!(!strlen.is_pure());
        assert!(strlen.reads_memory() && !strlen.writes_memory());
        assert_eq!(strlen.summary().kind, SsaEffectKind::Read);
    }

    #[test]
    fn a_call_that_writes_or_acts_is_a_call() {
        let out = Effects::pure().with_access(CallLocation::Field(3), AccessMode::Write);
        assert_eq!(out.summary().kind, SsaEffectKind::Call);
        let mut shown = Effects::pure();
        shown.world = true;
        assert_eq!(shown.summary().kind, SsaEffectKind::Call);
    }

    /// **Code the call runs reaches what has escaped, and no more.** A
    /// callback is the program's own code, and it finds this function's frame
    /// only through an address that left it - so a call that already reads
    /// and writes all escaped memory keeps its list when it calls back, and
    /// one that names less forgets it.
    #[test]
    fn a_call_back_reaches_escaped_memory() {
        let mut sort = Effects::pure().with_access(CallLocation::Escaped, AccessMode::Update);
        sort.calls_back = true;
        assert_eq!(sort.accesses().map(<[_]>::len), Some(1));
        let mut narrow = Effects::pure().with_access(CallLocation::Field(3), AccessMode::Read);
        narrow.calls_back = true;
        assert!(narrow.accesses().is_none());
    }

    /// Hidden inputs are real SSA uses without changing language argument identity.
    #[test]
    fn call_input_uses_remap_and_redirection_preserve_semantic_keys() {
        let key = CallInputKey::new(17);
        let other = CallInputKey::new(u32::MAX);
        let (dest, fptr, argument, hidden, last) = (
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
            SsaVarId::from_index(3),
            SsaVarId::from_index(7),
            SsaVarId::from_index(11),
        );
        let call = CallData::<MockTarget>::new(
            Some(dest),
            Callee::Indirect { fptr, signature: 0 },
            vec![argument],
        )
        .with_implicit_input(key, hidden)
        .with_implicit_input(other, hidden)
        .with_implicit_input(key, last)
        .with_effects(CallEffects::pure())
        .made_tail();
        let mut op = SsaOp::from(call.clone());
        assert_eq!(op.uses(), vec![fptr, argument, hidden, hidden, last]);
        let replacement = SsaVarId::from_index(19);
        assert_eq!(op.replace_uses(hidden, replacement), 2);
        assert_eq!(
            op.uses(),
            vec![fptr, argument, replacement, replacement, last]
        );
        assert_eq!(op.dest(), Some(dest));
        let rewritten = op.as_call().expect("a call");
        assert_eq!(rewritten.args, vec![argument]);
        assert_eq!(
            rewritten
                .implicit_inputs
                .iter()
                .map(|input| input.key)
                .collect::<Vec<_>>(),
            vec![key, other, key]
        );
        let shifted = op.remap_variables(|value| Some(SsaVarId::from_index(value.index() + 32)));
        assert_eq!(
            shifted.uses(),
            vec![34, 35, 51, 51, 43]
                .into_iter()
                .map(SsaVarId::from_index)
                .collect::<Vec<_>>()
        );
        let redirected = call.redirected(Callee::Direct(10), vec![]);
        assert_eq!(redirected.implicit_inputs, call.implicit_inputs);
        assert!(redirected.args.is_empty() && redirected.effects.is_none() && redirected.tail);
        assert!(redirected.receiver().is_none());
        assert!(
            CallData::<MockTarget>::new(None, Callee::Virtual(9), vec![])
                .with_implicit_input(key, hidden)
                .receiver()
                .is_none()
        );
    }

    /// Implicit inputs and staged candidates remain distinct under visitation and rewriting.
    #[test]
    fn staged_and_implicit_inputs_preserve_shared_uses_and_visitor_order() {
        let hidden = SsaVarId::from_index(7);
        let staged = SsaVarId::from_index(11);
        let mut call = CallData::<MockTarget>::new(None, Callee::Direct(9), vec![])
            .with_implicit_input(CallInputKey::new(17), hidden);
        call.staged = vec![
            StagedValue {
                offset: 0,
                value: hidden,
            },
            StagedValue {
                offset: 8,
                value: staged,
            },
        ];
        assert!(!call.only_stages(hidden));
        assert!(call.only_stages(staged));
        let mut op = SsaOp::from(call.clone());
        assert_eq!(op.uses(), vec![hidden, hidden, staged]);
        let mut mutable_uses = Vec::new();
        op.visit_operands_mut(|_, value| mutable_uses.push(*value));
        assert_eq!(mutable_uses, vec![hidden, hidden, staged]);
        let replacement = SsaVarId::from_index(19);
        assert_eq!(op.replace_uses(hidden, replacement), 2);
        assert_eq!(op.uses(), vec![replacement, replacement, staged]);
        let redirected = call.redirected(Callee::Direct(10), vec![]);
        assert_eq!(redirected.staged, call.staged);
        assert_eq!(redirected.implicit_inputs, call.implicit_inputs);
    }

    /// **A call defines what it returns beside its result.** NSIS's
    /// `GetIntFromParm` leaves an integer in `eax` and a character in `edx`;
    /// the call defines both, its result and the value under `edx`'s key, and
    /// the return hands the second back under that key. Each is a definition
    /// of the call and a use of the return, renamed, redirected and printed
    /// with it.
    #[test]
    fn a_call_defines_what_it_returns_beside_its_result() {
        let (result, arg, beside, renamed) = (
            SsaVarId::from_index(1),
            SsaVarId::from_index(3),
            SsaVarId::from_index(4),
            SsaVarId::from_index(5),
        );
        let key = CallInputKey::new(2);
        let call = CallData::<MockTarget>::new(Some(result), Callee::Direct(9), vec![arg])
            .with_result(key, beside);
        let mut op = SsaOp::from(call.clone());
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![result, beside]);
        assert_eq!(op.uses(), vec![arg]);
        assert!(op.to_string().starts_with("v1, 2:v4 = call"), "{op}");
        assert!(op.replace_def(beside, renamed));
        assert_eq!(
            op.as_call().map(|call| call.results.as_slice()),
            Some(
                &[CallResult {
                    key,
                    value: renamed
                }][..]
            )
        );
        let redirected = call.redirected(Callee::Direct(10), vec![]);
        assert_eq!(redirected.results, call.results);
        let unread = SsaOp::from(
            CallData::<MockTarget>::new(None, Callee::Direct(9), vec![]).with_result(key, beside),
        );
        assert!(unread.to_string().starts_with("_, 2:v4 = call"), "{unread}");
        let returned = SsaOp::<MockTarget>::returns_results(
            Some(result),
            vec![CallInput { key, value: beside }],
        );
        assert_eq!(returned.uses(), vec![result, beside]);
        assert!(
            returned.to_string().ends_with("[results 2:v4]"),
            "{returned}"
        );
    }

    /// Serialization retains a call's and a return's results, and reads either
    /// without them as returning none beside its value.
    #[cfg(feature = "serde")]
    #[test]
    fn call_results_serde_roundtrip_and_absent_field() {
        let key = CallInputKey::new(2);
        let call = SsaOp::from(
            CallData::<MockTarget>::new(Some(SsaVarId::from_index(1)), Callee::Direct(9), vec![])
                .with_result(key, SsaVarId::from_index(4)),
        );
        let returned = SsaOp::<MockTarget>::returns_results(
            None,
            vec![CallInput {
                key,
                value: SsaVarId::from_index(4),
            }],
        );
        for op in [call, returned] {
            let encoded = serde_json::to_value(&op).expect("serializable op");
            let decoded: SsaOp<MockTarget> = serde_json::from_value(encoded).expect("an op");
            assert_eq!(decoded, op);
        }
        let ordinary = SsaOp::from(CallData::<MockTarget>::new(None, Callee::Direct(9), vec![]));
        let encoded = serde_json::to_string(&ordinary).expect("serializable op");
        assert!(!encoded.contains("results"), "{encoded}");
        let plain = SsaOp::<MockTarget>::returns(None);
        let encoded = serde_json::to_string(&plain).expect("serializable op");
        assert!(!encoded.contains("results"), "{encoded}");
    }

    /// Serialization retains implicit keys and accepts an absent empty-input field.
    #[cfg(feature = "serde")]
    #[test]
    fn call_input_serde_roundtrip_and_legacy_empty_field() {
        let key = CallInputKey::new(u32::MAX);
        let call =
            CallData::<MockTarget>::new(None, Callee::Virtual(9), vec![SsaVarId::from_index(3)])
                .with_implicit_input(key, SsaVarId::from_index(7))
                .with_implicit_input(key, SsaVarId::from_index(11));
        let encoded = serde_json::to_value(&call).expect("serializable mock call");
        assert_eq!(encoded["implicit_inputs"][0]["key"], u32::MAX);
        let decoded: CallData<MockTarget> = serde_json::from_value(encoded).expect("complete call");
        assert_eq!(decoded, call);
        assert_eq!(decoded.args, vec![SsaVarId::from_index(3)]);
        assert_eq!(decoded.receiver(), Some(SsaVarId::from_index(3)));
        let ordinary = CallData::<MockTarget>::new(None, Callee::Direct(9), vec![]);
        let mut legacy = serde_json::to_value(&ordinary).expect("serializable mock call");
        legacy
            .as_object_mut()
            .expect("a call object")
            .remove("implicit_inputs");
        let decoded: CallData<MockTarget> =
            serde_json::from_value(legacy).expect("missing inputs default empty");
        assert_eq!(decoded, ordinary);
    }

    /// Empty-input construction remains usable in a const context.
    #[test]
    fn call_input_empty_construction_preserves_const_api() {
        const CALL: CallData<MockTarget> = CallData::new(None, Callee::Direct(9), Vec::new());
        assert!(CALL.implicit_inputs.is_empty());
        let op = SsaOp::from(CALL);
        assert!(op.uses().is_empty());
    }

    /// **Every call is one shape; its callee tells calls apart.** Each callee
    /// that names a method answers it, only a direct call's certainly runs,
    /// only a virtual call has a receiver object (a constrained call's first
    /// argument is an address), and only an indirect call reads a pointer.
    #[test]
    fn a_callee_says_what_a_call_runs() {
        let (this, arg) = (SsaVarId::from_index(0), SsaVarId::from_index(1));
        let callees = [
            Callee::<MockTarget>::Direct(9),
            Callee::Virtual(9),
            Callee::Constrained {
                method: 9,
                constraint_type: 3,
                instance: true,
            },
            Callee::Constrained {
                method: 9,
                constraint_type: 3,
                instance: false,
            },
            Callee::Indirect {
                fptr: this,
                signature: 0,
            },
        ];
        for callee in callees {
            let call = CallData::new(None, callee.clone(), vec![this, arg]);
            let names = !matches!(callee, Callee::Indirect { .. });
            assert_eq!(call.callee.method().is_some(), names, "{callee:?}");
            assert_eq!(
                call.callee.direct().is_some(),
                matches!(callee, Callee::Direct(_)),
                "{callee:?}"
            );
            assert_eq!(
                call.receiver(),
                matches!(callee, Callee::Virtual(_)).then_some(this),
                "{callee:?}"
            );
            assert_eq!(call.callee.fptr(), (!names).then_some(this), "{callee:?}");
            assert!(call.effects.is_none() && !call.tail);
        }
    }

    /// A call is rewritten in place: its arguments truncated, its method
    /// retargeted, its result renamed, its effects and tail call stated.
    #[test]
    fn a_call_is_rewritten_in_place() {
        let mut op: SsaOp<MockTarget> = CallData::new(
            None,
            Callee::Virtual(9),
            vec![SsaVarId::from_index(0), SsaVarId::from_index(1)],
        )
        .with_effects(Effects::pure())
        .made_tail()
        .into();
        let SsaOp::Call(call) = &mut op else {
            unreachable!("a call")
        };
        call.args.truncate(1);
        if let Some(method) = call.callee.method_mut() {
            *method = 4;
        }
        call.dest = Some(SsaVarId::from_index(3));
        let SsaOp::Call(call) = &op else {
            unreachable!("a call")
        };
        assert_eq!(call.callee, Callee::Virtual(4));
        assert_eq!(
            (call.dest, call.args.len(), call.tail),
            (Some(SsaVarId::from_index(3)), 1, true)
        );
        assert!(call.effects.as_ref().is_some_and(CallEffects::is_pure));
    }

    /// **A redirected call keeps what the caller sees of it.** Its result
    /// and its tail call stay; its stated effects go with the code it no
    /// longer runs.
    #[test]
    fn a_redirected_call_keeps_its_result_and_tail_call() {
        let (dest, arg) = (SsaVarId::from_index(3), SsaVarId::from_index(1));
        for tail in [false, true] {
            let mut call = CallData::<MockTarget>::new(Some(dest), Callee::Virtual(9), vec![arg])
                .with_effects(Effects::pure());
            call.tail = tail;
            let redirected = call.redirected(Callee::Direct(4), Vec::new());
            assert_eq!(
                (redirected.dest, redirected.tail),
                (Some(dest), tail),
                "tail {tail}"
            );
            assert_eq!(redirected.callee, Callee::Direct(4));
            assert!(redirected.args.is_empty() && redirected.effects.is_none());
        }
    }

    #[test]
    fn a_pointee_is_reached_through_its_argument() {
        let access = CallAccess::<MockTarget> {
            location: CallLocation::Pointee {
                arg: 1,
                extent: PointeeExtent::Bytes(16),
            },
            access: AccessMode::Write,
        };
        let args = [SsaVarId::from_index(4), SsaVarId::from_index(5)];
        assert_eq!(
            access.pointer_argument(&args),
            Some(SsaVarId::from_index(5))
        );
        assert_eq!(access.pointer_argument(&args[..1]), None);
    }
}
