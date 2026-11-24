//! One shared intra-procedural address model for the analyses that need to
//! reason about *where* a memory access lands.
//!
//! Address chains reach the SSA in many shapes - a frontend may emit the whole
//! `[base + index*scale + disp]` form as a single [`SsaOp::PtrAdd`], or shred it
//! into a `Shl`/`Mul`/`Add`/`Add` chain, and normalization passes freely rewrite
//! between the two. Analyses that key on the *address value id* are therefore
//! unstable under GVN and LICM. This module is the single home for the decoded
//! form instead:
//!
//! - [`AddressExpr`] - the normalized `base + index*stride_bytes + offset_bits`
//!   form, recovered by [`AddressWalk::normalize`] with one backward-recursive,
//!   [`get_definition`](crate::ir::SsaFunction::get_definition)-based,
//!   depth-bounded walk.
//! - [`AddressBase`] - what an address is measured from: an SSA value, a slot
//!   of the function's own frame whose address it took, or nothing at all, for
//!   an address the program states as a number.
//! - [`AliasKey`] - a memory-cell identity key, derived as the projection of an
//!   [`AddressExpr`] onto its base, index and offset.
//! - [`SsaFunction::const_i64`](crate::ir::function::SsaFunction::const_i64) / [`SsaFunction::const_u64`](crate::ir::function::SsaFunction::const_u64) - the single constant decoder, looking a
//!   value up through its definition and adopting the complete
//!   `True`/`False` mapping.
//!
//! # Absolute addresses
//!
//! An address the walk resolves to a constant has no base value: it is
//! absolute, and the constant is its displacement. That is what gives an image
//! cell one identity however it was materialised - two instructions loading
//! `0x601040` name one cell, and `0x601048` is provably a different one - where
//! naming the address by the instruction that produced the constant made every
//! materialisation its own unrelated base.
//!
//! An absolute address is in one of two spaces, and they are kept apart:
//! [`AddressBase::Symbol`], a symbol [`Target::symbol_address`] resolved, and
//! [`AddressBase::Number`], a bare integer. A host's symbols are measured in
//! its own space - module-relative, for one - while an integer the program
//! computes is in whatever space the program computes in, and nothing here
//! can relate the two. Comparing across them would prove two views of one cell
//! disjoint, which is the unsound direction, so across them everything may
//! alias.
//!
//! The walk never fabricates a relationship: anything it cannot decompose
//! normalizes to its own identity, which is the sound conservative answer.

use std::{cmp::Ordering, collections::BTreeMap};

use crate::{
    analysis::reach::greatest_common_step,
    ir::{
        CmpKind, ConstValue, PhiNode, SsaFunction, SsaOp, SsaVarId, VariableRole,
        ops::AddressedAccess,
    },
    pointer::PointerSize,
    target::Target,
};

/// Maximum backward-walk depth for [`AddressWalk::normalize`].
///
/// Bounds the steps of the walk over definition chains that attach an index.
///
/// The walk follows no phi, so a chain cannot cycle, and every step reads one
/// operand further, so it is as long as the code that built the address. A
/// step that only moves the address - a constant displacement, a
/// value-preserving conversion or copy - is free: a stack pointer is stepped by
/// every push, every reservation and every release before each call, and a
/// budget on those stopped every access past a few calls short of the entry
/// stack pointer, as though it were through an unrelated pointer.
const MAX_ADDRESS_DEPTH: usize = 16;

/// Reduces a folded displacement into the target's pointer width.
///
/// The model accumulates offsets as `i64`, but address arithmetic wraps at the
/// target's pointer width, and the two lowerings of one address do not agree
/// until they are reduced into it: `lea`-style arithmetic sign-extends `-8`,
/// while `add reg, 0xFFFF_FFF8` lifts to an *unsigned* 32-bit constant that
/// [`ConstValue::resolved_i64`] zero-extends to `+4294967288`. On a 32-bit target both
/// name the same cell; left as `i64` they sit 4 GiB apart, and
/// [`IndirectLocation::may_alias`](crate::analysis::memory::IndirectLocation::may_alias)
/// would prove them *disjoint*. That is a false NoAlias, the unsound direction:
/// it lets a stale value survive a store to the cell, and lets dead-store
/// elimination drop a store that was read.
///
/// Reducing both into the pointer width maps them to one canonical `-8`.
/// Displacements are held in bits, so the byte part is wrapped via
/// [`PointerSize::mask_signed`] and any sub-byte member offset rides along
/// unchanged. On 64-bit and wider targets `mask_signed` is the identity, which
/// is correct: `i64` already *is* the address domain there, and a displacement
/// large enough to wrap it cannot be constructed ([`ConstValue::resolved_i64`] rejects a
/// `u64` above `i64::MAX`, and the checked arithmetic bails).
///
/// # Arguments
///
/// * `offset_bits` - The folded displacement in bits, before wrapping.
/// * `ptr_size` - The width the displacement wraps at: the pointer width for an address,
///   an integer's own width for an integer ([`AddressWalk`] chooses).
///
/// # Returns
///
/// The displacement reduced into the pointer width, with any sub-byte member
/// offset preserved.
fn canonical_offset_bits(offset_bits: i64, ptr_size: PointerSize) -> i64 {
    let bytes = offset_bits.div_euclid(8);
    let sub_byte = offset_bits.rem_euclid(8);
    ptr_size
        .mask_signed(bytes)
        .saturating_mul(8)
        .saturating_add(sub_byte)
}

/// What an address is measured from.
///
/// The two absolute bases are address zero of a space: the displacement *is*
/// the address. Two absolute addresses in one space are comparable by
/// displacement alone, which is what no pair of value bases can be - two SSA
/// pointers may hold one address. Anything else may alias: a value can hold
/// any address, and the two absolute spaces are not known to agree (see the
/// module documentation).
///
/// Ordered so a key holding one sorts deterministically: value bases first, by
/// id, then the two absolute spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddressBase {
    /// The run-time contents of an SSA value: the address is that value plus a
    /// displacement.
    Value(SsaVarId),
    /// Zero of the space the host's symbols are measured in: the address was
    /// stated as a symbol [`Target::symbol_address`] resolved.
    Symbol,
    /// Zero of the space the program computes in: the address was stated as a
    /// bare integer.
    Number,
    /// The storage of the function's local slot `n`, as
    /// [`SsaOp::LoadLocalAddr`] takes its address.
    ///
    /// A slot is storage of its own: no other slot's address, no static field
    /// and no image address lands in it. So two addresses off different slots
    /// name disjoint memory, which an address held in an arbitrary value
    /// cannot promise.
    Local(u16),
    /// The storage of the function's argument slot `n`, as
    /// [`SsaOp::LoadArgAddr`] takes its address. Storage of its own, like
    /// [`Self::Local`].
    Argument(u16),
}

impl AddressBase {
    /// Returns the base value, or `None` for an absolute address.
    #[must_use]
    pub const fn value(self) -> Option<SsaVarId> {
        match self {
            Self::Value(value) => Some(value),
            Self::Symbol | Self::Number | Self::Local(_) | Self::Argument(_) => None,
        }
    }

    /// Returns `true` for a slot of the function's own frame
    /// ([`Self::Local`] or [`Self::Argument`]).
    #[must_use]
    pub const fn is_slot(self) -> bool {
        matches!(self, Self::Local(_) | Self::Argument(_))
    }

    /// Returns `true` for either absolute space.
    #[must_use]
    pub const fn is_absolute(self) -> bool {
        matches!(self, Self::Symbol | Self::Number)
    }
}

impl From<SsaVarId> for AddressBase {
    fn from(value: SsaVarId) -> Self {
        Self::Value(value)
    }
}

/// A normalized address expression: `base + index*stride_bytes + offset`, where
/// `offset` is carried in bits.
///
/// Recovered by [`AddressWalk::normalize`]. `root` is the deepest value the walk
/// reached; `base` is what the address is measured from - that value, or an
/// absolute space when the root is a constant, whose value then sits in
/// `offset_bits`. `index`/`stride_bytes` capture a single scaled index term
/// (`i * 4`, `i << 3`) when present; `offset_bits` accumulates the constant
/// byte displacements, stored in bits so sub-byte member offsets stay
/// representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressExpr {
    /// What the address is measured from.
    pub base: AddressBase,
    /// The deepest value the walk reached: the base value itself, or for an
    /// absolute address the constant that materialised it.
    ///
    /// Kept for a consumer that needs to know *how* the address was stated -
    /// whether its constant was a symbol, say - and never part of the cell's
    /// identity: two materialisations of one address have different roots and
    /// one [`AliasKey`].
    pub root: SsaVarId,
    /// Constant displacement from [`Self::root`], in bits.
    ///
    /// Equal to [`Self::offset_bits`] whenever the base is a value. For an
    /// absolute address it is what the walk added *to* the constant, which is
    /// the member offset a consumer reasoning about the object the constant
    /// names - a typed global record, say - wants.
    pub root_offset_bits: i64,
    /// Scaled index term, when the address adds one.
    pub index: Option<SsaVarId>,
    /// Stride applied to [`Self::index`], in bytes.
    pub stride_bytes: u64,
    /// Constant displacement from the base, in bits.
    pub offset_bits: i64,
}

impl AddressExpr {
    /// Returns the trivial address `value + 0` - `value` is its own base with
    /// no index and no offset.
    #[must_use]
    fn identity(value: SsaVarId) -> Self {
        Self {
            base: AddressBase::Value(value),
            root: value,
            root_offset_bits: 0,
            index: None,
            stride_bytes: 0,
            offset_bits: 0,
        }
    }

    /// Returns the address of a slot's own storage, `value` taking it.
    #[must_use]
    const fn at_slot(base: AddressBase, value: SsaVarId) -> Self {
        Self {
            base,
            root: value,
            root_offset_bits: 0,
            index: None,
            stride_bytes: 0,
            offset_bits: 0,
        }
    }

    /// The same address measured from where a loop's walk through an array
    /// starts, where its base is a phi that steps itself by constants from one
    /// start ([`AddressWalk::walk_origin`]): measured from the start instead, the phi
    /// standing for an index whose stride is the step every move of the walk
    /// is a whole number of.
    ///
    /// `p = phi(s + 12, p + 24)` read at `p + 8` reads `s + 20 + k * 24` for
    /// the trips `k` the loop has made, the place 20 of an array of 24-byte
    /// elements at `s`: what reads the array through its start and through
    /// the walk then reads one array. An address measured from no walk, or
    /// already indexed, is itself.
    ///
    /// # Arguments
    ///
    /// * `walk` - The addresses of the function whose definitions the walk
    ///   consults.
    #[must_use]
    pub fn measured_from_walk<T: Target>(self, walk: &AddressWalk<'_, T>) -> Self {
        let Some((walker, walk)) = self
            .base
            .value()
            .filter(|_| self.index.is_none())
            .and_then(|base| walk.ir.find_phi_defining(base))
            .and_then(|(_, phi)| Some((phi.result(), Walk::of(walk, phi)?)))
        else {
            return self;
        };
        let (Some(step), Some(root_offset_bits), Some(offset_bits)) = (
            walk.step.filter(|step| *step > 0),
            walk.origin
                .root_offset_bits
                .checked_add(self.root_offset_bits),
            walk.origin.offset_bits.checked_add(self.offset_bits),
        ) else {
            return self;
        };
        Self {
            base: walk.origin.base,
            root: walk.origin.root,
            root_offset_bits,
            index: Some(walker),
            stride_bytes: step,
            offset_bits,
        }
    }

    /// Returns `true` when the walk decomposed nothing: `value` is its own base,
    /// with no index and no offset.
    #[must_use]
    fn is_identity_of(&self, value: SsaVarId) -> bool {
        self.base == AddressBase::Value(value) && self.index.is_none() && self.offset_bits == 0
    }

    /// Decides `self <cmp> other` when the two are measured from one base,
    /// neither indexed, by their displacements: `&a[4] == &a[0]` is false,
    /// and two `lea`s of one symbol are equal. `None` when the relation turns
    /// on where the base is.
    ///
    /// Equality is exact from any base, since a base moves both alike, and so
    /// is every relation of two equal displacements. Order is the unsigned
    /// order of the displacements from a base no displacement wraps around: a
    /// symbol's space or a slot of the frame. From a value, which may hold any
    /// address, or a bare number, which constant folding orders, an order of
    /// two different displacements is not decided, nor is a signed one, which
    /// turns on where the addresses fall in the space.
    #[must_use]
    pub fn compares(&self, other: &Self, cmp: CmpKind, unsigned: bool) -> Option<bool> {
        if self.base != other.base || self.index.is_some() || other.index.is_some() {
            return None;
        }
        let order = self.offset_bits.cmp(&other.offset_bits);
        let ordered = order == Ordering::Equal
            || (unsigned
                && matches!(
                    self.base,
                    AddressBase::Symbol | AddressBase::Local(_) | AddressBase::Argument(_)
                ));
        match cmp {
            CmpKind::Eq => Some(order == Ordering::Equal),
            CmpKind::Ne => Some(order != Ordering::Equal),
            CmpKind::Lt => ordered.then_some(order == Ordering::Less),
            CmpKind::Le => ordered.then_some(order != Ordering::Greater),
            CmpKind::Gt => ordered.then_some(order == Ordering::Greater),
            CmpKind::Ge => ordered.then_some(order != Ordering::Less),
        }
    }

    /// The byte address this expression names, given the address its
    /// [`Self::root`] holds.
    ///
    /// A host that knows which values hold which constant addresses reads an
    /// access's address as that root's address moved by
    /// [`Self::root_offset_bits`]. `None` when the move is not a whole number
    /// of bytes, or leaves the address space.
    ///
    /// # Arguments
    ///
    /// * `root_address` - The address the root value holds.
    #[must_use]
    pub fn byte_address_from_root(&self, root_address: u64) -> Option<u64> {
        if self.root_offset_bits % 8 != 0 {
            return None;
        }
        root_address.checked_add_signed(self.root_offset_bits / 8)
    }

    /// Whether `other` names the same address: one base, index and stride,
    /// and one displacement. How each was materialised - its
    /// [`Self::root`] - does not matter.
    #[must_use]
    pub fn same_place(&self, other: &Self) -> bool {
        self.base == other.base
            && self.index == other.index
            && self.stride_bytes == other.stride_bytes
            && self.offset_bits == other.offset_bits
    }

    /// The byte offset from the stack pointer at entry this expression names,
    /// when it names one directly: measured from `entry_stack_pointer`, with no
    /// index and a whole number of bytes.
    ///
    /// # Arguments
    ///
    /// * `entry_stack_pointer` - The value the stack pointer holds at entry.
    #[must_use]
    pub fn frame_offset_bytes(&self, entry_stack_pointer: SsaVarId) -> Option<i64> {
        if self.base != AddressBase::Value(entry_stack_pointer)
            || self.index.is_some()
            || self.offset_bits % 8 != 0
        {
            return None;
        }
        Some(self.offset_bits / 8)
    }
}

/// One use of an address the walk places, as [`AddressWalk::for_each_absolute_use`] and
/// [`AddressWalk::for_each_frame_use`] report it: `A` is where the address lies, an image
/// address or a byte offset from the stack pointer at entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressUse<A> {
    /// A memory access through the address.
    Access {
        /// Where the access lands.
        address: A,
        /// The stride of the scaled index added, when one is: the access
        /// names an element of what lies at `address`, not the object itself.
        stride: Option<u64>,
        /// What the access does.
        access: AddressedAccess,
    },
    /// The address used without accessing through it: handed to a call,
    /// stored, given to a block copy, merged at a phi. The object is
    /// referenced; nothing about its width or type follows.
    Taken {
        /// Where the taken address lies.
        address: A,
        /// The stride of the scaled index added, when one is.
        stride: Option<u64>,
    },
    /// The address compared with another: `ceq`, `clt`, `cgt` or a compare
    /// and branch reads it and lets nothing reach memory through it. A loop
    /// walking an array stops at the address one past its last element, which
    /// no access reads.
    Compared {
        /// Where the compared address lies.
        address: A,
        /// The stride of the scaled index added, when one is.
        stride: Option<u64>,
        /// The value it is compared with: the pointer a loop walks to it, for
        /// a bound ([`AddressWalk::walk_origin`]).
        against: SsaVarId,
    },
}

/// One use of an address a function states as a constant, as
/// [`AddressWalk::for_each_absolute_use`] reports it: at the image address it names.
pub type AbsoluteUse = AddressUse<u64>;

/// One use of an address of the function's own frame, as
/// [`AddressWalk::for_each_frame_use`] reports it: at its byte offset from the stack
/// pointer the function was entered with.
pub type FrameUse = AddressUse<i64>;

/// Visits every use `ir` makes of an address `named` places, with where it
/// lies and the stride of the scaled index added, when one is.
///
/// Each memory access through such an address is reported once, as is each
/// comparison of one, and each other use: an operand of an operation that
/// neither accesses through it nor compares it, and every phi operand. An
/// operation that only moves an address along ([`SsaOp::steps_an_address`])
/// takes nothing; the use it feeds is read through it.
fn for_each_address_use<T: Target, A>(
    ir: &SsaFunction<T>,
    named: impl Fn(SsaVarId) -> Option<(A, Option<u64>)>,
    mut visit: impl FnMut(AddressUse<A>),
) {
    for block in ir.blocks() {
        for instruction in block.instructions() {
            let op = instruction.op();
            let mut accessed: Vec<SsaVarId> = Vec::new();
            op.for_each_addressed_access(|access| {
                accessed.push(access.addr);
                if let Some((address, stride)) = named(access.addr) {
                    visit(AddressUse::Access {
                        address,
                        stride,
                        access,
                    });
                }
            });
            if op.steps_an_address() {
                continue;
            }
            let compared = match op {
                SsaOp::Ceq { left, right, .. }
                | SsaOp::Clt { left, right, .. }
                | SsaOp::Cgt { left, right, .. }
                | SsaOp::BranchCmp { left, right, .. } => Some((*left, *right)),
                _ => None,
            };
            for used in op.uses() {
                if accessed.contains(&used) {
                    continue;
                }
                if let Some((address, stride)) = named(used) {
                    visit(match compared {
                        Some((left, right)) => AddressUse::Compared {
                            address,
                            stride,
                            against: if used == left { right } else { left },
                        },
                        None => AddressUse::Taken { address, stride },
                    });
                }
            }
        }
        for phi in block.phi_nodes() {
            for operand in phi.operands() {
                if let Some((address, stride)) = named(operand.value()) {
                    visit(AddressUse::Taken { address, stride });
                }
            }
        }
    }
}

/// The alignment an `And` with `mask` imposes on an address, in bytes.
///
/// A prologue aligns the stack by clearing the low bits of its pointer, which
/// is an `And` with a mask whose cleared bits are exactly those low bits:
/// `-16` at either pointer width, and equally the unsigned `0xffff_fff0` a
/// 32-bit operand may carry. Any other mask moves the address somewhere this
/// cannot name - `and eax, 0xff` is a truncation, not an alignment - and
/// answers `None`.
///
/// # Arguments
///
/// * `mask` - The constant operand of the `And`.
/// * `ptr_size` - The target pointer width, which the mask is read at.
#[must_use]
pub fn alignment_mask(mask: i64, ptr_size: PointerSize) -> Option<u64> {
    let width = ptr_size.mask_unsigned(u64::MAX);
    #[expect(
        clippy::cast_sign_loss,
        reason = "the mask is read as the bit pattern it is"
    )]
    let kept = (mask as u64) & width;
    let cleared = !kept & width;
    let alignment = cleared.checked_add(1)?;
    (cleared != width && alignment.is_power_of_two() && alignment > 1).then_some(alignment)
}

/// A phi a loop steps from one start ([`AddressWalk::walk_origin`]).
struct Walk {
    /// The address every operand that is no step of its own normalizes to.
    origin: AddressExpr,
    /// The greatest step, in bytes, every move of the operands measured from
    /// the phi itself is a whole number of: `0` where none moves it, `None`
    /// where one moves it by no whole number of bytes.
    step: Option<u64>,
}

impl Walk {
    /// The walk `phi` makes, or `None` where its starts disagree or one is
    /// indexed.
    fn of<T: Target>(walk: &AddressWalk<'_, T>, phi: &PhiNode) -> Option<Self> {
        let walked = phi.result();
        let mut origin: Option<AddressExpr> = None;
        let mut step = Some(0);
        for operand in phi.operands() {
            let start = walk.normalize(operand.value());
            if start.base == AddressBase::Value(walked) && start.index.is_none() {
                let moved = start.offset_bits.unsigned_abs();
                step = step
                    .zip(
                        moved
                            .is_multiple_of(8)
                            .then(|| moved.checked_div(8))
                            .flatten(),
                    )
                    .map(|(step, moved)| greatest_common_step(step, moved));
                continue;
            }
            if start.index.is_some() || origin.is_some_and(|known| !known.same_place(&start)) {
                return None;
            }
            origin = Some(start);
        }
        Some(Self {
            origin: origin?,
            step,
        })
    }
}

/// The addresses of one function at one pointer width: where each value
/// points, measured from the value it is built from ([`Self::normalize`]),
/// and the questions answered from that.
///
/// Walks address definitions: displacement chains iteratively, index
/// recursion bounded by depth. Every offset is canonicalised into the pointer
/// width.
#[derive(Debug, Clone, Copy)]
pub struct AddressWalk<'a, T: Target> {
    /// The body supplying definitions, variable roles and the cycle bound.
    ir: &'a SsaFunction<T>,
    /// The target pointer width offsets wrap at.
    ptr_size: PointerSize,
}

impl<'a, T: Target> AddressWalk<'a, T> {
    /// The width a displacement of `value` wraps at: an address, and an integer of no fixed
    /// width, at the pointer width; any other integer at its own, which its arithmetic
    /// wraps at. A 64-bit `x + 2^32` computed where pointers are 32 bits moves `x` by
    /// `2^32`, which wrapped at the pointer width would read as no move at all.
    fn wrap_width(&self, value: SsaVarId) -> PointerSize {
        self.ir
            .variable(value)
            .map(|variable| variable.var_type())
            .filter(|ty| T::is_integer(ty) && !T::is_pointer(ty))
            .and_then(T::bit_width)
            .and_then(|bits| PointerSize::try_from_bytes(bits / 8))
            .unwrap_or(self.ptr_size)
    }

    /// Opens the addresses of `ir` at `ptr_size`.
    #[must_use]
    pub const fn new(ir: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self { ir, ptr_size }
    }

    /// Normalizes `value` into an [`AddressExpr`] by walking its definition chain.
    ///
    /// Copy / `Conv` / `Bitcast` pass through to their source; [`SsaOp::PtrAdd`] is
    /// read directly onto the recursively-normalized base; `Add` / `Sub` fold a
    /// constant byte displacement into [`AddressExpr::offset_bits`]; an `Add` of a
    /// scaled index (`Mul`/`Shl` by a constant) records the [`AddressExpr::index`].
    /// A constant ends the walk at an absolute base with the constant as the
    /// displacement: [`AddressBase::Symbol`] for a symbol [`Target::symbol_address`]
    /// resolves, [`AddressBase::Number`] for any other constant with an integer
    /// value. Anything else - or a depth-exhausted / overflowing chain -
    /// terminates the walk with that value as the [`AddressExpr::base`]. The walk
    /// never fabricates a relationship: an undecodable value normalizes to its own
    /// identity.
    ///
    /// # Arguments
    ///
    /// * `value` - The address value to decode.
    ///
    /// # Returns
    ///
    /// The normalized `base + index*stride_bytes + offset_bits` form.
    #[must_use]
    pub fn normalize(&self, value: SsaVarId) -> AddressExpr {
        self.inner(value, MAX_ADDRESS_DEPTH)
    }

    /// The address a pointer a loop walks starts from: [`AddressWalk::normalize`] of
    /// `value`, and where that is measured from a phi, the one address every
    /// operand of the phi that is no step of the walk normalizes to.
    ///
    /// A loop filling an array keeps `p = phi(&a[0], p + 4)` and compares `p + 4`
    /// with its bound: each step is measured from the phi itself, and the walk
    /// starts where the phi's other operand points, `&a[0]`. That is the object
    /// the walk stays in, and the one its bound ends. `None` where the phi's
    /// starts disagree or one is indexed, and for a value measured from no phi
    /// that is not an address of its own.
    ///
    /// # Arguments
    ///
    /// * `value` - The walked pointer.
    #[must_use]
    pub fn walk_origin(&self, value: SsaVarId) -> Option<AddressExpr> {
        let address = self.normalize(value);
        match address
            .base
            .value()
            .and_then(|base| self.ir.find_phi_defining(base))
        {
            Some((_, phi)) => Walk::of(self, phi).map(|walk| walk.origin),
            None => Some(address),
        }
    }

    /// Decides `left <cmp> right` over two values of `ir` by the addresses they
    /// are ([`AddressExpr::compares`]): two `lea`s of one symbol are equal, and
    /// `__TMC_END__ == __TMC_LIST__` holds in an image that lists no clones.
    ///
    /// `None` for a float or a value of unknown type, whose comparison with
    /// itself is false on NaN, and wherever the displacements decide nothing.
    ///
    /// # Arguments
    ///
    /// * `left` - The left operand of the comparison.
    /// * `right` - The right operand.
    /// * `cmp` - The relation compared.
    /// * `unsigned` - Whether an order is the unsigned one.
    #[must_use]
    pub fn compare(
        &self,
        left: SsaVarId,
        right: SsaVarId,
        cmp: CmpKind,
        unsigned: bool,
    ) -> Option<bool> {
        for value in [left, right] {
            let ty = self.ir.variable(value)?.var_type();
            if T::is_floating(ty) || T::is_unknown(ty) {
                return None;
            }
        }
        let left = self.normalize(left);
        let right = self.normalize(right);
        left.compares(&right, cmp, unsigned)
    }

    /// The value the function's frame is measured from where its prologue aligns
    /// the stack, with the alignment it imposes.
    ///
    /// An aligned frame has an anchor of its own: how far the aligned pointer
    /// sits below the pointer the function was entered with is not a number -
    /// it is the entry pointer rounded down - so no access below the alignment
    /// has an exact offset from entry, while every one of them has an exact
    /// offset from the aligned value. That value is this anchor, and a consumer
    /// placing the frame's objects measures from it instead of from entry.
    ///
    /// `None` where the function aligns nothing, and equally where more than one
    /// alignment applies to an address of the frame, which is more than one
    /// anchor and names no single space.
    ///
    /// # Arguments
    ///
    /// * `entry` - The stack pointer the function was entered with.
    #[must_use]
    pub fn frame_anchor(&self, entry: SsaVarId) -> Option<(SsaVarId, u64)> {
        let mut found: Option<(SsaVarId, u64)> = None;
        for block in self.ir.blocks() {
            for instruction in block.instructions() {
                let SsaOp::And {
                    dest, left, right, ..
                } = instruction.op()
                else {
                    continue;
                };
                let aligned = [(*left, *right), (*right, *left)]
                    .into_iter()
                    .filter(|(value, _)| self.normalize(*value).frame_offset_bytes(entry).is_some())
                    .filter_map(|(_, mask)| alignment_mask(self.ir.const_i64(mask)?, self.ptr_size))
                    .next();
                if let Some(alignment) = aligned {
                    if found.is_some() {
                        return None;
                    }
                    found = Some((*dest, alignment));
                }
            }
        }
        found
    }

    /// Visits every use `ir` makes of an address one of `constants` states.
    ///
    /// An address is a constant one when [`AddressWalk::normalize`] reaches a root
    /// `constants` holds an address for, moved by a whole number of bytes. Each
    /// use is reported once, as the [`AddressUse`] it is: an access through
    /// the address, a comparison of it, or any other use.
    ///
    /// # Arguments
    ///
    /// * `constants` - The values known to hold constant addresses, and the
    ///   address each holds.
    /// * `visit` - Called for each use, in block and instruction order.
    pub fn for_each_absolute_use(
        &self,
        constants: &BTreeMap<SsaVarId, u64>,
        visit: impl FnMut(AbsoluteUse),
    ) {
        if constants.is_empty() {
            return;
        }
        for_each_address_use(
            self.ir,
            |value| {
                let expr = self.normalize(value);
                let address = expr.byte_address_from_root(*constants.get(&expr.root)?)?;
                Some((address, expr.index.map(|_| expr.stride_bytes)))
            },
            visit,
        );
    }

    /// Visits every use `ir` makes of an address of its own frame: one
    /// [`AddressWalk::normalize`] measures from `entry_stack_pointer`, the value the
    /// stack pointer holds at entry, by a whole number of bytes, reported at that
    /// byte offset. Each use is reported once, as the [`AddressUse`] it is: an
    /// access through the address, a comparison of it, or any other use.
    ///
    /// # Arguments
    ///
    /// * `entry_stack_pointer` - The value the stack pointer holds at entry.
    /// * `visit` - Called for each use, in block and instruction order.
    pub fn for_each_frame_use(&self, entry_stack_pointer: SsaVarId, visit: impl FnMut(FrameUse)) {
        for_each_address_use(
            self.ir,
            |value| {
                let expr = self.normalize(value);
                if expr.base != AddressBase::Value(entry_stack_pointer) || expr.offset_bits % 8 != 0
                {
                    return None;
                }
                Some((expr.offset_bits / 8, expr.index.map(|_| expr.stride_bytes)))
            },
            visit,
        );
    }

    /// Computes alias keys for the address chains in `ir`.
    ///
    /// Each address-producing value (`Const`/`Copy`/`Conv`/`Bitcast`/`Add`/`Sub`)
    /// is normalized via [`AddressWalk::normalize`]; a value that reduces past its own
    /// trivial identity gets a key. A constant always does when it has a number:
    /// it is an absolute address. Values without a simplified address fall back
    /// to their own identity at lookup time (see [`alias_key_for_value`]).
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// A map from address-value id to its [`AliasKey`], holding only values that
    /// simplified past their own trivial identity.
    #[must_use]
    pub fn alias_keys(&self) -> BTreeMap<u32, AliasKey> {
        let mut keys = BTreeMap::new();
        for block in self.ir.blocks() {
            for instruction in block.instructions() {
                let dest = match instruction.op() {
                    SsaOp::Const { dest, .. }
                    | SsaOp::Copy { dest, .. }
                    | SsaOp::IntConv { dest, .. }
                    | SsaOp::IntToPtr { dest, .. }
                    | SsaOp::PtrToInt { dest, .. }
                    | SsaOp::Bitcast { dest, .. }
                    | SsaOp::Add { dest, .. }
                    | SsaOp::Sub { dest, .. } => *dest,
                    _ => continue,
                };
                let address = self.normalize(dest);
                // A value that did not simplify past its own identity (no index, its
                // own base, zero offset) keeps the lookup fallback - no recorded key.
                // A `base + index*stride (+ offset)` chain DOES get a key, so
                // array-element cells share an identity.
                if address.is_identity_of(dest) {
                    continue;
                }
                keys.insert(
                    dest.as_u32(),
                    AliasKey {
                        base: address.base,
                        index: address.index.map(SsaVarId::as_u32),
                        stride_bytes: address.stride_bytes,
                        offset_bits: address.offset_bits,
                    },
                );
            }
        }
        keys
    }
}

impl<'a, T: Target> AddressWalk<'a, T> {
    /// The values `value` is computed from by moves of the address model alone
    /// (copies, conversions that keep the width, and constant displacements,
    /// [`AddressWalk::normalize`]'s steps) that name the place `value` does, nearest
    /// first: `(sp - 16) + 16` through its integer conversions is the `sp` it
    /// started from, and a stack pointer moved down for a call's stack arguments
    /// and up past them is the one it was before them, however deep in the frame
    /// that one lies.
    ///
    /// # Arguments
    ///
    /// * `value` - The value whose ancestors are asked for.
    #[must_use]
    pub fn same_place_ancestors(&self, value: SsaVarId) -> Vec<SsaVarId> {
        let mut found = Vec::new();
        let mut current = value;
        // How far `value` lies past `current`, in bits.
        let mut past_bits = 0i64;
        let mut laps = 0usize;
        loop {
            laps = laps.saturating_add(1);
            if laps > self.ir.var_id_bound() {
                return found;
            }
            let step = match self.ir.get_definition(current) {
                Some(SsaOp::Copy { src, .. } | SsaOp::Bitcast { operand: src, .. }) => {
                    Some((*src, 0))
                }
                Some(
                    SsaOp::IntConv { operand: src, .. }
                    | SsaOp::IntToPtr { operand: src, .. }
                    | SsaOp::PtrToInt { operand: src, .. },
                ) if self.keeps_width(current, *src) => Some((*src, 0)),
                Some(SsaOp::PtrAdd {
                    base,
                    index: None,
                    offset,
                    ..
                }) => Some((*base, *offset)),
                Some(SsaOp::Add { left, right, .. }) => self
                    .displacement(*right)
                    .map(|offset| (*left, offset))
                    .or_else(|| self.displacement(*left).map(|offset| (*right, offset))),
                Some(SsaOp::Sub { left, right, .. }) => self
                    .displacement(*right)
                    .and_then(i64::checked_neg)
                    .map(|offset| (*left, offset)),
                _ => None,
            };
            let Some((from, offset)) = step else {
                return found;
            };
            let Some(bits) = offset
                .checked_mul(8)
                .and_then(|bits| past_bits.checked_add(bits))
            else {
                return found;
            };
            past_bits = canonical_offset_bits(bits, self.wrap_width(current));
            current = from;
            // A copy is the value it copies, which the walk reaches next.
            let copies = matches!(
                self.ir.get_definition(current),
                Some(SsaOp::Copy { .. } | SsaOp::Bitcast { .. })
            );
            if past_bits == 0 && current != value && !copies {
                found.push(current);
            }
        }
    }

    /// Whether `converted` is as wide as `operand`, the value it converts,
    /// where both widths are known: a conversion between them only renames
    /// the bits.
    fn keeps_width(&self, converted: SsaVarId, operand: SsaVarId) -> bool {
        let width = |value: SsaVarId| self.ir.bit_width(value);
        match (width(converted), width(operand)) {
            (Some(converted), Some(operand)) => converted == operand,
            _ => true,
        }
    }

    /// Reads a numeric displacement, excluding symbols that denote address bases.
    fn displacement(&self, value: SsaVarId) -> Option<i64> {
        match self.ir.get_definition(value) {
            Some(SsaOp::Const { value, .. }) if !matches!(value, ConstValue::Symbol(_)) => {
                value.resolved_i64()
            }
            _ => None,
        }
    }

    /// Returns `Some((index, stride_bytes))` when `value` is a constant-scaled
    /// index - a multiply or a left-shift by a constant - and `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `value` - The candidate scaled-index value.
    ///
    /// # Returns
    ///
    /// `Some((index, stride_bytes))` for a `Mul`/`Shl` by a constant, where the
    /// stride is the multiplier or `1 << amount`; `None` otherwise.
    #[must_use]
    pub fn scaled_index(&self, value: SsaVarId) -> Option<(SsaVarId, u64)> {
        self.linear(value, Self::LINEAR_DEPTH)
            .filter(|(_, factor)| *factor > 1)
    }

    /// How deep [`Self::linear`] follows a value's steps.
    const LINEAR_DEPTH: usize = 8;

    /// `value` as one variable times a constant: through a multiply or a
    /// shift by a constant, and a sum or difference of two such terms of the
    /// same variable (`(i << 1) + i` is `i * 3`). Any other value is itself
    /// times one. `None` where a difference leaves no positive factor or a
    /// factor overflows.
    fn linear(&self, value: SsaVarId, depth: usize) -> Option<(SsaVarId, u64)> {
        let Some(depth) = depth.checked_sub(1) else {
            return Some((value, 1));
        };
        let scaled = |term: SsaVarId, factor: u64| {
            let (variable, inner) = self.linear(term, depth)?;
            Some((variable, inner.checked_mul(factor)?))
        };
        match self.ir.get_definition(value) {
            Some(SsaOp::Mul { left, right, .. }) => {
                match (
                    self.ir
                        .const_i64(*right)
                        .and_then(|v| u64::try_from(v).ok()),
                    self.ir.const_i64(*left).and_then(|v| u64::try_from(v).ok()),
                ) {
                    (Some(factor), _) => scaled(*left, factor),
                    (None, Some(factor)) => scaled(*right, factor),
                    (None, None) => Some((value, 1)),
                }
            }
            Some(SsaOp::Shl {
                value: shifted,
                amount,
                ..
            }) => match self
                .ir
                .const_i64(*amount)
                .and_then(|v| u32::try_from(v).ok())
            {
                Some(amount) => scaled(*shifted, 1_u64.checked_shl(amount)?),
                None => Some((value, 1)),
            },
            Some(SsaOp::Add { left, right, .. } | SsaOp::Sub { left, right, .. }) => {
                let (Some((one, first)), Some((other, second))) =
                    (self.linear(*left, depth), self.linear(*right, depth))
                else {
                    return Some((value, 1));
                };
                if one != other || self.ir.const_i64(one).is_some() {
                    return Some((value, 1));
                }
                let factor = if matches!(self.ir.get_definition(value), Some(SsaOp::Sub { .. })) {
                    first.checked_sub(second).filter(|factor| *factor > 0)?
                } else {
                    first.checked_add(second)?
                };
                Some((one, factor))
            }
            _ => Some((value, 1)),
        }
    }

    /// Walks displacement chains iteratively and bounds index recursion by depth.
    fn inner(&self, value: SsaVarId, depth: usize) -> AddressExpr {
        if depth == 0 {
            return AddressExpr::identity(value);
        }
        let mut steps = Vec::new();
        let mut current = value;
        let mut laps = 0usize;
        let root = loop {
            laps = laps.saturating_add(1);
            if laps > self.ir.var_id_bound() {
                return AddressExpr::identity(value);
            }
            let Some(op) = self.ir.get_definition(current) else {
                break AddressExpr::identity(current);
            };
            match op {
                SsaOp::Const {
                    value: constant, ..
                } => {
                    let space = if matches!(constant, ConstValue::Symbol(_)) {
                        AddressBase::Symbol
                    } else {
                        AddressBase::Number
                    };
                    break match constant
                        .resolved_i64()
                        .and_then(|address| address.checked_mul(8))
                    {
                        Some(bits) => AddressExpr {
                            base: space,
                            root: current,
                            root_offset_bits: 0,
                            index: None,
                            stride_bytes: 0,
                            offset_bits: canonical_offset_bits(bits, self.wrap_width(current)),
                        },
                        None => AddressExpr::identity(current),
                    };
                }
                SsaOp::LoadLocalAddr { local_index, .. } => {
                    break AddressExpr::at_slot(AddressBase::Local(*local_index), current);
                }
                SsaOp::LoadArgAddr { arg_index, .. } => {
                    break AddressExpr::at_slot(AddressBase::Argument(*arg_index), current);
                }
                SsaOp::Copy { src, .. } | SsaOp::Bitcast { operand: src, .. } => current = *src,
                // A conversion that keeps the width only renames the bits; one
                // that cuts or extends them makes another number, which no
                // term of the address says how to undo: it is a value of its
                // own.
                SsaOp::IntConv { operand: src, .. }
                | SsaOp::IntToPtr { operand: src, .. }
                | SsaOp::PtrToInt { operand: src, .. } => {
                    if !self.keeps_width(current, *src) {
                        break AddressExpr::identity(current);
                    }
                    current = *src;
                }
                SsaOp::PtrAdd {
                    base,
                    index: None,
                    offset,
                    ..
                } => {
                    let Some(bits) = offset.checked_mul(8) else {
                        break AddressExpr::identity(current);
                    };
                    steps.push((current, bits));
                    current = *base;
                }
                SsaOp::PtrAdd {
                    base,
                    index: Some(index),
                    stride,
                    offset,
                    ..
                } => {
                    break self.indexed(current, (*base, *index, *stride, *offset), depth);
                }
                SsaOp::Add { left, right, .. } => {
                    if let Some(offset) = self.displacement(*right).and_then(|v| v.checked_mul(8)) {
                        steps.push((current, offset));
                        current = *left;
                    } else if let Some(offset) =
                        self.displacement(*left).and_then(|v| v.checked_mul(8))
                    {
                        steps.push((current, offset));
                        current = *right;
                    } else {
                        break self.add(current, *left, *right, depth);
                    }
                }
                SsaOp::Sub { left, right, .. } => {
                    let Some(offset) = self
                        .displacement(*right)
                        .and_then(|v| v.checked_mul(8))
                        .and_then(i64::checked_neg)
                    else {
                        break AddressExpr::identity(current);
                    };
                    steps.push((current, offset));
                    current = *left;
                }
                _ => break AddressExpr::identity(current),
            }
        };
        let mut result = root;
        for (at, offset) in steps.into_iter().rev() {
            result = fold_offset(result, offset, at, self.wrap_width(at));
        }
        result
    }

    /// Attaches one scaled index after normalizing its base and displacement.
    fn indexed(
        &self,
        value: SsaVarId,
        parts: (SsaVarId, SsaVarId, u64, i64),
        depth: usize,
    ) -> AddressExpr {
        let (base, index, stride, offset) = parts;
        let base = self.inner(base, depth.saturating_sub(1));
        let Some(bits) = offset.checked_mul(8) else {
            return AddressExpr::identity(value);
        };
        let mut result = fold_offset(base, bits, value, self.wrap_width(value));
        if result.base == AddressBase::Value(value) {
            return result;
        }
        if result.index.is_some() {
            return AddressExpr::identity(value);
        }
        let (index, scale) = self.scaled_index(index).unwrap_or((index, 1));
        let Some(stride) = stride.checked_mul(scale) else {
            return AddressExpr::identity(value);
        };
        result.index = Some(index);
        result.stride_bytes = stride;
        result
    }

    /// Decodes scaled indices or a named region indexed by an unscaled byte offset.
    fn add(&self, value: SsaVarId, left: SsaVarId, right: SsaVarId, depth: usize) -> AddressExpr {
        for (base, scaled) in [(left, right), (right, left)] {
            if let Some((index, stride)) = self.scaled_index(scaled) {
                let mut result = self.inner(base, depth.saturating_sub(1));
                if result.index.is_none() {
                    result.index = Some(index);
                    result.stride_bytes = stride;
                    return result;
                }
            }
        }
        let ir = self.ir;
        let region = |expr: &AddressExpr| match expr.base {
            AddressBase::Value(base) => ir
                .variable(base)
                .is_some_and(|variable| variable.role() == Some(VariableRole::EntryStackPointer)),
            AddressBase::Symbol
            | AddressBase::Number
            | AddressBase::Local(_)
            | AddressBase::Argument(_) => true,
        };
        for (base, other) in [(left, right), (right, left)] {
            let result = self.inner(base, depth.saturating_sub(1));
            if result.index.is_some() || !region(&result) {
                continue;
            }
            let index = self.inner(other, depth.saturating_sub(1));
            let AddressBase::Value(stepped) = index.base else {
                continue;
            };
            if index.index.is_some() || region(&index) {
                continue;
            }
            let indexed = AddressExpr {
                index: Some(stepped),
                stride_bytes: 1,
                ..result
            };
            return if index.offset_bits == 0 {
                indexed
            } else {
                fold_offset(indexed, index.offset_bits, value, self.wrap_width(value))
            };
        }
        AddressExpr::identity(value)
    }
}

/// Adds a constant byte `offset` to an address expression's accumulated
/// offset, tagging the result with the originating `value`.
///
/// # Arguments
///
/// * `base` - The address expression to extend.
/// * `offset` - The displacement to add, in bits.
/// * `value` - The originating value, used as the identity fallback on overflow.
/// * `width` - The width the summed offset is canonicalised into: the pointer width for an
///   address, the integer's own width for an integer.
///
/// # Returns
///
/// `base` with the offset folded in and canonicalised, or `value`'s identity
/// when the addition overflows.
fn fold_offset(
    mut base: AddressExpr,
    offset: i64,
    value: SsaVarId,
    width: PointerSize,
) -> AddressExpr {
    match (
        base.offset_bits.checked_add(offset),
        base.root_offset_bits.checked_add(offset),
    ) {
        (Some(folded), Some(from_root)) => {
            // Canonicalise into the value's width so the sign-extended and
            // zero-extended lowerings of one displacement agree.
            base.offset_bits = canonical_offset_bits(folded, width);
            base.root_offset_bits = canonical_offset_bits(from_root, width);
            base
        }
        _ => AddressExpr::identity(value),
    }
}

/// Rewrites a [`SsaOp::PtrAdd`] whose scaled index is a known constant into the
/// plain `base + offset` it computes.
///
/// `base + c*stride + offset` *is* `base + (offset + c*stride)`: the index term
/// names no value the address depends on at run time, only more displacement.
/// Keeping it as an index leaves the canonical address form ambiguous - two
/// spellings of one cell - and hands every consumer a constant in the one
/// position that is meant to hold a variable. A consumer that lowers the index
/// as a named value then names a variable no instruction defines, since a
/// constant is materialised as a literal rather than a local; that is how
/// `mov eax, 0x1000; lea eax, [rdi + rax + 0x337]` came to print an undeclared
/// name where `0x1337` belongs.
///
/// The folded displacement wraps at the pointer width, as the address does.
/// A symbolic index keeps its coordinate space: an unscaled symbol becomes
/// the base, and a scaled one is left unchanged.
///
/// # Arguments
///
/// * `op` - The operation to rewrite.
/// * `constants` - Constant-valued variables, consulted for the index.
/// * `ptr_size` - Target pointer width the displacement wraps at.
///
/// # Returns
///
/// The index-free `PtrAdd`, or `None` when `op` is not a `PtrAdd` with a
/// constant index or the displacement does not fit an `i64`.
#[must_use]
pub fn fold_constant_index<T: Target>(
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    ptr_size: PointerSize,
) -> Option<SsaOp<T>> {
    let SsaOp::PtrAdd {
        dest,
        base,
        index: Some(index),
        stride,
        offset,
        result_type,
    } = op
    else {
        return None;
    };
    let constant = constants.get(index)?;
    if matches!(constant, ConstValue::Symbol(_)) {
        // Symbols are coordinates in the host's address space, not numeric
        // displacements. An unscaled symbolic term can become the base;
        // retaining it explicitly keeps module and loaded addresses distinct.
        if *stride != 1 || matches!(constants.get(base), Some(ConstValue::Symbol(_))) {
            return None;
        }
        return Some(SsaOp::PtrAdd {
            dest: *dest,
            base: *index,
            index: Some(*base),
            stride: 1,
            offset: *offset,
            result_type: result_type.clone(),
        });
    }
    let index = constant.resolved_i64()?;
    let scaled = i64::try_from(*stride).ok()?.checked_mul(index)?;
    Some(SsaOp::PtrAdd {
        dest: *dest,
        base: *base,
        index: None,
        stride: 0,
        offset: ptr_size.mask_signed(offset.checked_add(scaled)?),
        result_type: result_type.clone(),
    })
}

/// Memory-cell identity key for a `base + index*stride + offset` address.
///
/// The cell-identity projection of an [`AddressExpr`]: two addresses share a
/// key when they reduce to the same base, the same scaled index (by SSA value
/// identity and stride), and the same constant offset. An absolute address is
/// keyed by its displacement alone, so every materialisation of one image
/// address shares a key. Keeping the scaled-index
/// term lets array-element accesses participate: `arr[i]` loaded then stored, or
/// `ptr[i].field` chains, denote the same cell. It is sound as a cell identity
/// because in SSA the same index value-id denotes one value - distinct indices
/// (`arr[i]` vs `arr[j]`) keep distinct keys.
///
/// Note this is an *equality* key, not a may-alias relation: it carries no
/// access width, so two overlapping accesses of different widths at nearby
/// offsets get distinct keys. Alias reasoning that needs overlap belongs in
/// [`MemoryLocation`](crate::analysis::memory::MemoryLocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AliasKey {
    /// What the address is measured from.
    pub base: AddressBase,
    /// Scaled-index SSA value id, when the address adds one.
    pub index: Option<u32>,
    /// Stride applied to [`Self::index`], in bytes (`0` when no index).
    pub stride_bytes: u64,
    /// Constant offset in bits from the root.
    pub offset_bits: i64,
}

/// Returns the alias key for `value`, falling back to the value's own identity
/// (`base = value`, `offset = 0`) when no simplified key was recorded.
///
/// # Arguments
///
/// * `value` - The value to key.
/// * `keys` - The map produced by [`AddressWalk::alias_keys`].
///
/// # Returns
///
/// The recorded [`AliasKey`], or a fresh identity key for `value`; always
/// `Some`.
#[must_use]
pub fn alias_key_for_value(value: SsaVarId, keys: &BTreeMap<u32, AliasKey>) -> Option<AliasKey> {
    keys.get(&value.as_u32()).copied().or(Some(AliasKey {
        base: AddressBase::Value(value),
        index: None,
        stride_bytes: 0,
        offset_bits: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            PhiNode, PhiOperand, SsaBlock, SsaInstruction,
            ops::AccessQualifiers,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// Builds an empty mock SSA function to append test instructions to.
    fn function() -> SsaFunction<MockTarget> {
        SsaFunction::<MockTarget>::with_capacity(0, 0, 1, 8)
    }

    /// Declares an untyped variable defined at the entry.
    fn var(ir: &mut SsaFunction<MockTarget>, origin: VariableOrigin) -> SsaVarId {
        ir.create_variable(origin, 0, DefSite::entry(), MockType::Unknown)
    }

    /// Wraps `op` in an instruction. `MockTarget` carries no original-instruction
    /// breadcrumb (its `OriginalInstruction` is `()`), so the unit is passed
    /// literally rather than through `synthetic_instruction()`.
    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::new((), op)
    }

    /// A `PtrAdd` over `base` with `index` scaled by `stride` and `offset` added.
    fn ptr_add(
        dest: SsaVarId,
        base: SsaVarId,
        index: Option<SsaVarId>,
        stride: u64,
        offset: i64,
    ) -> SsaOp<MockTarget> {
        SsaOp::PtrAdd {
            dest,
            base,
            index,
            stride,
            offset,
            result_type: MockType::Unknown,
        }
    }

    /// A stack pointer stepped once per pushed register, and read back through
    /// the frame pointer's integer round trip, still resolves to the entry
    /// value it started from.
    ///
    /// `-O0` x86-64 saves `rbp`, `r12` and `rbx`, keeps the stack pointer in
    /// `rbp` as an integer, and restores through it: ten definitions, four of
    /// them conversions. At a depth of eight counting every definition, the
    /// restore's address stopped short, and the restore was a read of memory
    /// nobody could place. A conversion moves nothing, and costs no depth.
    #[test]
    fn a_pushed_stack_pointer_read_back_through_integers_resolves() {
        let mut ir = function();
        let sp = var(&mut ir, VariableOrigin::EntryLiveIn);
        let mut block = SsaBlock::new(0);
        let mut current = sp;
        let mut position = 0usize;
        let mut define = |ir: &mut SsaFunction<MockTarget>,
                          block: &mut SsaBlock<MockTarget>,
                          op: &dyn Fn(SsaVarId) -> SsaOp<MockTarget>| {
            let dest = ir.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, position),
                MockType::Unknown,
            );
            block.add_instruction(instr(op(dest)));
            position = position.saturating_add(1);
            dest
        };
        for _ in 0..6 {
            let from = current;
            current = define(&mut ir, &mut block, &|dest| {
                ptr_add(dest, from, None, 0, -8)
            });
        }
        for _ in 0..2 {
            let from = current;
            let number = define(&mut ir, &mut block, &|dest| SsaOp::PtrToInt {
                dest,
                operand: from,
                target: MockType::U64,
            });
            current = define(&mut ir, &mut block, &|dest| SsaOp::IntToPtr {
                dest,
                operand: number,
                target: MockType::Ptr,
            });
        }
        ir.add_block(block);
        let expr = AddressWalk::new(&ir, PointerSize::Bit64).normalize(current);
        assert_eq!(expr.base, AddressBase::Value(sp));
        assert_eq!(expr.offset_bits, -6 * 8 * 8);
    }

    /// **An alignment mask is the one that clears low bits.** `-16` at either
    /// width and the unsigned `0xffff_fff0` a 32-bit operand carries are both
    /// an alignment of sixteen; a byte truncation, an all-ones mask that
    /// aligns nothing and a zero mask are no alignment.
    #[test]
    fn an_alignment_mask_clears_low_bits_and_nothing_else() {
        assert_eq!(alignment_mask(-16, PointerSize::Bit64), Some(16));
        assert_eq!(alignment_mask(-16, PointerSize::Bit32), Some(16));
        assert_eq!(alignment_mask(0xffff_fff0, PointerSize::Bit32), Some(16));
        assert_eq!(alignment_mask(-4096, PointerSize::Bit64), Some(4096));
        assert_eq!(alignment_mask(0xff, PointerSize::Bit32), None);
        assert_eq!(alignment_mask(-1, PointerSize::Bit64), None);
        assert_eq!(alignment_mask(0, PointerSize::Bit64), None);
        assert_eq!(alignment_mask(-24, PointerSize::Bit64), None);
    }

    /// **A prologue that aligns the stack states the frame's own anchor.**
    /// `lea ecx, [esp + 4]; and esp, -16` leaves every address below it
    /// measured from the aligned value, which is the anchor the walk reports
    /// with its alignment. A second alignment over the frame names no single
    /// space and reports none, and so does a mask of a value that is no frame
    /// address.
    #[test]
    fn a_prologue_that_aligns_the_stack_states_the_frames_anchor() {
        let aligned = |masks: usize, of_the_frame: bool| {
            let mut ir = function();
            let sp = var(&mut ir, VariableOrigin::EntryLiveIn);
            let foreign = var(&mut ir, VariableOrigin::EntryLiveIn);
            let mut block = SsaBlock::new(0);
            let mut position = 0usize;
            let mut define =
                |ir: &mut SsaFunction<MockTarget>,
                 block: &mut SsaBlock<MockTarget>,
                 op: &dyn Fn(SsaVarId) -> SsaOp<MockTarget>| {
                    let dest = ir.create_variable(
                        VariableOrigin::Phi,
                        0,
                        DefSite::instruction(0, position),
                        MockType::Unknown,
                    );
                    block.add_instruction(instr(op(dest)));
                    position = position.saturating_add(1);
                    dest
                };
            let reserved = define(&mut ir, &mut block, &|dest| ptr_add(dest, sp, None, 0, -4));
            let mask = define(&mut ir, &mut block, &|dest| SsaOp::Const {
                dest,
                value: ConstValue::I32(-16),
            });
            let masked = if of_the_frame { reserved } else { foreign };
            let mut anchors = Vec::new();
            for _ in 0..masks {
                anchors.push(define(&mut ir, &mut block, &|dest| SsaOp::And {
                    dest,
                    left: masked,
                    right: mask,
                    flags: None,
                }));
            }
            ir.add_block(block);
            (
                AddressWalk::new(&ir, PointerSize::Bit32).frame_anchor(sp),
                anchors.first().copied(),
            )
        };
        let (found, anchor) = aligned(1, true);
        assert_eq!(found, Some((anchor.expect("one mask"), 16)));
        assert_eq!(aligned(2, true).0, None);
        assert_eq!(aligned(1, false).0, None);
        assert_eq!(aligned(0, true).0, None);
    }

    /// A stack pointer stepped around every call of a long function still
    /// resolves to the entry value: 32-bit x86 pushes each argument, releases
    /// the area with an integer `add` and reserves the next with an integer
    /// `sub`, three displacements and four conversions per call. Forty calls
    /// are 120 displacements, and every one of them only moves the address.
    #[test]
    fn a_stack_pointer_stepped_around_every_call_resolves() {
        let mut ir = function();
        let sp = var(&mut ir, VariableOrigin::EntryLiveIn);
        let mut block = SsaBlock::new(0);
        let mut current = sp;
        let mut position = 0usize;
        let mut define = |ir: &mut SsaFunction<MockTarget>,
                          block: &mut SsaBlock<MockTarget>,
                          op: &dyn Fn(SsaVarId) -> SsaOp<MockTarget>| {
            let dest = ir.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, position),
                MockType::Unknown,
            );
            block.add_instruction(instr(op(dest)));
            position = position.saturating_add(1);
            dest
        };
        for _ in 0..40 {
            let from = current;
            let pushed = define(&mut ir, &mut block, &|dest| {
                ptr_add(dest, from, None, 0, -4)
            });
            for (step, add) in [(0x10, true), (0x10, false)] {
                let previous = if add { pushed } else { current };
                let number = define(&mut ir, &mut block, &|dest| SsaOp::PtrToInt {
                    dest,
                    operand: previous,
                    target: MockType::U32,
                });
                let amount = define(&mut ir, &mut block, &|dest| SsaOp::Const {
                    dest,
                    value: ConstValue::I32(step),
                });
                let moved = define(&mut ir, &mut block, &|dest| {
                    if add {
                        SsaOp::Add {
                            dest,
                            left: number,
                            right: amount,
                            flags: None,
                        }
                    } else {
                        SsaOp::Sub {
                            dest,
                            left: number,
                            right: amount,
                            flags: None,
                        }
                    }
                });
                current = define(&mut ir, &mut block, &|dest| SsaOp::IntToPtr {
                    dest,
                    operand: moved,
                    target: MockType::Ptr,
                });
            }
        }
        ir.add_block(block);
        let expr = AddressWalk::new(&ir, PointerSize::Bit32).normalize(current);
        assert_eq!(expr.base, AddressBase::Value(sp));
        assert_eq!(expr.offset_bits, -40 * 4 * 8);
    }

    /// A constant index is displacement: `base + 0x1000*1 + 0x337` is
    /// `base + 0x1337`, with no index left for a consumer to name.
    #[test]
    fn a_constant_index_folds_into_the_displacement() {
        let (dest, base, index) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let constants = BTreeMap::from([(index, ConstValue::<MockTarget>::U64(0x1000))]);
        let folded = fold_constant_index(
            &ptr_add(dest, base, Some(index), 1, 0x337),
            &constants,
            PointerSize::Bit64,
        );
        assert_eq!(folded, Some(ptr_add(dest, base, None, 0, 0x1337)));

        let scaled = fold_constant_index(
            &ptr_add(dest, base, Some(index), 4, -8),
            &constants,
            PointerSize::Bit64,
        );
        assert_eq!(
            scaled,
            Some(ptr_add(dest, base, None, 0, 0x4000 - 8)),
            "the index is scaled by the stride before it joins the offset"
        );
    }

    /// The folded displacement wraps at the pointer width, the way the address
    /// it is part of does.
    #[test]
    fn a_folded_displacement_wraps_at_the_pointer_width() {
        let (dest, base, index) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let constants = BTreeMap::from([(index, ConstValue::<MockTarget>::U32(0xFFFF_FFFF))]);
        let folded = fold_constant_index(
            &ptr_add(dest, base, Some(index), 1, 1),
            &constants,
            PointerSize::Bit32,
        );
        assert_eq!(folded, Some(ptr_add(dest, base, None, 0, 0)));
    }

    /// An index that is not a known constant is a run-time term and stays.
    #[test]
    fn a_variable_index_is_left_alone() {
        let (dest, base, index) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let op = ptr_add(dest, base, Some(index), 8, 16);
        assert_eq!(
            fold_constant_index(&op, &BTreeMap::new(), PointerSize::Bit64),
            None
        );
        assert_eq!(
            fold_constant_index(
                &ptr_add(dest, base, None, 0, 16),
                &BTreeMap::new(),
                PointerSize::Bit64
            ),
            None,
            "an address with no index has nothing to fold"
        );
    }

    /// **A conversion that changes the width is a term of its own.** A table
    /// indexed by `(signed char)c` adds the byte sign-extended back to a word,
    /// another number than the word `c` it was cut from: the index is the
    /// extension. A conversion that keeps the width only renames the bits,
    /// and the index is what it converts.
    #[test]
    fn a_conversion_that_changes_the_width_is_a_term_of_its_own() {
        let index_of = |cut: bool| {
            let mut ir = function();
            let typed = |ir: &mut SsaFunction<MockTarget>, ty: MockType| {
                ir.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), ty)
            };
            let word = typed(&mut ir, MockType::I32);
            let (table, converted, sum) = (
                typed(&mut ir, MockType::Ptr),
                typed(&mut ir, MockType::I32),
                typed(&mut ir, MockType::I32),
            );
            let mut block = SsaBlock::new(0);
            block.add_instruction(instr(SsaOp::Const {
                dest: table,
                value: ConstValue::Symbol(0x1_101c),
            }));
            if cut {
                let byte = typed(&mut ir, MockType::I8);
                block.add_instruction(instr(SsaOp::IntConv {
                    dest: byte,
                    operand: word,
                    target: MockType::I8,
                    overflow_check: false,
                    unsigned: false,
                }));
                block.add_instruction(instr(SsaOp::IntConv {
                    dest: converted,
                    operand: byte,
                    target: MockType::I32,
                    overflow_check: false,
                    unsigned: false,
                }));
            } else {
                block.add_instruction(instr(SsaOp::IntConv {
                    dest: converted,
                    operand: word,
                    target: MockType::U32,
                    overflow_check: false,
                    unsigned: true,
                }));
            }
            block.add_instruction(instr(SsaOp::Add {
                dest: sum,
                left: table,
                right: converted,
                flags: None,
            }));
            ir.add_block(block);
            (
                AddressWalk::new(&ir, PointerSize::Bit32)
                    .normalize(sum)
                    .index,
                word,
                converted,
            )
        };
        let (index, _, converted) = index_of(true);
        assert_eq!(index, Some(converted), "the extension of the cut byte");
        let (index, word, _) = index_of(false);
        assert_eq!(index, Some(word), "a renaming is looked through");
    }

    /// A `base + const` add decodes to that base with the constant folded into
    /// the bit offset.
    #[test]
    fn normalize_folds_base_plus_constant() {
        let mut ir = function();
        let base = var(&mut ir, VariableOrigin::Argument(0));
        let offset = var(&mut ir, VariableOrigin::Local(0));
        let addr = var(&mut ir, VariableOrigin::Local(1));
        let mut block = SsaBlock::with_capacity(0, 0, 4);
        block.add_instruction(instr(SsaOp::Const {
            dest: offset,
            value: ConstValue::I32(4),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: addr,
            left: base,
            right: offset,
            flags: None,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr);
        assert_eq!(address.base, AddressBase::Value(base));
        assert_eq!(address.index, None);
        // 4 bytes folds to 32 bits.
        assert_eq!(address.offset_bits, 32);
    }

    /// **A chain of constant steps is walked whatever its length.** A
    /// function adjusting one pointer step by step - add eight, subtract four,
    /// a `PtrAdd` of four, a copy, a hundred thousand times over - builds a
    /// chain as long as its code, and a walk recursing once per step
    /// overflowed the stack of the thread normalizing it, aborting the whole
    /// worker. Walked in a loop on a test thread's default stack, the chain
    /// resolves to its root with every step folded: eight bytes per round.
    #[test]
    fn a_chain_of_constant_steps_is_walked_whatever_its_length() {
        const ROUNDS: i64 = 25_000;
        let mut ir = SsaFunction::<MockTarget>::with_capacity(0, 0, 1, 8);
        let base = var(&mut ir, VariableOrigin::Argument(0));
        let eight = ir.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::Unknown,
        );
        let four = ir.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::Unknown,
        );
        let mut block = SsaBlock::with_capacity(0, 0, 4);
        block.add_instruction(instr(SsaOp::Const {
            dest: eight,
            value: ConstValue::I64(8),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: four,
            value: ConstValue::I64(4),
        }));
        let mut current = base;
        let mut at = 2usize;
        for _ in 0..ROUNDS {
            for step in 0..4 {
                // Defined where it is, so its definition is found at once.
                let dest = ir.create_variable(
                    VariableOrigin::Local(2),
                    0,
                    DefSite::instruction(0, at),
                    MockType::Unknown,
                );
                at = at.saturating_add(1);
                block.add_instruction(instr(match step {
                    0 => SsaOp::Add {
                        dest,
                        left: current,
                        right: eight,
                        flags: None,
                    },
                    1 => SsaOp::Sub {
                        dest,
                        left: current,
                        right: four,
                        flags: None,
                    },
                    2 => ptr_add(dest, current, None, 0, 4),
                    _ => SsaOp::Copy { dest, src: current },
                }));
                current = dest;
            }
        }
        ir.add_block(block);
        ir.recompute_uses();

        let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(current);
        assert_eq!(address.base, AddressBase::Value(base));
        assert_eq!(address.index, None);
        assert_eq!(address.offset_bits, ROUNDS * 8 * 8);
    }

    /// **A chain that returns to itself is no address.** Code no edge reaches
    /// can define a value through itself without a phi: `a = b + 8; b = a`,
    /// the loop's phi gone once its entry was. Walked step after step, such a
    /// chain has no end, and every step it folds is kept until the end it
    /// never reaches, without bound in time or memory. Every kind of step the
    /// walk follows closes a cycle here; each normalizes to the value asked
    /// about.
    #[test]
    fn a_chain_that_returns_to_itself_is_no_address() {
        for kind in 0..4 {
            let mut ir = SsaFunction::<MockTarget>::with_capacity(0, 0, 1, 4);
            let eight = ir.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::Unknown,
            );
            let looped = ir.create_variable(
                VariableOrigin::Local(1),
                0,
                DefSite::instruction(0, 1),
                MockType::Unknown,
            );
            let back = ir.create_variable(
                VariableOrigin::Local(2),
                0,
                DefSite::instruction(0, 2),
                MockType::Unknown,
            );
            let mut block = SsaBlock::with_capacity(0, 0, 3);
            block.add_instruction(instr(SsaOp::Const {
                dest: eight,
                value: ConstValue::I64(8),
            }));
            block.add_instruction(instr(match kind {
                0 => SsaOp::Add {
                    dest: looped,
                    left: back,
                    right: eight,
                    flags: None,
                },
                1 => SsaOp::Sub {
                    dest: looped,
                    left: back,
                    right: eight,
                    flags: None,
                },
                2 => ptr_add(looped, back, None, 0, 8),
                _ => SsaOp::Copy {
                    dest: looped,
                    src: back,
                },
            }));
            block.add_instruction(instr(SsaOp::Copy {
                dest: back,
                src: looped,
            }));
            ir.add_block(block);
            ir.recompute_uses();

            let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(looped);
            assert_eq!(address, AddressExpr::identity(looped), "kind {kind}");
        }
    }

    /// **A symbol added to a scaled index is the indexed base.** dash's
    /// `(i << 2) + &table` read the table's address as a displacement of
    /// `i << 2`, rooting the address at the shift, so the table lost every
    /// indexed access it has. The symbol is the base and the shift the index;
    /// the control, a numeric constant, is still a displacement.
    #[test]
    fn a_symbol_added_to_a_scaled_index_is_the_indexed_base() {
        let decode = |constant| {
            let mut ir = function();
            let index = var(&mut ir, VariableOrigin::Argument(0));
            let amount = var(&mut ir, VariableOrigin::Local(0));
            let scaled = var(&mut ir, VariableOrigin::Local(1));
            let table = var(&mut ir, VariableOrigin::Local(2));
            let addr = var(&mut ir, VariableOrigin::Local(3));
            let mut block = SsaBlock::with_capacity(0, 0, 4);
            block.add_instruction(instr(SsaOp::Const {
                dest: amount,
                value: ConstValue::I32(2),
            }));
            block.add_instruction(instr(SsaOp::Shl {
                dest: scaled,
                value: index,
                amount,
                flags: None,
            }));
            block.add_instruction(instr(SsaOp::Const {
                dest: table,
                value: constant,
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: addr,
                left: scaled,
                right: table,
                flags: None,
            }));
            ir.add_block(block);
            ir.recompute_uses();
            (
                AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr),
                index,
                scaled,
            )
        };
        let (indexed, index, _) = decode(ConstValue::Symbol(0x1000));
        assert_eq!(indexed.base, AddressBase::Symbol);
        assert_eq!(indexed.index, Some(index));
        assert_eq!(indexed.stride_bytes, 4);
        assert_eq!(indexed.offset_bits, 0x1000 * 8);
        let (displaced, _, scaled) = decode(ConstValue::U64(0x1000));
        assert_eq!(displaced.base, AddressBase::Value(scaled));
        assert_eq!(displaced.offset_bits, 0x1000 * 8);
    }

    /// The two lowerings of one displacement must decode to one address.
    ///
    /// A 32-bit target reaches `[base - 8]` either as a sign-extended `-8` or as
    /// `add reg, 0xFFFF_FFF8`, whose unsigned constant `ConstValue::resolved_i64`
    /// zero-extends to `+4294967288`. Both name the same cell. Left as `i64`
    /// they sit 4 GiB apart, and `IndirectLocation::may_alias` would then prove
    /// them *disjoint* - a false NoAlias, which lets a stale value survive a
    /// store to that cell and lets DSE drop a store that was read.
    #[test]
    fn both_lowerings_of_one_displacement_canonicalise_together() {
        fn address_for(value: ConstValue<MockTarget>, ptr_size: PointerSize) -> AddressExpr {
            let mut ir = function();
            let base = var(&mut ir, VariableOrigin::Argument(0));
            let offset = var(&mut ir, VariableOrigin::Local(1));
            let addr = var(&mut ir, VariableOrigin::Local(2));
            let mut block = SsaBlock::with_capacity(0, 0, 2);
            block.add_instruction(instr(SsaOp::Const {
                dest: offset,
                value,
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: addr,
                left: base,
                right: offset,
                flags: None,
            }));
            ir.add_block(block);
            ir.recompute_uses();
            AddressWalk::new(&ir, ptr_size).normalize(addr)
        }

        let wrapped = address_for(ConstValue::U32(0xFFFF_FFF8), PointerSize::Bit32);
        let signed = address_for(ConstValue::I32(-8), PointerSize::Bit32);
        assert_eq!(
            wrapped, signed,
            "on a 32-bit target 0xFFFF_FFF8 and -8 are the same displacement"
        );
        assert_eq!(signed.offset_bits, -64, "-8 bytes is -64 bits");

        // On a 64-bit target the two really are different addresses, and the
        // model must not conflate them.
        let wide_wrapped = address_for(ConstValue::U32(0xFFFF_FFF8), PointerSize::Bit64);
        let wide_signed = address_for(ConstValue::I32(-8), PointerSize::Bit64);
        assert_ne!(
            wide_wrapped, wide_signed,
            "on a 64-bit target +4294967288 and -8 are distinct addresses"
        );
        assert_eq!(wide_signed.offset_bits, -64);
    }

    /// Narrower targets wrap too, and the canonical form is the signed one.
    #[test]
    fn canonicalisation_follows_the_target_width() {
        assert_eq!(canonical_offset_bits(-64, PointerSize::Bit32), -64);
        // 0xFFFF_FFF8 bytes, in bits.
        assert_eq!(
            canonical_offset_bits(4_294_967_288 * 8, PointerSize::Bit32),
            -64
        );
        // 0xFFF8 bytes on a 16-bit target is likewise -8.
        assert_eq!(canonical_offset_bits(65_528 * 8, PointerSize::Bit16), -64);
        // 64-bit is the identity: `i64` already is the address domain.
        assert_eq!(
            canonical_offset_bits(4_294_967_288 * 8, PointerSize::Bit64),
            4_294_967_288 * 8
        );
        // A sub-byte member offset rides along unchanged.
        assert_eq!(canonical_offset_bits(-64 + 3, PointerSize::Bit32), -64 + 3);
    }

    /// **An integer's displacement wraps at the integer's own width, not the
    /// pointer's.** `x + 2^32` over a 64-bit `x` where pointers are 32 bits
    /// moves `x` by `2^32`: wrapped at the pointer width it read as no move, and
    /// the algebraic pass replaced the addition with `x`. A 32-bit integer where
    /// pointers are 64 bits wraps at 32, as its arithmetic does: `x +
    /// 0xFFFF_FFF8` is `x - 8` there.
    #[test]
    fn an_integers_displacement_wraps_at_its_own_width() {
        fn address_for(
            ty: MockType,
            value: ConstValue<MockTarget>,
            ptr_size: PointerSize,
        ) -> (AddressExpr, SsaVarId) {
            let mut ir = function();
            let base = ir.create_variable(VariableOrigin::Argument(0), 0, DefSite::entry(), ty);
            let offset = ir.create_variable(VariableOrigin::Local(1), 0, DefSite::entry(), ty);
            let addr = ir.create_variable(VariableOrigin::Local(2), 0, DefSite::entry(), ty);
            let mut block = SsaBlock::with_capacity(0, 0, 2);
            block.add_instruction(instr(SsaOp::Const {
                dest: offset,
                value,
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: addr,
                left: base,
                right: offset,
                flags: None,
            }));
            ir.add_block(block);
            ir.recompute_uses();
            (AddressWalk::new(&ir, ptr_size).normalize(addr), base)
        }

        let (wide, base) = address_for(MockType::I64, ConstValue::I64(1 << 32), PointerSize::Bit32);
        assert_eq!(wide.base, AddressBase::Value(base));
        assert_eq!(
            wide.offset_bits,
            (1i64 << 32) * 8,
            "a 64-bit move of 2^32 stays one"
        );

        let (narrow, base) = address_for(
            MockType::I32,
            ConstValue::U32(0xFFFF_FFF8),
            PointerSize::Bit64,
        );
        assert_eq!(narrow.base, AddressBase::Value(base));
        assert_eq!(narrow.offset_bits, -64, "a 32-bit integer wraps at 32 bits");
    }

    /// Ordinary displacements - including genuinely negative ones - must keep
    /// folding, or the guard above would cost real precision.
    #[test]
    fn ordinary_displacements_still_fold() {
        for (value, expected_bits) in [
            (ConstValue::<MockTarget>::I32(-8), -64i64),
            (ConstValue::<MockTarget>::I32(16), 128),
            (ConstValue::<MockTarget>::I32(4096), 32768),
        ] {
            let mut ir = function();
            let base = var(&mut ir, VariableOrigin::Argument(0));
            let offset = var(&mut ir, VariableOrigin::Local(1));
            let addr = var(&mut ir, VariableOrigin::Local(2));
            let mut block = SsaBlock::with_capacity(0, 0, 2);
            block.add_instruction(instr(SsaOp::Const {
                dest: offset,
                value: value.clone(),
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: addr,
                left: base,
                right: offset,
                flags: None,
            }));
            ir.add_block(block);
            ir.recompute_uses();

            let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr);
            assert_eq!(
                address.base,
                AddressBase::Value(base),
                "{value:?} should decompose onto the base"
            );
            assert_eq!(address.offset_bits, expected_bits, "for {value:?}");
        }
    }

    /// **A value added to an address of a region the function names indexes
    /// it a byte at a time.** A byte array is walked as `add a0, sp, a0; sb a1,
    /// 0(a0)`: no scale names the index, and the address normalized to
    /// nothing, so the frame had no use there. The frame, through the stack
    /// pointer the function was entered with, and an image address are such
    /// regions, and an offset on the index side moves the address; the
    /// controls - two values neither of which names a region, and two regions
    /// - stay unresolved, since two addresses are never added.
    #[test]
    fn a_value_added_to_a_region_indexes_it_by_bytes() {
        let normalize = |stepped: bool, region_base: bool| {
            let mut ir = function();
            let entry = var(&mut ir, VariableOrigin::EntryLiveIn);
            if region_base && let Some(variable) = ir.variable_mut(entry) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            let index = var(&mut ir, VariableOrigin::Argument(0));
            let displacement = var(&mut ir, VariableOrigin::Local(0));
            let frame = var(&mut ir, VariableOrigin::Local(1));
            let one = var(&mut ir, VariableOrigin::Local(2));
            let moved = var(&mut ir, VariableOrigin::Local(3));
            let addr = var(&mut ir, VariableOrigin::Local(4));
            let mut block = SsaBlock::with_capacity(0, 0, 5);
            block.add_instruction(instr(SsaOp::Const {
                dest: displacement,
                value: ConstValue::I64(-32),
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: frame,
                left: entry,
                right: displacement,
                flags: None,
            }));
            block.add_instruction(instr(SsaOp::Const {
                dest: one,
                value: ConstValue::I64(1),
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: moved,
                left: index,
                right: one,
                flags: None,
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: addr,
                left: if stepped { moved } else { index },
                right: frame,
                flags: None,
            }));
            ir.add_block(block);
            ir.recompute_uses();
            (
                AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr),
                entry,
                index,
                addr,
            )
        };
        let (walked, entry, index, _) = normalize(false, true);
        assert_eq!(walked.base, AddressBase::Value(entry));
        assert_eq!(walked.index, Some(index));
        assert_eq!(walked.stride_bytes, 1);
        assert_eq!(walked.offset_bits, -32 * 8);
        let (stepped, _, index, _) = normalize(true, true);
        assert_eq!(stepped.index, Some(index));
        assert_eq!(
            stepped.offset_bits,
            -31 * 8,
            "the index side's offset moves it"
        );
        let (unknown, _, _, addr) = normalize(false, false);
        assert!(unknown.is_identity_of(addr), "no region: {unknown:?}");
        // An image address plus the frame: two regions.
        let (both, _, _, addr) = {
            let mut ir = function();
            let entry = var(&mut ir, VariableOrigin::EntryLiveIn);
            if let Some(variable) = ir.variable_mut(entry) {
                variable.set_role(VariableRole::EntryStackPointer);
            }
            let table = var(&mut ir, VariableOrigin::Local(0));
            let addr = var(&mut ir, VariableOrigin::Local(1));
            let mut block = SsaBlock::with_capacity(0, 0, 2);
            block.add_instruction(instr(SsaOp::Const {
                dest: table,
                value: ConstValue::Symbol(0x1000),
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: addr,
                left: entry,
                right: table,
                flags: None,
            }));
            ir.add_block(block);
            ir.recompute_uses();
            (
                AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr),
                entry,
                table,
                addr,
            )
        };
        assert!(both.is_identity_of(addr), "two regions: {both:?}");
        // An image address indexed by a value.
        let mut ir = function();
        let index = var(&mut ir, VariableOrigin::Argument(0));
        let table = var(&mut ir, VariableOrigin::Local(0));
        let addr = var(&mut ir, VariableOrigin::Local(1));
        let mut block = SsaBlock::with_capacity(0, 0, 2);
        block.add_instruction(instr(SsaOp::Const {
            dest: table,
            value: ConstValue::Symbol(0x1000),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: addr,
            left: table,
            right: index,
            flags: None,
        }));
        ir.add_block(block);
        ir.recompute_uses();
        let indexed = AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr);
        assert_eq!(indexed.base, AddressBase::Symbol);
        assert_eq!(indexed.index, Some(index));
        assert_eq!(indexed.stride_bytes, 1);
        assert_eq!(indexed.offset_bits, 0x1000 * 8);
    }

    /// A `base + index*stride` add records the scaled index and its stride.
    #[test]
    fn normalize_records_scaled_index() {
        let mut ir = function();
        let base = var(&mut ir, VariableOrigin::Argument(0));
        let index = var(&mut ir, VariableOrigin::Argument(1));
        let stride = var(&mut ir, VariableOrigin::Local(2));
        let scaled = var(&mut ir, VariableOrigin::Local(3));
        let addr = var(&mut ir, VariableOrigin::Local(4));
        let mut block = SsaBlock::with_capacity(0, 0, 4);
        block.add_instruction(instr(SsaOp::Const {
            dest: stride,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: scaled,
            left: index,
            right: stride,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: addr,
            left: base,
            right: scaled,
            flags: None,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr);
        assert_eq!(address.base, AddressBase::Value(base));
        assert_eq!(address.index, Some(index));
        assert_eq!(address.stride_bytes, 8);
        assert_eq!(address.offset_bits, 0);
    }

    /// **An index scaled in steps is one scaled index.** RISC-V forms
    /// `base + i * 24` as `base + (((i << 1) + i) << 3)` and `i * 7` as
    /// `(i << 3) - i`: the index is `i` and the stride what the steps
    /// together multiply it by, where the last shift alone was taken for the
    /// stride over an index of `(i << 1) + i` (24-byte records read as 8-byte
    /// elements). A difference that leaves nothing of `i` scales nothing.
    #[test]
    fn an_index_scaled_in_steps_is_one_scaled_index() {
        // `base + ((i << shift) op i) << 3`.
        let normalized = |shift: i32, subtract: bool| {
            let mut ir = function();
            let base = var(&mut ir, VariableOrigin::Argument(0));
            let index = var(&mut ir, VariableOrigin::Argument(1));
            let first = var(&mut ir, VariableOrigin::Local(2));
            let three = var(&mut ir, VariableOrigin::Local(3));
            let doubled = var(&mut ir, VariableOrigin::Local(4));
            let combined = var(&mut ir, VariableOrigin::Local(5));
            let scaled = var(&mut ir, VariableOrigin::Local(6));
            let addr = var(&mut ir, VariableOrigin::Local(7));
            let mut block = SsaBlock::with_capacity(0, 0, 7);
            block.add_instruction(instr(SsaOp::Const {
                dest: first,
                value: ConstValue::I32(shift),
            }));
            block.add_instruction(instr(SsaOp::Const {
                dest: three,
                value: ConstValue::I32(3),
            }));
            block.add_instruction(instr(SsaOp::Shl {
                dest: doubled,
                value: index,
                amount: first,
                flags: None,
            }));
            block.add_instruction(instr(if subtract {
                SsaOp::Sub {
                    dest: combined,
                    left: doubled,
                    right: index,
                    flags: None,
                }
            } else {
                SsaOp::Add {
                    dest: combined,
                    left: doubled,
                    right: index,
                    flags: None,
                }
            }));
            block.add_instruction(instr(SsaOp::Shl {
                dest: scaled,
                value: combined,
                amount: three,
                flags: None,
            }));
            block.add_instruction(instr(SsaOp::Add {
                dest: addr,
                left: base,
                right: scaled,
                flags: None,
            }));
            ir.add_block(block);
            ir.recompute_uses();
            let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr);
            (
                address.base == AddressBase::Value(base),
                address.index == Some(index),
                address.stride_bytes,
            )
        };
        assert_eq!(normalized(1, false), (true, true, 24));
        assert_eq!(normalized(3, true), (true, true, 56));
        // `(i << 0) - i` is nothing of `i`: no index of it.
        assert!(!normalized(0, true).1);
    }

    /// A `PtrAdd` decodes to the same `AddressExpr` as the equivalent shredded
    /// add-chain, so consumers are insensitive to the lowering.
    #[test]
    fn normalize_reads_ptradd_directly() {
        // A `PtrAdd` (the de-shredded address) must normalize to exactly the
        // `AddressExpr` the equivalent `base + index*stride + disp` chain
        // produces, so consumers are insensitive to which lowering they see.
        let mut ir = function();
        let base = var(&mut ir, VariableOrigin::Argument(0));
        let index = var(&mut ir, VariableOrigin::Argument(1));
        let addr = var(&mut ir, VariableOrigin::Local(0));
        let mut block = SsaBlock::with_capacity(0, 0, 1);
        block.add_instruction(instr(SsaOp::PtrAdd {
            dest: addr,
            base,
            index: Some(index),
            stride: 4,
            offset: 8,
            result_type: MockType::Ptr,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr);
        assert_eq!(address.base, AddressBase::Value(base));
        assert_eq!(address.index, Some(index));
        assert_eq!(address.stride_bytes, 4);
        // 8 bytes → 64 bits, matching the chain-based normalization.
        assert_eq!(address.offset_bits, 64);
    }

    /// A value with no definition normalizes to its own identity address.
    #[test]
    fn normalize_unknown_value_is_identity() {
        let ir = function();
        let value = SsaVarId::from_index(7);
        let address = AddressWalk::new(&ir, PointerSize::Bit64).normalize(value);
        assert_eq!(address.base, AddressBase::Value(value));
        assert_eq!(address.index, None);
        assert_eq!(address.offset_bits, 0);
    }

    /// Emits `dest = value` into `block`.
    fn constant(block: &mut SsaBlock<MockTarget>, dest: SsaVarId, value: ConstValue<MockTarget>) {
        block.add_instruction(instr(SsaOp::Const { dest, value }));
    }

    /// A constant address has no base value: it is measured from zero, and the
    /// constant is its displacement.
    #[test]
    fn a_constant_address_is_absolute() {
        let mut ir = function();
        let address = var(&mut ir, VariableOrigin::Local(0));
        let mut block = SsaBlock::with_capacity(0, 0, 1);
        constant(&mut block, address, ConstValue::U64(0x60_1040));
        ir.add_block(block);
        ir.recompute_uses();

        let decoded = AddressWalk::new(&ir, PointerSize::Bit64).normalize(address);
        assert_eq!(decoded.base, AddressBase::Number);
        assert_eq!(
            decoded.root, address,
            "the root is the constant that stated it"
        );
        assert_eq!(decoded.offset_bits, 0x60_1040 * 8);
        assert_eq!(decoded.index, None);
    }

    /// An access through a constant address plus a displacement is reported at
    /// the moved address; the bare address stored to memory is taken, not
    /// accessed; and the access through an unknown pointer is not reported.
    #[test]
    fn constant_addresses_are_reported_where_they_are_used() {
        let mut ir = function();
        let base = var(&mut ir, VariableOrigin::Local(0));
        let eight = var(&mut ir, VariableOrigin::Local(1));
        let field = var(&mut ir, VariableOrigin::Local(2));
        let value = var(&mut ir, VariableOrigin::Local(3));
        let elsewhere = var(&mut ir, VariableOrigin::Argument(0));
        let mut block = SsaBlock::with_capacity(0, 0, 5);
        constant(&mut block, base, ConstValue::U64(0x1000));
        constant(&mut block, eight, ConstValue::U64(8));
        block.add_instruction(instr(SsaOp::Add {
            dest: field,
            left: base,
            right: eight,
            flags: None,
        }));
        constant(&mut block, value, ConstValue::U32(7));
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr: field,
            value,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr: elsewhere,
            value: base,
            value_type: MockType::I64,
            access: AccessQualifiers::FLAT,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let constants = BTreeMap::from([(base, 0x1000)]);
        let mut uses = Vec::new();
        AddressWalk::new(&ir, PointerSize::Bit64)
            .for_each_absolute_use(&constants, |used| uses.push(used));
        let reported: Vec<(u64, bool)> = uses
            .iter()
            .map(|used| match used {
                AbsoluteUse::Access {
                    address, access, ..
                } => (*address, access.writes),
                AbsoluteUse::Taken { address, .. } | AbsoluteUse::Compared { address, .. } => {
                    (*address, false)
                }
            })
            .collect();
        assert_eq!(reported, [(0x1008, true), (0x1000, false)]);
        assert!(matches!(uses[1], AbsoluteUse::Taken { .. }));
    }

    /// An index a `PtrAdd` adds after scaling it itself is the scaled index
    /// the add chain states: `[x1, x0]` over `x0 = i << 2` normalizes as `x1 +
    /// i*4`, and with a stride of its own the two scales multiply.
    #[test]
    fn a_ptr_add_of_a_scaled_index_is_indexed_by_the_unscaled_one() {
        let decode = |stride| {
            let mut ir = function();
            let base = var(&mut ir, VariableOrigin::Argument(0));
            let index = var(&mut ir, VariableOrigin::Argument(1));
            let amount = var(&mut ir, VariableOrigin::Local(0));
            let scaled = var(&mut ir, VariableOrigin::Local(1));
            let addr = var(&mut ir, VariableOrigin::Local(2));
            let mut block = SsaBlock::with_capacity(0, 0, 3);
            constant(&mut block, amount, ConstValue::I32(2));
            block.add_instruction(instr(SsaOp::Shl {
                dest: scaled,
                value: index,
                amount,
                flags: None,
            }));
            block.add_instruction(instr(ptr_add(addr, base, Some(scaled), stride, 8)));
            ir.add_block(block);
            ir.recompute_uses();
            (
                AddressWalk::new(&ir, PointerSize::Bit64).normalize(addr),
                base,
                index,
            )
        };
        let (address, base, index) = decode(1);
        assert_eq!(address.base, AddressBase::Value(base));
        assert_eq!(address.index, Some(index));
        assert_eq!(address.stride_bytes, 4);
        assert_eq!(address.offset_bits, 64);
        let (address, _, index) = decode(2);
        assert_eq!(address.index, Some(index));
        assert_eq!(address.stride_bytes, 8);
    }

    /// A pointer a loop steps starts where its phi's other operand points:
    /// `p = phi(base - 8, p + 1)` compared as `p + 1` starts at `base - 8`.
    /// Where the phi's starts disagree there is no one origin.
    #[test]
    fn a_walked_pointer_starts_where_its_phi_enters() {
        let walk = |second_start: i64| {
            let mut ir = function();
            let base = var(&mut ir, VariableOrigin::Argument(0));
            let low = var(&mut ir, VariableOrigin::Local(0));
            let start = var(&mut ir, VariableOrigin::Local(1));
            let other = var(&mut ir, VariableOrigin::Local(2));
            let one = var(&mut ir, VariableOrigin::Local(3));
            let next = var(&mut ir, VariableOrigin::Local(4));
            let walked =
                ir.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::Unknown);
            let mut entry = SsaBlock::with_capacity(0, 0, 4);
            entry.add_instruction(instr(ptr_add(start, base, None, 0, -8)));
            constant(&mut entry, low, ConstValue::I64(second_start));
            entry.add_instruction(instr(SsaOp::Add {
                dest: other,
                left: base,
                right: low,
                flags: None,
            }));
            entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
            let mut body = SsaBlock::with_capacity(1, 0, 3);
            let mut phi = PhiNode::new(walked, VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(start, 0));
            phi.add_operand(PhiOperand::new(other, 0));
            phi.add_operand(PhiOperand::new(next, 1));
            body.add_phi(phi);
            constant(&mut body, one, ConstValue::I64(1));
            body.add_instruction(instr(SsaOp::Add {
                dest: next,
                left: walked,
                right: one,
                flags: None,
            }));
            body.add_instruction(instr(SsaOp::returns(None)));
            ir.add_block(entry);
            ir.add_block(body);
            ir.recompute_uses();
            (
                AddressWalk::new(&ir, PointerSize::Bit64).walk_origin(next),
                base,
            )
        };
        let (origin, base) = walk(-8);
        let origin = origin.expect("one origin");
        assert_eq!(origin.base, AddressBase::Value(base));
        assert_eq!(origin.offset_bits, -64);
        assert_eq!(walk(-4).0, None, "two starts that disagree");
    }

    /// An access through a pointer a loop walks is measured from where the
    /// walk starts, the walk an index of its step: `p = phi(base + 4, p +
    /// 12)` read at `p + 8` reads `base + 12 + k * 12`. The controls: a phi
    /// no operand moves, and one whose starts disagree, stay their own bases.
    #[test]
    fn an_access_through_a_walk_is_measured_from_its_start() {
        let walk = |moves: i64, second_start: Option<i64>| {
            let mut ir = function();
            let base = var(&mut ir, VariableOrigin::Argument(0));
            let start = var(&mut ir, VariableOrigin::Local(0));
            let other = var(&mut ir, VariableOrigin::Local(1));
            let next = var(&mut ir, VariableOrigin::Local(2));
            let read = var(&mut ir, VariableOrigin::Local(3));
            let walked =
                ir.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::Unknown);
            let mut entry = SsaBlock::with_capacity(0, 0, 3);
            entry.add_instruction(instr(ptr_add(start, base, None, 0, 4)));
            entry.add_instruction(instr(ptr_add(
                other,
                base,
                None,
                0,
                second_start.unwrap_or(4),
            )));
            entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
            let mut body = SsaBlock::with_capacity(1, 0, 3);
            let mut phi = PhiNode::new(walked, VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(start, 0));
            if second_start.is_some() {
                phi.add_operand(PhiOperand::new(other, 0));
            }
            phi.add_operand(PhiOperand::new(next, 1));
            body.add_phi(phi);
            body.add_instruction(instr(ptr_add(read, walked, None, 0, 8)));
            body.add_instruction(instr(ptr_add(next, walked, None, 0, moves)));
            body.add_instruction(instr(SsaOp::returns(None)));
            ir.add_block(entry);
            ir.add_block(body);
            ir.recompute_uses();
            let walk = AddressWalk::new(&ir, PointerSize::Bit64);
            (walk.normalize(read).measured_from_walk(&walk), base, walked)
        };
        let (address, base, walked) = walk(12, None);
        assert_eq!(address.base, AddressBase::Value(base));
        assert_eq!(address.root, base);
        assert_eq!(address.root_offset_bits, 96);
        assert_eq!(address.index, Some(walked));
        assert_eq!(address.stride_bytes, 12);
        let (still, _, walked) = walk(0, None);
        assert_eq!(
            still.base,
            AddressBase::Value(walked),
            "a phi nothing moves"
        );
        assert_eq!(still.index, None);
        let (split, _, walked) = walk(12, Some(8));
        assert_eq!(split.base, AddressBase::Value(walked), "two starts");
    }

    /// A frame address is reported at its offset from the entry stack
    /// pointer: the start of an array where a store lands and where its
    /// address is stored, and the address one past its end where a branch
    /// compares it, which reads nothing through it.
    #[test]
    fn frame_addresses_are_reported_where_they_are_used() {
        let mut ir = function();
        let entry = var(&mut ir, VariableOrigin::Argument(0));
        let size = var(&mut ir, VariableOrigin::Local(0));
        let start = var(&mut ir, VariableOrigin::Local(1));
        let span = var(&mut ir, VariableOrigin::Local(2));
        let end = var(&mut ir, VariableOrigin::Local(3));
        let value = var(&mut ir, VariableOrigin::Local(4));
        let cursor = var(&mut ir, VariableOrigin::Argument(1));
        let mut block = SsaBlock::with_capacity(0, 0, 8);
        constant(&mut block, size, ConstValue::I64(-32));
        block.add_instruction(instr(SsaOp::Add {
            dest: start,
            left: entry,
            right: size,
            flags: None,
        }));
        constant(&mut block, span, ConstValue::I64(32));
        block.add_instruction(instr(SsaOp::Add {
            dest: end,
            left: start,
            right: span,
            flags: None,
        }));
        constant(&mut block, value, ConstValue::U32(7));
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr: start,
            value,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
        block.add_instruction(instr(SsaOp::StoreIndirect {
            addr: cursor,
            value: start,
            value_type: MockType::I64,
            access: AccessQualifiers::FLAT,
        }));
        block.add_instruction(instr(SsaOp::BranchCmp {
            left: cursor,
            right: end,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 0,
            false_target: 0,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let mut uses = Vec::new();
        AddressWalk::new(&ir, PointerSize::Bit64).for_each_frame_use(entry, |used| uses.push(used));
        assert!(matches!(
            uses.as_slice(),
            [
                FrameUse::Access {
                    address: -32,
                    stride: None,
                    ..
                },
                FrameUse::Taken {
                    address: -32,
                    stride: None
                },
                FrameUse::Compared {
                    address: 0,
                    stride: None,
                    against,
                },
            ] if *against == cursor
        ));
    }

    /// Two materialisations of one address name one cell, and a displacement
    /// off one of them lands where the other constant would.
    #[test]
    fn every_materialisation_of_an_address_shares_a_key() {
        let mut ir = function();
        let first = var(&mut ir, VariableOrigin::Local(0));
        let second = var(&mut ir, VariableOrigin::Local(1));
        let eight = var(&mut ir, VariableOrigin::Local(2));
        let displaced = var(&mut ir, VariableOrigin::Local(3));
        let direct = var(&mut ir, VariableOrigin::Local(4));
        let mut block = SsaBlock::with_capacity(0, 0, 5);
        constant(&mut block, first, ConstValue::U64(0x60_1040));
        constant(&mut block, second, ConstValue::I64(0x60_1040));
        constant(&mut block, eight, ConstValue::I32(8));
        block.add_instruction(instr(SsaOp::Add {
            dest: displaced,
            left: first,
            right: eight,
            flags: None,
        }));
        constant(&mut block, direct, ConstValue::U32(0x60_1048));
        ir.add_block(block);
        ir.recompute_uses();

        let keys = AddressWalk::new(&ir, PointerSize::Bit64).alias_keys();
        let key = |value| alias_key_for_value(value, &keys);
        assert_eq!(
            key(first),
            key(second),
            "one address, two constants, one cell"
        );
        assert_eq!(key(displaced), key(direct), "0x601040 + 8 is 0x601048");
        assert_ne!(key(first), key(direct));
        let displaced_expr = AddressWalk::new(&ir, PointerSize::Bit64).normalize(displaced);
        assert_eq!(displaced_expr.root, first);
        assert_eq!(
            displaced_expr.root_offset_bits, 64,
            "8 bytes past the constant"
        );
        assert_eq!(key(direct).map(|key| key.base), Some(AddressBase::Number));
    }

    /// An absolute address above the signed range of a 32-bit target is
    /// canonicalised like any other displacement, so both spellings of it agree.
    #[test]
    fn a_high_absolute_address_canonicalises_at_the_pointer_width() {
        let decode = |value: ConstValue<MockTarget>| {
            let mut ir = function();
            let address = var(&mut ir, VariableOrigin::Local(0));
            let mut block = SsaBlock::with_capacity(0, 0, 1);
            constant(&mut block, address, value);
            ir.add_block(block);
            ir.recompute_uses();
            AddressWalk::new(&ir, PointerSize::Bit32).normalize(address)
        };
        assert_eq!(
            decode(ConstValue::U32(0xFFFF_FFF8)).offset_bits,
            decode(ConstValue::I32(-8)).offset_bits
        );
    }

    /// A value with no recorded key falls back to its own identity key.
    #[test]
    fn alias_key_for_value_falls_back_to_identity() {
        let keys = BTreeMap::new();
        let value = SsaVarId::from_index(3);
        assert_eq!(
            alias_key_for_value(value, &keys),
            Some(AliasKey {
                base: AddressBase::Value(value),
                index: None,
                stride_bytes: 0,
                offset_bits: 0,
            })
        );
    }

    /// An array-element address gets an alias key carrying its scaled index and
    /// stride, so the cell keeps a stable identity.
    #[test]
    fn alias_key_records_scaled_index_cell() {
        // `base + index*8` is an array-element address; the alias key records
        // its scaled index and stride so the cell keeps a stable identity.
        let mut ir = function();
        let base = var(&mut ir, VariableOrigin::Argument(0));
        let index = var(&mut ir, VariableOrigin::Argument(1));
        let stride = ir.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::Unknown,
        );
        let scaled = ir.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::Unknown,
        );
        let addr = ir.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::Unknown,
        );
        let mut block = SsaBlock::with_capacity(0, 0, 3);
        block.add_instruction(instr(SsaOp::Const {
            dest: stride,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(instr(SsaOp::Mul {
            dest: scaled,
            left: index,
            right: stride,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: addr,
            left: base,
            right: scaled,
            flags: None,
        }));
        ir.add_block(block);
        ir.recompute_uses();

        let keys = AddressWalk::new(&ir, PointerSize::Bit64).alias_keys();
        assert_eq!(
            keys.get(&addr.as_u32()),
            Some(&AliasKey {
                base: AddressBase::Value(base),
                index: Some(index.as_u32()),
                stride_bytes: 8,
                offset_bits: 0,
            })
        );
    }

    /// A module symbol keeps its address space when it occupies the index.
    #[test]
    fn a_symbolic_index_becomes_the_symbolic_base() {
        let dest = SsaVarId::from_index(0);
        let base = SsaVarId::from_index(1);
        let index = SsaVarId::from_index(2);
        let constants = BTreeMap::from([(index, ConstValue::<MockTarget>::Symbol(0x3040))]);
        for size in [PointerSize::Bit32, PointerSize::Bit64] {
            let folded =
                fold_constant_index(&ptr_add(dest, base, Some(index), 1, 7), &constants, size);
            assert_eq!(folded, Some(ptr_add(dest, index, Some(base), 1, 7)));
            assert_eq!(
                fold_constant_index(&ptr_add(dest, base, Some(index), 4, 7), &constants, size),
                None,
                "multiplying a module coordinate cannot become a numeric displacement"
            );
        }
        let both = BTreeMap::from([
            (base, ConstValue::<MockTarget>::Symbol(0x100)),
            (index, ConstValue::<MockTarget>::Symbol(0x3040)),
        ]);
        assert_eq!(
            fold_constant_index(
                &ptr_add(dest, base, Some(index), 1, 0),
                &both,
                PointerSize::Bit64
            ),
            None,
            "two symbolic terms must not oscillate between base and index"
        );
    }

    /// A symbol, a slot, a byte index and a wrapped offset normalize to their
    /// literal forms.
    #[test]
    fn canonical_address_walk_matches_independent_literal_oracles() {
        let mut ir = function();
        let input = var(&mut ir, VariableOrigin::Argument(0));
        let symbol = var(&mut ir, VariableOrigin::Local(0));
        let shift = var(&mut ir, VariableOrigin::Local(1));
        let displaced = var(&mut ir, VariableOrigin::Local(2));
        let scaled = var(&mut ir, VariableOrigin::Local(3));
        let indexed = var(&mut ir, VariableOrigin::Local(4));
        let slot = var(&mut ir, VariableOrigin::Local(5));
        let slot_indexed = var(&mut ir, VariableOrigin::Local(6));
        let number = var(&mut ir, VariableOrigin::Local(7));
        let wrapped = var(&mut ir, VariableOrigin::Local(8));
        let mut block = SsaBlock::new(0);
        for operation in [
            SsaOp::Const {
                dest: symbol,
                value: ConstValue::Symbol(0x200),
            },
            SsaOp::Const {
                dest: shift,
                value: ConstValue::I32(2),
            },
            ptr_add(displaced, symbol, None, 0, -8),
            SsaOp::Shl {
                dest: scaled,
                value: input,
                amount: shift,
                flags: None,
            },
            SsaOp::Add {
                dest: indexed,
                left: displaced,
                right: scaled,
                flags: None,
            },
            SsaOp::LoadLocalAddr {
                dest: slot,
                local_index: 3,
            },
            SsaOp::Add {
                dest: slot_indexed,
                left: slot,
                right: input,
                flags: None,
            },
            SsaOp::Const {
                dest: number,
                value: ConstValue::U32(0x1000),
            },
            ptr_add(wrapped, number, None, 0, 0xFFFF_FFF8),
        ] {
            block.add_instruction(instr(operation));
        }
        ir.add_block(block);
        for (value, pointer, expected) in [
            (
                indexed,
                PointerSize::Bit64,
                AddressExpr {
                    base: AddressBase::Symbol,
                    root: symbol,
                    root_offset_bits: -64,
                    index: Some(input),
                    stride_bytes: 4,
                    offset_bits: 4032,
                },
            ),
            (
                slot_indexed,
                PointerSize::Bit64,
                AddressExpr {
                    base: AddressBase::Local(3),
                    root: slot,
                    root_offset_bits: 0,
                    index: Some(input),
                    stride_bytes: 1,
                    offset_bits: 0,
                },
            ),
            (
                wrapped,
                PointerSize::Bit32,
                AddressExpr {
                    base: AddressBase::Number,
                    root: number,
                    root_offset_bits: -64,
                    index: None,
                    stride_bytes: 0,
                    offset_bits: 32704,
                },
            ),
        ] {
            assert_eq!(AddressWalk::new(&ir, pointer).normalize(value), expected);
        }
    }

    /// A scalar query keeps the full signed 64-bit domain.
    #[test]
    fn canonical_address_scalar_queries_preserve_the_full_integer_domain() {
        let mut ir = function();
        let high = var(&mut ir, VariableOrigin::Local(0));
        let low = var(&mut ir, VariableOrigin::Local(1));
        let input = var(&mut ir, VariableOrigin::Argument(0));
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: high,
            value: ConstValue::I64(i64::MAX),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: low,
            value: ConstValue::I64(i64::MIN),
        }));
        ir.add_block(block);
        for (value, expected) in [(high, Some(i64::MAX)), (low, Some(i64::MIN)), (input, None)] {
            assert_eq!(ir.const_i64(value), expected);
        }
    }
}
