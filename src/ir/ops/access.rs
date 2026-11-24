//! Memory accesses with the address they go through.
//!
//! [`SsaOp::memory_effect`] answers for an operation with one address and one
//! value. [`AddressedAccess`] answers for every operation that carries an
//! address: which address, what moved through it and how wide, in which
//! direction - both ends of a copy included. Whole-program analyses that follow
//! an address from where it is formed to where it is used read both this and
//! [`SsaOp::steps_an_address`], which names the operations that only carry an
//! address along.

use crate::{
    ir::{
        ops::{
            SsaOp,
            kinds::{AccessQualifiers, CompareBlkData},
        },
        variable::SsaVarId,
    },
    target::Target,
};

/// Calls `$visit` with each access qualifier `$op` carries: once for an access,
/// twice for a copy (its source's, then its destination's), and never for an
/// operation that touches no memory through qualifiers. One list serves the
/// shared and the mutable visit, which bind by the scrutinee's reference.
macro_rules! each_qualifier {
    ($op:expr, $visit:expr $(, $mutability:tt)?) => {
        match $op {
            SsaOp::CopyBlk {
                src_access,
                dest_access,
                ..
            }
            | SsaOp::MoveBlk {
                src_access,
                dest_access,
                ..
            } => {
                $visit(src_access);
                $visit(dest_access);
            }
            SsaOp::CompareBlk(data) => {
                let CompareBlkData {
                    left_access,
                    right_access,
                    ..
                } = &$($mutability)? **data;
                $visit(left_access);
                $visit(right_access);
            }
            SsaOp::VectorLoad { access, .. }
            | SsaOp::VectorStore { access, .. }
            | SsaOp::VectorMaskedLoad { access, .. }
            | SsaOp::VectorMaskedStore { access, .. }
            | SsaOp::VectorBroadcastLoad { access, .. }
            | SsaOp::VectorGather { access, .. }
            | SsaOp::VectorFaultingLoad { access, .. }
            | SsaOp::VectorSegmentLoad { access, .. }
            | SsaOp::VectorScatter { access, .. }
            | SsaOp::VectorSegmentStore { access, .. }
            | SsaOp::VectorPackLoad { access, .. }
            | SsaOp::VectorPackStore { access, .. }
            | SsaOp::LoadIndirect { access, .. }
            | SsaOp::StoreIndirect { access, .. }
            | SsaOp::LoadField { access, .. }
            | SsaOp::StoreField { access, .. }
            | SsaOp::LoadStaticField { access, .. }
            | SsaOp::StoreStaticField { access, .. }
            | SsaOp::LoadObj { access, .. }
            | SsaOp::StoreObj { access, .. }
            | SsaOp::InitBlk { access, .. }
            | SsaOp::ScanBlk { access, .. }
            | SsaOp::LoadBlk { access, .. }
            | SsaOp::AtomicLoad { access, .. }
            | SsaOp::AtomicStore { access, .. }
            | SsaOp::AtomicStoreConditional { access, .. }
            | SsaOp::AtomicPairLoad { access, .. }
            | SsaOp::AtomicPairStoreConditional { access, .. }
            | SsaOp::AtomicExchange { access, .. }
            | SsaOp::AtomicLockRmw { access, .. }
            | SsaOp::AtomicCondRmw { access, .. }
            | SsaOp::AtomicCmpXchg { access, .. }
            | SsaOp::AtomicPairCmpXchg { access, .. }
            | SsaOp::AtomicPairRmw { access, .. }
            | SsaOp::AtomicCheckedRmw { access, .. } => $visit(access),
            _ => {}
        }
    };
}

/// One memory access through an address an operation names.
///
/// Every operation that touches memory reduces to this - which address, what
/// moved through it, how wide, in which direction - so a walk over a function
/// asks one question of one shape instead of growing an arm per opcode. That is what the
/// vocabulary drifted out of before: recovery understood `LoadIndirect` and
/// `StoreIndirect` and nothing else, so an SSE store, an atomic increment, a
/// value-type copy and a `rep stos` to a constant address each contributed no
/// observation at all, silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressedAccess {
    /// The address operand.
    pub addr: SsaVarId,
    /// How the access is made: two accesses to one address in different
    /// address spaces reach different memory.
    pub qualifiers: AccessQualifiers,
    /// The value moved, when the operation names one first-class value.
    ///
    /// Its recovered type is the evidence for the object's own type. `None`
    /// where the operation moves an opaque region (a block copy, an object
    /// initialiser) or moves a *pair*, where neither half describes the object.
    pub value: Option<SsaVarId>,
    /// The access width in bits, when the operation states one.
    ///
    /// From the operation, not from the recovered type: the instruction moved
    /// that many bits whatever the propagation later concludes.
    pub width_bits: Option<u32>,
    /// Whether the access reads the object.
    pub reads: bool,
    /// Whether it writes the object.
    pub writes: bool,
}

impl<T: Target> SsaOp<T> {
    /// The direction flag a block operation reads (x86's `DF`), where it
    /// reads one: set, the operation walks from high addresses to low. `None`
    /// for an operation that walks upward by definition or is no block
    /// operation.
    #[must_use]
    pub fn direction_flag(&self) -> Option<SsaVarId> {
        match self {
            Self::CopyBlk { backward, .. }
            | Self::InitBlk { backward, .. }
            | Self::ScanBlk { backward, .. }
            | Self::LoadBlk { backward, .. } => *backward,
            Self::CompareBlk(data) => data.backward,
            _ => None,
        }
    }

    /// The direction flag operand of a block operation that may read one
    /// ([`Self::direction_flag`]), to restate it; `None` for any other
    /// operation.
    pub fn direction_flag_mut(&mut self) -> Option<&mut Option<SsaVarId>> {
        match self {
            Self::CopyBlk { backward, .. }
            | Self::InitBlk { backward, .. }
            | Self::ScanBlk { backward, .. }
            | Self::LoadBlk { backward, .. } => Some(backward),
            Self::CompareBlk(data) => Some(&mut data.backward),
            _ => None,
        }
    }

    /// Visits the qualifiers of every memory access this operation makes
    /// ([`AccessQualifiers`]): one for an access, a copy's source's then its
    /// destination's, none for an operation that makes no qualified access.
    pub fn visit_qualifiers(&self, mut visit: impl FnMut(&AccessQualifiers)) {
        each_qualifier!(self, visit);
    }

    /// Visits the qualifiers of every memory access this operation makes, to
    /// restate them ([`Self::visit_qualifiers`]): a CIL `volatile.` prefix
    /// makes the access it prefixes volatile.
    pub fn visit_qualifiers_mut(&mut self, mut visit: impl FnMut(&mut AccessQualifiers)) {
        each_qualifier!(self, visit, mut);
    }

    /// Visits every memory access this operation performs.
    ///
    /// Most operations perform none and are skipped by the catch-all; the ones
    /// listed are every variant of the op vocabulary that carries an address
    /// operand. A copy performs **two** - one at each end - which is why this
    /// visits rather than returns. An access's width is the target's
    /// ([`Target::bit_width`]) where a type states it
    /// ([`Self::for_each_addressed_access_by`]).
    pub fn for_each_addressed_access(&self, visit: impl FnMut(AddressedAccess)) {
        self.for_each_addressed_access_by(|ty| T::bit_width(ty), visit);
    }

    /// Visits every memory access this operation performs, as
    /// [`Self::for_each_addressed_access`] does, with the width of an access
    /// a type states read by `type_bits`: a reader that reads no type (one
    /// answering `None`, as an execution that reads types only from its own
    /// layouts passes) leaves those widths unknown and inspects no type, while
    /// a width the operation states apart from a type (an atomic's machine
    /// width, a state image's) is known still.
    pub fn for_each_addressed_access_by(
        &self,
        type_bits: impl Fn(&T::Type) -> Option<u32>,
        mut visit: impl FnMut(AddressedAccess),
    ) {
        let access = |space: AccessQualifiers,
                      addr: SsaVarId,
                      value: Option<SsaVarId>,
                      width_bits: Option<u32>,
                      reads: bool,
                      writes: bool| AddressedAccess {
            addr,
            qualifiers: space,
            value,
            width_bits,
            reads,
            writes,
        };
        let read =
            |space, addr, value, width_bits| access(space, addr, value, width_bits, true, false);
        let write =
            |space, addr, value, width_bits| access(space, addr, value, width_bits, false, true);
        let modify =
            |space, addr, value, width_bits| access(space, addr, value, width_bits, true, true);
        match self {
            SsaOp::LoadIndirect {
                dest,
                addr,
                value_type,
                access: space,
            } => visit(read(*space, *addr, Some(*dest), type_bits(value_type))),
            SsaOp::StoreIndirect {
                addr,
                value,
                value_type,
                access: space,
            } => visit(write(*space, *addr, Some(*value), type_bits(value_type))),

            // Vector forms. The value is the whole vector, and its recovered
            // type describes the object as truly as a scalar's does.
            SsaOp::VectorLoad {
                dest,
                addr,
                vector_type,
                access: space,
            }
            | SsaOp::VectorBroadcastLoad {
                dest,
                addr,
                vector_type,
                access: space,
            }
            | SsaOp::VectorMaskedLoad {
                dest,
                addr,
                vector_type,
                access: space,
                ..
            }
            | SsaOp::VectorFaultingLoad {
                dest,
                addr,
                vector_type,
                access: space,
                ..
            }
            | SsaOp::VectorPackLoad {
                dest,
                addr,
                vector_type,
                access: space,
                ..
            } => visit(read(*space, *addr, Some(*dest), type_bits(vector_type))),
            SsaOp::VectorStore {
                addr,
                value,
                vector_type,
                access: space,
            }
            | SsaOp::VectorMaskedStore {
                addr,
                value,
                vector_type,
                access: space,
                ..
            }
            | SsaOp::VectorPackStore {
                addr,
                value,
                vector_type,
                access: space,
                ..
            } => visit(write(*space, *addr, Some(*value), type_bits(vector_type))),
            // A structure access moves its `segments` vectors through the one
            // run of memory at its base, interleaved or one after another, and
            // no single value is the object's.
            SsaOp::VectorSegmentLoad {
                base,
                vector_type,
                segments,
                access: space,
                ..
            } => visit(read(
                *space,
                *base,
                None,
                type_bits(vector_type).and_then(|bits| bits.checked_mul(*segments)),
            )),
            SsaOp::VectorSegmentStore {
                base,
                vector_type,
                segments,
                access: space,
                ..
            } => visit(write(
                *space,
                *base,
                None,
                type_bits(vector_type).and_then(|bits| bits.checked_mul(*segments)),
            )),

            // Atomics. `width` is the machine access width and is exact, so it
            // is preferred to the value type where both exist.
            SsaOp::AtomicLoad {
                dest,
                addr,
                value_type,
                width,
                access: space,
                ..
            } => visit(read(
                *space,
                *addr,
                Some(*dest),
                width.bits().or_else(|| type_bits(value_type)),
            )),
            SsaOp::AtomicStore {
                addr,
                value,
                value_type,
                width,
                access: space,
                ..
            }
            | SsaOp::AtomicStoreConditional {
                addr,
                value,
                value_type,
                width,
                access: space,
                ..
            } => visit(write(
                *space,
                *addr,
                Some(*value),
                width.bits().or_else(|| type_bits(value_type)),
            )),
            SsaOp::AtomicExchange {
                addr,
                value,
                width,
                access: space,
                ..
            }
            | SsaOp::AtomicLockRmw {
                addr,
                value,
                width,
                access: space,
                ..
            }
            | SsaOp::AtomicCondRmw {
                addr,
                value,
                width,
                access: space,
                ..
            } => visit(modify(*space, *addr, Some(*value), width.bits())),
            SsaOp::AtomicCmpXchg {
                addr,
                desired,
                width,
                access: space,
                ..
            } => visit(modify(*space, *addr, Some(*desired), width.bits())),
            // A pair moves two values through one address; neither half
            // describes the object, so the type channel stays empty and only
            // the width is taken.
            SsaOp::AtomicPairLoad {
                addr,
                width,
                access: space,
                ..
            } => visit(read(*space, *addr, None, width.bits())),
            SsaOp::AtomicPairStoreConditional {
                addr,
                width,
                access: space,
                ..
            } => {
                visit(write(*space, *addr, None, width.bits()));
            }
            SsaOp::AtomicPairCmpXchg {
                addr,
                width,
                access: space,
                ..
            }
            | SsaOp::AtomicPairRmw {
                addr,
                width,
                access: space,
                ..
            }
            | SsaOp::AtomicCheckedRmw {
                addr,
                width,
                access: space,
                ..
            } => {
                visit(modify(*space, *addr, None, width.bits()));
            }

            // Value-type copies, which a managed runtime makes in its one
            // space. The type names the object exactly, but as a type
            // reference rather than a value type, so it establishes the access
            // without establishing a width.
            SsaOp::LoadObj { dest, src_addr, .. } => {
                visit(read(AccessQualifiers::FLAT, *src_addr, Some(*dest), None));
            }
            SsaOp::StoreObj {
                dest_addr, value, ..
            } => visit(write(
                AccessQualifiers::FLAT,
                *dest_addr,
                Some(*value),
                None,
            )),
            SsaOp::InitObj { dest_addr, .. } => {
                visit(write(AccessQualifiers::FLAT, *dest_addr, None, None));
            }
            SsaOp::CopyObj {
                dest_addr,
                src_addr,
                ..
            } => {
                visit(write(AccessQualifiers::FLAT, *dest_addr, None, None));
                visit(read(AccessQualifiers::FLAT, *src_addr, None, None));
            }

            // Block moves. The region is opaque - the size is a value, not a
            // constant this can read - so they establish the object and its
            // direction and nothing else.
            SsaOp::InitBlk {
                dest_addr,
                access: space,
                ..
            } => visit(write(*space, *dest_addr, None, None)),
            SsaOp::CopyBlk {
                dest_addr,
                src_addr,
                src_access: src_space,
                dest_access: dest_space,
                ..
            }
            | SsaOp::MoveBlk {
                dest_addr,
                src_addr,
                src_access: src_space,
                dest_access: dest_space,
                ..
            } => {
                visit(write(*dest_space, *dest_addr, None, None));
                visit(read(*src_space, *src_addr, None, None));
            }
            // Block reads, the first block's first.
            SsaOp::CompareBlk(data) => {
                visit(read(data.left_access, data.left, None, None));
                visit(read(data.right_access, data.right, None, None));
            }
            SsaOp::ScanBlk {
                address, access, ..
            }
            | SsaOp::LoadBlk {
                address, access, ..
            } => visit(read(*access, *address, None, None)),

            // A state image moves at the address its first input is, as many
            // bytes as its layout holds where the layout fixes them.
            SsaOp::FpuControl(data) if data.kind.moves_image() => {
                if let Some(&addr) = data.inputs.first() {
                    let width_bits = data.kind.image_bytes().map(|bytes| bytes.saturating_mul(8));
                    let kind = data.kind;
                    visit(access(
                        AccessQualifiers::FLAT,
                        addr,
                        None,
                        width_bits,
                        kind.restores_image(),
                        kind.saves_image(),
                    ));
                }
            }

            // Everything else names no address.
            _ => {}
        }
    }

    /// Returns whether this operation only steps an address along.
    ///
    /// Such an operation converts an address between integer and pointer,
    /// copies it, or offsets it. Its result is the address it was handed,
    /// moved; whatever takes the address is a use of the result, not of this.
    #[must_use]
    pub const fn steps_an_address(&self) -> bool {
        matches!(
            self,
            SsaOp::PtrToInt { .. }
                | SsaOp::IntToPtr { .. }
                | SsaOp::PtrAdd { .. }
                | SsaOp::Copy { .. }
                | SsaOp::Add { .. }
                | SsaOp::Sub { .. }
                | SsaOp::IntConv { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU16;

    use super::*;
    use crate::{
        ir::ops::{
            AccessPermission, AtomicAccessWidth, AtomicOrdering, AtomicRmwOp, VectorSegmentLayout,
        },
        testing::{MockTarget, MockType},
    };

    /// Collects the accesses `op` performs.
    fn accesses(op: &SsaOp<MockTarget>) -> Vec<AddressedAccess> {
        let mut found = Vec::new();
        op.for_each_addressed_access(|access| found.push(access));
        found
    }

    /// **An access's qualifiers are visited where they are stated.** A copy
    /// states its source's and its destination's, a load its one, and an
    /// operation that makes no qualified access none; restating one through the
    /// mutable visit (a CIL `volatile.`) makes the operation volatile.
    #[test]
    fn an_access_qualifier_is_visited_where_it_is_stated() {
        let (a, b, c) = (
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
            SsaVarId::from_index(3),
        );
        let count = |op: &SsaOp<MockTarget>| {
            let mut seen = 0;
            op.visit_qualifiers(|_| seen += 1);
            seen
        };
        let mut copy = SsaOp::<MockTarget>::CopyBlk {
            dest_addr: a,
            src_addr: b,
            size: c,
            backward: None,
            dest_access: AccessQualifiers::FLAT,
            src_access: AccessQualifiers::FLAT,
            element_bytes: 1,
        };
        let mut load = SsaOp::<MockTarget>::LoadIndirect {
            dest: a,
            addr: b,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        };
        let add = SsaOp::<MockTarget>::Add {
            dest: a,
            left: b,
            right: c,
            flags: None,
        };
        assert_eq!((count(&copy), count(&load), count(&add)), (2, 1, 0));
        assert!(!load.is_volatile());
        load.visit_qualifiers_mut(|qualifiers| qualifiers.volatile = true);
        assert!(load.is_volatile() && load.effects().volatile);
        let mut visited = 0;
        copy.visit_qualifiers_mut(|qualifiers| {
            visited += 1;
            if visited == 2 {
                qualifiers.volatile = true;
            }
        });
        assert!(
            matches!(copy, SsaOp::CopyBlk { dest_access, src_access, .. }
                if dest_access.volatile && !src_access.volatile),
            "the second visited is the destination"
        );
    }

    /// A block copy touches memory at both ends - which `memory_effect`, with
    /// room for one address, declines to describe - and a load at one, with
    /// the value it moves and the width its type states.
    #[test]
    fn a_copy_is_two_accesses_and_a_load_one() {
        let (dest, src, size) = (
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
            SsaVarId::from_index(3),
        );
        for op in [
            SsaOp::CopyBlk {
                dest_addr: dest,
                src_addr: src,
                size,
                element_bytes: 1,
                backward: None,
                dest_access: AccessQualifiers::FLAT,
                src_access: AccessQualifiers::FLAT,
            },
            SsaOp::MoveBlk {
                dest_addr: dest,
                src_addr: src,
                size,
                dest_access: AccessQualifiers::FLAT,
                src_access: AccessQualifiers::FLAT,
            },
        ] {
            let copy = accesses(&op);
            assert_eq!(copy.len(), 2, "{op}");
            assert!(copy.iter().any(|a| a.addr == dest && a.writes && !a.reads));
            assert!(copy.iter().any(|a| a.addr == src && a.reads && !a.writes));
            assert!(op.memory_effect().is_none(), "{op}");
        }

        let loaded = SsaVarId::from_index(4);
        let load = accesses(&SsaOp::LoadIndirect {
            dest: loaded,
            addr: src,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        });
        assert_eq!(
            load,
            vec![AddressedAccess {
                addr: src,
                value: Some(loaded),
                width_bits: Some(32),
                reads: true,
                writes: false,
                qualifiers: AccessQualifiers::FLAT,
            }]
        );
        // Control: an operation that carries no address performs none.
        assert!(accesses(&SsaOp::Copy { dest: loaded, src }).is_empty());
    }

    /// A structure access is one access at its base, over every segment it
    /// moves: `ld2 {v0.4s, v1.4s}, [x0]` reads 256 bits from `x0`, `st2` writes
    /// them, and neither names one value as the object's.
    #[test]
    fn a_structure_access_reads_or_writes_every_segment_at_its_base() {
        let (base, first, second) = (
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
            SsaVarId::from_index(3),
        );
        let load = accesses(&SsaOp::VectorSegmentLoad {
            dests: vec![first, second],
            base,
            mask: None,
            vector_type: MockType::V4I32,
            segments: 2,
            layout: VectorSegmentLayout::Interleaved,
            access: AccessQualifiers::FLAT,
        });
        assert_eq!(
            load,
            vec![AddressedAccess {
                addr: base,
                value: None,
                width_bits: Some(256),
                reads: true,
                writes: false,
                qualifiers: AccessQualifiers::FLAT,
            }]
        );
        let store = accesses(&SsaOp::VectorSegmentStore {
            base,
            values: vec![first, second],
            mask: None,
            vector_type: MockType::V4I32,
            segments: 2,
            layout: VectorSegmentLayout::Consecutive,
            access: AccessQualifiers::FLAT,
        });
        assert_eq!(
            store,
            vec![AddressedAccess {
                addr: base,
                value: None,
                width_bits: Some(256),
                reads: false,
                writes: true,
                qualifiers: AccessQualifiers::FLAT,
            }]
        );
    }

    /// Every access names the space it lands in: an atomic through `fs:` its
    /// segment's, each end of a copy its own, and a managed object copy the
    /// one flat space. Without it a walk over accesses would read a
    /// segmented atomic and a flat access to the same address as one object.
    #[test]
    fn every_access_names_its_space() {
        let (addr, value, size, old) = (
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
            SsaVarId::from_index(3),
            SsaVarId::from_index(4),
        );
        let segment = AccessQualifiers::new(NonZeroU16::new(1), AccessPermission::Current);
        let user = AccessQualifiers::new(None, AccessPermission::Unprivileged);
        let rmw = accesses(&SsaOp::AtomicLockRmw {
            dest: old,
            addr,
            value,
            op: AtomicRmwOp::Add,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            access: segment,
        });
        assert_eq!(
            rmw.iter()
                .map(|access| access.qualifiers)
                .collect::<Vec<_>>(),
            [segment]
        );
        let copy = accesses(&SsaOp::CopyBlk {
            dest_addr: addr,
            src_addr: value,
            size,
            element_bytes: 1,
            backward: None,
            src_access: segment,
            dest_access: user,
        });
        assert!(
            copy.iter()
                .any(|access| access.writes && access.qualifiers == user)
        );
        assert!(
            copy.iter()
                .any(|access| access.reads && access.qualifiers == segment)
        );
        let object = accesses(&SsaOp::<MockTarget>::InitObj {
            dest_addr: addr,
            value_type: 0,
        });
        assert_eq!(
            object
                .iter()
                .map(|access| access.qualifiers)
                .collect::<Vec<_>>(),
            [AccessQualifiers::FLAT]
        );
    }
}
