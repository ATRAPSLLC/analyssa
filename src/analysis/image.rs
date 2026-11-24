//! What a host knows about the image a program was loaded from: where its
//! regions are, whether each may be written, and the bytes they start with.
//!
//! [`Target::symbol_address`] answers
//! *where* a symbol is; [`ImageData`] answers *what is there*. Both are host
//! knowledge an analysis cannot derive, and a host with nothing to say
//! supplies no image at all, in which case every analysis that could read one
//! behaves as though the program's memory were unknown - which it is.
//!
//! The concept is not native-specific. A static readonly array initialised from
//! a field RVA in a CIL module is initialised data at a stable address, and it
//! is exactly where an obfuscator keeps an encrypted string table.
//!
//! # What an answer promises
//!
//! A region the oracle reports [`Mutability::ReadOnly`] holds its bytes for the
//! whole run: a write to it faults, so no execution continues past one. That is
//! what lets an analysis read those bytes at any point of any function without
//! asking what ran before it. A host must therefore not report a region
//! read-only when the program can change it - data the loader relocates, an
//! import table the loader binds, anything the program remaps writable - and a
//! region whose initial contents are not the file's bytes (zero-initialised
//! storage, say) must not be served at all.

use std::{
    fmt,
    ops::{Deref, Range},
};

use crate::{
    ir::{instruction::InstrId, value::ConstValue},
    target::{Endianness, Target},
};

/// Whether the program may write a region of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    /// Never written: the bytes the image holds are the bytes every read sees.
    ReadOnly,
    /// May be written, so its bytes are only its contents before any write.
    Writable,
}

/// One region of the image: a half-open extent and whether it may be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageRegion {
    /// First address of the region, in the host's symbol space.
    pub start: u64,
    /// One past its last address.
    pub end: u64,
    /// Whether the program may write it.
    pub mutability: Mutability,
}

impl ImageRegion {
    /// Returns `true` when `[address, address + len)` lies wholly inside.
    #[must_use]
    pub fn covers(&self, address: u64, len: u64) -> bool {
        address >= self.start && address.checked_add(len).is_some_and(|end| end <= self.end)
    }
}

/// An opaque identity of an immutable image provider's complete query universe.
///
/// The provider derives this identity from domain-separated canonical contents,
/// mapping and semantic revisions. Equal identities attest equal positive and
/// negative answers and the same lifetime promises for every [`ImageData`] query.
/// An identity certifies the provider's contract, not the physical running target,
/// unconditional execution truth, or permission to reuse an analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageSnapshotId([u8; 32]);

impl ImageSnapshotId {
    /// Creates an identity from a provider's domain-separated canonical digest.
    ///
    /// Constructing this value does not independently validate the provider's
    /// immutability or completeness contract.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the opaque identity bytes without interpreting their provider domain.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// An image access an analysis replaces or removes.
///
/// Once a load folds, the load is gone, and with it the only statement that
/// the function read the location: a global whose one access was that load is
/// no longer an object anything accesses. The fold therefore hands the host
/// this record ([`Target::record_image_read`]), which keeps it with the
/// function it describes.
///
/// The instruction is named by identity rather than by the value it defines:
/// its position and its destination's id both move as passes edit and compact
/// the function, and an identity is never reissued.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageRead<T: Target> {
    /// The address read, in the host's symbol space.
    pub address: u64,
    /// How many bytes were read.
    pub len: u64,
    /// The source instruction that read them; `None` for an instruction without an identity.
    /// It may define a constant after folding or be bypassed by a loop residual.
    pub instruction: Option<InstrId>,
    /// The host's record of that instruction as it was lifted.
    ///
    /// The identity names the instruction while it lives; the constant it now
    /// defines is propagated into its users and removed, and with it the only
    /// place the host could still read where in the program the load was. A
    /// cross-reference from the reading instruction to the bytes it read
    /// needs that address after the fold.
    pub original: Option<T::OriginalInstruction>,
    /// The scalar load type, repeated-element type or byte type of a block transfer.
    pub value_type: T::Type,
    /// The constant supplied by this exact read on every visit, when one is known.
    /// Ranged block accesses and discarded reads need not produce one constant.
    ///
    /// Kept because the constant's definition need not outlive the read: a
    /// conversion of it folds into a constant of its own, and the first one,
    /// read by nothing, is removed. The value is still what the cell held,
    /// and any constant equal to it in the function is that value.
    pub value: Option<ConstValue<T>>,
    /// Whether control transfers to the value read
    /// ([`SsaFunction::entered_values`](crate::ir::SsaFunction::entered_values)):
    /// the cell a call or branch goes through, which the read folds into the
    /// transfer's target.
    pub entered: bool,
}

/// A host's view of the image a program was loaded from.
///
/// Addressed in the host's **symbol** space - the one
/// [`Target::symbol_address`] answers
/// in - so an address stated as a symbol reads here directly. An address the
/// program states as a bare integer is in the program's own space, which the
/// host relates to its symbol space through [`Self::number_offset`], or does
/// not.
pub trait ImageData: Send + Sync {
    /// Returns a certificate for this provider's complete immutable answer universe.
    ///
    /// A returned identity covers every positive and negative answer from
    /// [`Self::endianness`], [`Self::region`], [`Self::read_only_overlap`],
    /// [`Self::bytes`], [`Self::number_offset`] and [`Self::address_at`], including
    /// the meaning and lifetime of its mutability promises. Those answers must stay
    /// unchanged for this provider's lifetime. The identity must bind its canonical
    /// data and query semantics; changing either requires a different identity.
    ///
    /// The default is uncertified. Owning a provider through an `Arc`, observing
    /// equal digests before and after execution, or attaching an external label
    /// does not establish this contract. Wrappers that add answers need their own
    /// whole-universe certificate; forwarding only a base identity is insufficient.
    /// Runtime writes still belong to execution state. This certificate alone
    /// neither establishes physical target truth nor authorizes unconditional reuse.
    #[must_use]
    fn snapshot_identity(&self) -> Option<ImageSnapshotId> {
        None
    }

    /// The byte order the image's multi-byte values are stored in.
    fn endianness(&self) -> Endianness;

    /// The region `address` lies in, or `None` when it lies in no region the
    /// host will serve.
    ///
    /// # Arguments
    ///
    /// * `address` - An address in the host's symbol space.
    fn region(&self, address: u64) -> Option<ImageRegion>;

    /// Whether the loader maps all `len` bytes at `address` before the
    /// program runs, so no read of them faults - whether or not the host
    /// serves their bytes. A host maps more than it serves: code, whose bytes
    /// a program may rewrite, and zero-filled storage are mapped and never
    /// served. The default is the served regions ([`Self::region`]).
    ///
    /// # Arguments
    ///
    /// * `address` - An address in the host's symbol space.
    /// * `len` - How many bytes.
    #[must_use]
    fn mapped(&self, address: u64, len: u64) -> bool {
        self.region(address)
            .is_some_and(|region| region.covers(address, len))
    }

    /// Returns whether the linear span intersects storage promised immutable for the whole run.
    ///
    /// The span ends at the last representable address rather than wrapping.
    /// Empty spans touch no storage. The default queries each address because
    /// an oracle may refine a writable region with smaller immutable extents;
    /// hosts with an ordered region index can answer without a byte walk.
    #[must_use]
    fn read_only_overlap(&self, address: u64, len: u64) -> bool {
        for offset in 0..len {
            let Some(at) = address.checked_add(offset) else {
                break;
            };
            if self
                .region(at)
                .is_some_and(|region| region.mutability == Mutability::ReadOnly)
            {
                return true;
            }
        }
        false
    }

    /// The `len` bytes starting at `address`, or `None` when the host cannot
    /// serve all of them.
    ///
    /// # Arguments
    ///
    /// * `address` - An address in the host's symbol space.
    /// * `len` - How many bytes.
    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]>;

    /// What to add to an address the program states as a bare integer to reach
    /// the host's symbol space, or `None` when the host does not know.
    ///
    /// `Some(0)` when the two coincide. A host whose symbols are measured from
    /// the module base while the program computes with loaded addresses
    /// answers the negated base. The default is `None`: an integer address then
    /// reads nothing, which costs precision and never a wrong answer.
    fn number_offset(&self) -> Option<i64> {
        None
    }

    /// The address the `len`-byte cell at `address` holds, in the host's
    /// symbol space, when the host proves the cell holds one.
    ///
    /// Bytes say what a cell holds, not what it *is*: `0x00400000` in a global
    /// offset table entry and `0x00400000` in a counter read the same. A host
    /// that knows better - the loader relocates the cell, or the ABI defines
    /// the table it sits in as a table of addresses - says so here, and a load
    /// of it folds to the location it names rather than to a number. `None` by
    /// default, and a cell then folds to its bytes' value.
    ///
    /// # Arguments
    ///
    /// * `address` - The cell's first address, in the host's symbol space.
    /// * `len` - Its width in bytes.
    fn address_at(&self, address: u64, len: u64) -> Option<u64> {
        let _ = (address, len);
        None
    }
}

/// What a host has proven about memory beyond what its image says: writable
/// extents nothing writes, and words outside the image it can serve.
///
/// The facts behind a [`ReadOnlyOverlay`].
pub trait ReadOnlyExtents: Send + Sync {
    /// The extent holding `address` that the host proved nothing writes, if
    /// any.
    ///
    /// # Arguments
    ///
    /// * `address` - An address in the host's symbol space.
    fn sealed_extent(&self, address: u64) -> Option<Range<u64>>;

    /// A word outside every region of the image that the host serves anyway,
    /// read-only, with its bytes: a literal the loader leaves in place.
    ///
    /// # Arguments
    ///
    /// * `address` - An address in the host's symbol space.
    fn served_word(&self, address: u64) -> Option<(Range<u64>, &[u8])>;

    /// The address the `len`-byte cell at `address` holds, for a served word
    /// that holds one; see [`ImageData::address_at`].
    ///
    /// # Arguments
    ///
    /// * `address` - The cell's first address.
    /// * `len` - Its width in bytes.
    fn served_address(&self, address: u64, len: u64) -> Option<u64>;

    /// The identity of an image whose identity is `base` with these extents
    /// served over it: a certificate, per [`ImageData::snapshot_identity`],
    /// covering every answer the base gives and every answer these extents
    /// add. `None`, the default, leaves the overlay uncertified.
    ///
    /// # Arguments
    ///
    /// * `base` - The certified identity of the image beneath.
    fn overlay_identity(&self, base: ImageSnapshotId) -> Option<ImageSnapshotId> {
        let _ = base;
        None
    }
}

/// An image with a host's proven extents served read-only.
///
/// A load from writable memory folds only to the bytes the memory held before
/// any write, so it does not fold at all; once the host has proven an extent
/// unwritten, its bytes are what every read sees, and the overlay serves it as
/// read-only. Words the host serves outside the image are added the same way.
///
/// It holds both as handles: borrowed (`&dyn ImageData`, `&E`) over a lift
/// that runs while the host's image does, shared (`Arc`) to go on with the
/// functions normalized in it once the host's image is gone.
pub struct ReadOnlyOverlay<I, E> {
    /// The image as the host serves it.
    image: I,
    /// The proven extents.
    extents: E,
}

impl<I, E> ReadOnlyOverlay<I, E>
where
    I: Deref,
    I::Target: ImageData,
    E: Deref,
    E::Target: ReadOnlyExtents,
{
    /// `image`, with `extents` read-only.
    #[must_use]
    pub const fn new(image: I, extents: E) -> Self {
        Self { image, extents }
    }
}

impl<I, E> fmt::Debug for ReadOnlyOverlay<I, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadOnlyOverlay").finish_non_exhaustive()
    }
}

impl<I, E> ImageData for ReadOnlyOverlay<I, E>
where
    I: Deref + Send + Sync,
    I::Target: ImageData,
    E: Deref + Send + Sync,
    E::Target: ReadOnlyExtents,
{
    /// Certified only when the image beneath is and the extents certify what
    /// they add over it ([`ReadOnlyExtents::overlay_identity`]).
    fn snapshot_identity(&self) -> Option<ImageSnapshotId> {
        self.extents
            .overlay_identity(self.image.snapshot_identity()?)
    }

    fn endianness(&self) -> Endianness {
        self.image.endianness()
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        let Some(region) = self.image.region(address) else {
            return self
                .extents
                .served_word(address)
                .map(|(extent, _)| ImageRegion {
                    start: extent.start,
                    end: extent.end,
                    mutability: Mutability::ReadOnly,
                });
        };
        if region.mutability != Mutability::Writable {
            return Some(region);
        }
        let Some(extent) = self.extents.sealed_extent(address) else {
            return Some(region);
        };
        Some(ImageRegion {
            start: region.start.max(extent.start),
            end: region.end.min(extent.end),
            mutability: Mutability::ReadOnly,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        if self.image.region(address).is_some() {
            return self.image.bytes(address, len);
        }
        let (extent, bytes) = self.extents.served_word(address)?;
        let end = address.checked_add(u64::try_from(len).ok()?)?;
        if end > extent.end {
            return None;
        }
        let from = usize::try_from(address.checked_sub(extent.start)?).ok()?;
        bytes.get(from..from.checked_add(len)?)
    }

    fn number_offset(&self) -> Option<i64> {
        self.image.number_offset()
    }

    fn address_at(&self, address: u64, len: u64) -> Option<u64> {
        self.image
            .address_at(address, len)
            .or_else(|| self.extents.served_address(address, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-region image: writable data at `0x1000..0x1010`, nothing else.
    struct Data([u8; 16]);

    impl ImageData for Data {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            (0x1000..0x1010).contains(&address).then_some(ImageRegion {
                start: 0x1000,
                end: 0x1010,
                mutability: Mutability::Writable,
            })
        }

        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            let from = usize::try_from(address.checked_sub(0x1000)?).ok()?;
            self.0.get(from..from.checked_add(len)?)
        }
    }

    /// `0x1004..0x1008` proven unwritten, and a served word at `0x2000`.
    struct Proven([u8; 4]);

    impl ReadOnlyExtents for Proven {
        fn sealed_extent(&self, address: u64) -> Option<Range<u64>> {
            (0x1004..0x1008)
                .contains(&address)
                .then_some(0x1004..0x1008)
        }

        fn served_word(&self, address: u64) -> Option<(Range<u64>, &[u8])> {
            (0x2000..0x2004)
                .contains(&address)
                .then_some((0x2000..0x2004, &self.0[..]))
        }

        fn served_address(&self, address: u64, len: u64) -> Option<u64> {
            (address == 0x2000 && len == 4).then_some(0x1000)
        }
    }

    /// A proven extent of writable data reads read-only, clipped to the
    /// extent; the rest stays writable; a served word is read-only data.
    #[test]
    fn proven_extents_are_served_read_only() {
        let image = Data([0; 16]);
        let proven = Proven([1, 2, 3, 4]);
        let overlay = ReadOnlyOverlay::new(&image, &proven);

        let sealed = overlay.region(0x1005).expect("in the image");
        assert_eq!((sealed.start, sealed.end), (0x1004, 0x1008));
        assert_eq!(sealed.mutability, Mutability::ReadOnly);
        assert_eq!(
            overlay.region(0x1000).map(|region| region.mutability),
            Some(Mutability::Writable)
        );
        assert_eq!(overlay.bytes(0x2001, 2), Some(&[2u8, 3][..]));
        assert_eq!(overlay.bytes(0x2002, 4), None, "runs past the word");
        assert_eq!(overlay.address_at(0x2000, 4), Some(0x1000));
    }

    /// A span starting writable can still intersect a smaller immutable overlay or a served word.
    #[test]
    fn readonly_overlap_checks_interior_overlay_and_gap_boundaries() {
        let image = Data([0; 16]);
        let proven = Proven([1, 2, 3, 4]);
        let overlay = ReadOnlyOverlay::new(&image, &proven);
        for (start, len, expected) in [
            (0x1000, 0, false),
            (0x1000, 4, false),
            (0x1000, 5, true),
            (0x1003, 6, true),
            (0x1008, 8, false),
            (0x1ffe, 2, false),
            (0x1ffe, 3, true),
            (0x2003, 1, true),
            (0x2004, 1, false),
        ] {
            assert_eq!(
                overlay.read_only_overlap(start, len),
                expected,
                "{start:#x}+{len}"
            );
        }
        assert!(!image.read_only_overlap(0x1000, 16));
    }

    /// An extent is covered only when both of its ends are inside, and an
    /// extent that would wrap the address space is never covered.
    #[test]
    fn a_region_covers_only_whole_extents() {
        let region = ImageRegion {
            start: 0x1000,
            end: 0x1010,
            mutability: Mutability::ReadOnly,
        };
        assert!(region.covers(0x1000, 16));
        assert!(region.covers(0x100c, 4));
        assert!(!region.covers(0x100d, 4), "runs past the end");
        assert!(!region.covers(0x0ffc, 8), "starts before the region");
        assert!(!region.covers(u64::MAX, 2), "wraps");
    }

    /// A fixed provider whose identity names its complete, immutable test universe.
    struct CertifiedData;

    /// Immutable storage served by the certified fixture.
    static CERTIFIED_BYTES: [u8; 16] = [0; 16];

    impl ImageData for CertifiedData {
        fn snapshot_identity(&self) -> Option<ImageSnapshotId> {
            Some(ImageSnapshotId::from_bytes([0x5a; 32]))
        }
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn region(&self, address: u64) -> Option<ImageRegion> {
            (0x1000..0x1010).contains(&address).then_some(ImageRegion {
                start: 0x1000,
                end: 0x1010,
                mutability: Mutability::Writable,
            })
        }
        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            let start = usize::try_from(address.checked_sub(0x1000)?).ok()?;
            CERTIFIED_BYTES.get(start..start.checked_add(len)?)
        }
    }

    /// An overlay cannot certify its additional answers by forwarding the base certificate.
    #[test]
    fn snapshot_certificates_default_to_none_and_do_not_cross_overlays() {
        assert!(Data([0; 16]).snapshot_identity().is_none());
        let image = CertifiedData;
        let identity = image.snapshot_identity();
        assert_eq!(identity, Some(ImageSnapshotId::from_bytes([0x5a; 32])));
        assert_eq!(identity.map(|id| *id.as_bytes()), Some([0x5a; 32]));
        let proven = Proven([1, 2, 3, 4]);
        let overlay = ReadOnlyOverlay::new(&image, &proven);
        assert!(overlay.snapshot_identity().is_none());
        assert_eq!(overlay.bytes(0x2000, 4), Some([1, 2, 3, 4].as_slice()));
        assert!(image.bytes(0x2000, 4).is_none());
    }
}
