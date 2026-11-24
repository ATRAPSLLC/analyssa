//! How an operation reaches memory: access qualifiers, atomic widths, orderings
//! and read-modify-write operations, fences, and checked read-check-writes.

use std::{
    fmt,
    num::{NonZeroU8, NonZeroU16},
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::{integer::BitWidth, variable::SsaVarId};

/// Memory fence / barrier kind for atomic ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FenceKind {
    /// Full memory barrier
    Full,
    /// Acquire barrier
    Acquire,
    /// Release barrier
    Release,
    /// Acquire+Release barrier
    AcqRel,
    /// Sequentially consistent barrier
    SeqCst,
    /// Orders earlier loads before later loads (x86 `lfence`, RISC-V `fence
    /// r,r`, MIPS `sync 0x13`).
    Loads,
    /// Orders earlier stores before later stores (x86 `sfence`, AArch64 `dmb
    /// ishst`, RISC-V `fence w,w`, MIPS `sync 0x4`).
    Stores,
}

impl fmt::Display for FenceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acqrel"),
            Self::SeqCst => write!(f, "seqcst"),
            Self::Loads => write!(f, "loads"),
            Self::Stores => write!(f, "stores"),
        }
    }
}

impl FenceKind {
    /// Returns the closest atomic ordering represented by this fence: a
    /// fence of loads keeps a later access from moving above an earlier load,
    /// as an acquire does, and one of stores an earlier store from moving
    /// below a later one, as a release does.
    #[must_use]
    pub const fn ordering(self) -> AtomicOrdering {
        match self {
            Self::Full | Self::SeqCst => AtomicOrdering::SeqCst,
            Self::Acquire | Self::Loads => AtomicOrdering::Acquire,
            Self::Release | Self::Stores => AtomicOrdering::Release,
            Self::AcqRel => AtomicOrdering::AcqRel,
        }
    }
}

/// Memory ordering constraint for native atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AtomicOrdering {
    /// No cross-thread ordering beyond atomicity.
    Relaxed,
    /// Acquire ordering for operations that read memory.
    Acquire,
    /// Release ordering for operations that write memory.
    Release,
    /// Acquire and release ordering.
    AcqRel,
    /// Sequentially consistent ordering.
    SeqCst,
}

impl fmt::Display for AtomicOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relaxed => write!(f, "relaxed"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acqrel"),
            Self::SeqCst => write!(f, "seqcst"),
        }
    }
}

/// The permission a native memory access is checked with, apart from the
/// address space it reaches ([`AccessQualifiers`]).
///
/// It changes which accesses fault, never which memory is named, so alias
/// analysis reads none of it: an unprivileged `ldtr` names the bytes a plain
/// `ldr` of its address does. It is carried so an access lowers back to the
/// instruction it lifted from and the source can name it (`__user`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AccessPermission {
    /// The permission of the code running the access.
    #[default]
    Current,
    /// User permission, whatever privilege runs the access: AArch64 `ldtr`
    /// and the FEAT_LSUI atomics, ARM `ldrt`, MIPS EVA `lwe`.
    Unprivileged,
    /// Execute permission standing for read (RISC-V `hlvx`).
    Execute,
    /// Shadow-stack permission, which only a shadow-stack access has: x86
    /// `wrss`, RISC-V `ssamoswap`.
    ShadowStack,
    /// Shadow-stack permission at user privilege (x86 `wruss`).
    UnprivilegedShadowStack,
}

impl AccessPermission {
    /// Returns the IR suffix of this permission, such as `.user`, empty for
    /// [`Self::Current`].
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Current => "",
            Self::Unprivileged => ".user",
            Self::Execute => ".exec",
            Self::ShadowStack => ".shadow",
            Self::UnprivilegedShadowStack => ".user.shadow",
        }
    }
}

/// Whether an access hints that the data it touches will be used again soon:
/// a non-temporal one says it will not, so a cache need not keep it (AArch64
/// `ldnp`, x86 `movntdq`, RISC-V's Zihintntl prefixes). The hint changes
/// neither what is read or written nor where.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Locality {
    /// No hint: the data may be used again.
    #[default]
    Temporal,
    /// The data will not be used again soon.
    NonTemporal,
}

/// The payload of [`SsaOp::CompareBlk`]: two blocks compared element by
/// element, from `left` and `right`, while the elements agree as
/// `while_equal` says and fewer than `count` were compared (x86 `repe cmps`,
/// `repne cmps`). Each compare sets the flags as subtracting the second
/// block's element from the first's.
///
/// [`SsaOp::CompareBlk`]: crate::ir::ops::def::SsaOp::CompareBlk
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompareBlkData {
    /// The elements left uncompared when the comparison stopped.
    pub remaining: SsaVarId,
    /// The address after the last element compared, in the first block.
    pub left_next: SsaVarId,
    /// The address after the last element compared, in the second block.
    pub right_next: SsaVarId,
    /// The flags of the last pair compared, or `flags_in` where none was.
    pub flags: SsaVarId,
    /// The first block's first element.
    pub left: SsaVarId,
    /// The second block's first element.
    pub right: SsaVarId,
    /// The most elements compared.
    pub count: SsaVarId,
    /// The flags an empty comparison leaves; `None` where `count` is never
    /// zero, so no comparison is empty and nothing reads them.
    pub flags_in: Option<SsaVarId>,
    /// The direction flag the comparison reads (x86 `DF`): set, it walks
    /// from high to low addresses.
    pub backward: Option<SsaVarId>,
    /// The width of an element, in bytes.
    pub element_bytes: u32,
    /// Whether the comparison goes on while the elements are equal
    /// (`repe`) rather than while they differ (`repne`).
    pub while_equal: bool,
    /// How the first block is read ([`AccessQualifiers`]).
    pub left_access: AccessQualifiers,
    /// How the second block is read ([`AccessQualifiers`]).
    pub right_access: AccessQualifiers,
}

/// How a memory access is made: where it lands, the permission it is checked
/// with, the locality it hints, whether it is volatile, the alignment its
/// address is guaranteed and the alignment it faults without. Every operation
/// that reads or writes memory carries it, native and managed alike.
///
/// What an access reads or writes is decided by the space and the permission;
/// the locality is a hint, and alias analysis does not read it. A volatile
/// access happens exactly as written: no pass removes it, merges it with
/// another, or forwards a value through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AccessQualifiers {
    /// Address space the access is qualified by, or `None` for the target's
    /// default (flat) space.
    ///
    /// Opaque and host-defined: on x86 a frontend maps a segment override
    /// (`fs:`/`gs:`) to a distinct id here. It qualifies the *access*, not
    /// the address arithmetic - `lea rax, fs:[0x30]` computes `0x30`,
    /// because segmentation is applied on dereference - which is why this
    /// rides the access rather than [`SsaOp::PtrAdd`].
    ///
    /// Alias analysis treats different address spaces as disjoint; without
    /// it, `fs:[0x30]` and a flat `[0x30]` decode to one memory location.
    /// An id is never zero: `None` is the flat space, which no id names
    /// again.
    ///
    /// [`SsaOp::PtrAdd`]: crate::ir::ops::def::SsaOp::PtrAdd
    pub address_space: Option<NonZeroU16>,
    /// The permission the access is checked with.
    pub permission: AccessPermission,
    /// The locality the access hints.
    pub locality: Locality,
    /// Whether the access is volatile: made exactly as written, every time
    /// (CIL `volatile.`, a C `volatile` object).
    pub volatile: bool,
    /// The alignment in bytes the address is guaranteed where it is below the
    /// access's natural alignment (CIL `unaligned.` 1, 2 or 4); `None` for a
    /// naturally aligned access.
    pub alignment: Option<NonZeroU8>,
    /// The alignment in bytes the access faults without, whatever the
    /// machine's rule for ordinary accesses ([`Machine::misaligned`]): x86
    /// `movaps`' 16, a NEON `:128` qualifier's 16, ARMv7 `ldrd`'s 4, an
    /// exclusive's own size; `None` where the machine's rule decides.
    ///
    /// [`Machine::misaligned`]: crate::analysis::Machine::misaligned
    pub required: Option<NonZeroU8>,
}

impl AccessQualifiers {
    /// The target's default space at the running code's permission, with no
    /// locality hint.
    pub const FLAT: Self = Self {
        address_space: None,
        permission: AccessPermission::Current,
        locality: Locality::Temporal,
        volatile: false,
        alignment: None,
        required: None,
    };

    /// Creates the space of an access to `address_space` at `permission`,
    /// with no locality hint.
    #[must_use]
    pub const fn new(address_space: Option<NonZeroU16>, permission: AccessPermission) -> Self {
        Self {
            address_space,
            permission,
            ..Self::FLAT
        }
    }

    /// This space with the access hinting `locality`.
    #[must_use]
    pub const fn with_locality(mut self, locality: Locality) -> Self {
        self.locality = locality;
        self
    }

    /// This access made volatile ([`Self::volatile`]).
    #[must_use]
    pub const fn made_volatile(mut self) -> Self {
        self.volatile = true;
        self
    }

    /// This access with its address guaranteed only `bytes` of alignment
    /// ([`Self::alignment`]).
    #[must_use]
    pub const fn with_alignment(mut self, bytes: NonZeroU8) -> Self {
        self.alignment = Some(bytes);
        self
    }

    /// This access faulting unless its address is aligned to `bytes`
    /// ([`Self::required`]).
    #[must_use]
    pub const fn requiring_alignment(mut self, bytes: NonZeroU8) -> Self {
        self.required = Some(bytes);
        self
    }

    /// Returns whether this is [`Self::FLAT`], which the IR text leaves
    /// unwritten.
    #[must_use]
    pub const fn is_flat(self) -> bool {
        self.address_space.is_none()
            && matches!(self.permission, AccessPermission::Current)
            && matches!(self.locality, Locality::Temporal)
            && !self.volatile
            && self.alignment.is_none()
            && self.required.is_none()
    }
}

impl fmt::Display for AccessQualifiers {
    /// Writes an `addrspace(N)` qualifier, the permission's suffix, a
    /// `nontemporal` hint, `volatile`, an `align(N)` guarantee and an
    /// `aligned(N)` requirement, each followed by a space, or nothing for
    /// [`AccessQualifiers::FLAT`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(space) = self.address_space {
            write!(f, "addrspace({space}) ")?;
        }
        if !matches!(self.permission, AccessPermission::Current) {
            write!(
                f,
                "perm({}) ",
                self.permission.suffix().trim_start_matches('.')
            )?;
        }
        if matches!(self.locality, Locality::NonTemporal) {
            f.write_str("nontemporal ")?;
        }
        if self.volatile {
            f.write_str("volatile ")?;
        }
        if let Some(bytes) = self.alignment {
            write!(f, "align({bytes}) ")?;
        }
        match self.required {
            Some(bytes) => write!(f, "aligned({bytes}) "),
            None => Ok(()),
        }
    }
}

/// Access width for native atomic memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AtomicAccessWidth {
    /// 8-bit atomic access.
    Bits8,
    /// 16-bit atomic access.
    Bits16,
    /// 32-bit atomic access.
    Bits32,
    /// 64-bit atomic access.
    Bits64,
    /// 128-bit atomic access.
    Bits128,
    /// 512-bit single-copy-atomic access of a 64-byte block (AArch64 FEAT_LS64
    /// `ld64b`, `st64b`, `st64bv`, `st64bv0` and their ordered forms).
    Bits512,
    /// Target pointer-sized atomic access.
    Pointer,
}

impl AtomicAccessWidth {
    /// Returns the concrete bit width when it is target-independent.
    #[must_use]
    pub const fn bits(self) -> Option<u32> {
        match self {
            Self::Bits8 => Some(8),
            Self::Bits16 => Some(16),
            Self::Bits32 => Some(32),
            Self::Bits64 => Some(64),
            Self::Bits128 => Some(128),
            Self::Bits512 => Some(512),
            Self::Pointer => None,
        }
    }
}

/// The access of a width in bits; the width itself as the error where no
/// access is that wide.
impl TryFrom<u32> for AtomicAccessWidth {
    type Error = u32;

    fn try_from(bits: u32) -> Result<Self, Self::Error> {
        match bits {
            8 => Ok(Self::Bits8),
            16 => Ok(Self::Bits16),
            32 => Ok(Self::Bits32),
            64 => Ok(Self::Bits64),
            128 => Ok(Self::Bits128),
            512 => Ok(Self::Bits512),
            _ => Err(bits),
        }
    }
}

/// Whether a read-check-write's protected bit is enabled
/// ([`SsaOp::AtomicCheckedRmw`](crate::ir::ops::SsaOp::AtomicCheckedRmw)):
/// Arm's `ProtectionEnabled` of the exception level the access runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RcwProtection {
    /// Enabled: 128-bit translation descriptors are (`IsD128Enabled`), which
    /// a 128-bit form requires.
    Enabled,
    /// As the translation regime configures it, `TCR2_ELx.PnCH` (or
    /// `TCR_EL3.PnCH`) of the regime the access runs in: machine state the
    /// program does not state. A 64-bit form runs only without 128-bit
    /// descriptors, where this is what decides.
    Regime,
}

/// Arm's `RCWCheck` of one read-check-write access: whether a change of a
/// value from `old` to `new` passes the read-check-write check and, for a
/// soft form, the soft check, stated as the NZCV the access returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RcwCheck {
    /// The access width: 64 bits, or 128 for a pair.
    pub bits: u32,
    /// Whether the soft check is made as well (`rcws` forms).
    pub soft: bool,
    /// Whether the protected bit is enabled (`ProtectionEnabled`).
    pub protected: bool,
}

impl RcwCheck {
    /// The NZCV of a passed check: `C` set, the rest clear.
    pub const PASSED: u8 = 0b0010;

    /// The NZCV of a compare-and-swap whose compare failed: `N` set, the
    /// check not made.
    pub const COMPARE_FAILED: u8 = 0b1010;

    /// The protected bit: 52 of a 64-bit value, 114 of a 128-bit one.
    #[must_use]
    pub const fn protected_bit(self) -> u32 {
        if self.bits == 128 { 114 } else { 52 }
    }

    /// The effective mask of `RCWMASK_EL1` (`soft` false) or `RCWSMASK_EL1`
    /// (`soft` true) as `RCWCheck` computes it: its bit 17 (16 of 128 bits)
    /// repeated over the bits above to 49 (55), the bits the format reserves
    /// clear, and the soft mask's protected bit clear where the bit is in use.
    #[must_use]
    pub fn effective_mask(self, register: u128, soft: bool) -> u128 {
        let ones = |high: u32, low: u32| -> u128 {
            let width = high.saturating_sub(low).saturating_add(1);
            BitWidth::new(width).mask_u128() << low
        };
        let bit = |value: u128, at: u32| (value >> at) & 1 == 1;
        if self.bits == 128 {
            let mut mask = register;
            mask = if bit(mask, 16) {
                mask | ones(55, 17)
            } else {
                mask & !ones(55, 17)
            };
            let reserved =
                ones(126, 125) | ones(120, 119) | ones(107, 101) | ones(90, 56) | ones(1, 0);
            mask &= !reserved;
            if soft {
                mask &= !(1u128 << 114);
            }
            mask
        } else {
            let mut mask = register & u128::from(u64::MAX);
            mask = if bit(mask, 17) {
                mask | ones(49, 18)
            } else {
                mask & !ones(49, 18)
            };
            mask &= !1;
            if soft && self.protected {
                mask &= !(1u128 << 52);
            }
            mask
        }
    }

    /// The NZCV `RCWCheck` returns for a change from `old` to `new` under
    /// the registers `rcw_mask` (`RCWMASK_EL1`) and `rcws_mask`
    /// (`RCWSMASK_EL1`, read only for a soft check): `0`, `Z` where the
    /// read-check-write check fails, `C` where the soft check passes, `0`.
    #[must_use]
    pub fn status(self, old: u128, new: u128, rcw_mask: u128, rcws_mask: u128) -> u8 {
        let bit = |value: u128, at: u32| (value >> at) & 1;
        let p = self.protected_bit();
        let changed = old ^ new;
        let mut rcw_fail = false;
        if self.protected {
            // The state check: a protected value keeps its protected and
            // valid bits, an unprotected one its protected bit.
            rcw_fail |= if bit(old, p) == 1 {
                bit(new, p) != bit(old, p) || bit(new, 0) != bit(old, 0)
            } else {
                bit(new, p) != bit(old, p)
            };
            // The mask check, of a valid protected value.
            if bit(old, p) == 1 && bit(old, 0) == 1 {
                rcw_fail |= changed & !self.effective_mask(rcw_mask, false) != 0;
            }
        }
        let mut rcws_fail = false;
        if self.soft {
            if bit(old, 0) == 1 || !self.protected || bit(old, p) == 0 {
                rcws_fail |= bit(new, 0) != bit(old, 0);
            }
            if bit(old, 0) == 1 {
                rcws_fail |= changed & !self.effective_mask(rcws_mask, true) != 0;
            }
        }
        (u8::from(rcw_fail) << 2) | (u8::from(!rcws_fail) << 1)
    }
}

impl fmt::Display for RcwProtection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Enabled => write!(f, "protected"),
            Self::Regime => write!(f, "regime"),
        }
    }
}

impl fmt::Display for AtomicAccessWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits8 => write!(f, "i8"),
            Self::Bits16 => write!(f, "i16"),
            Self::Bits32 => write!(f, "i32"),
            Self::Bits64 => write!(f, "i64"),
            Self::Bits128 => write!(f, "i128"),
            Self::Bits512 => write!(f, "i512"),
            Self::Pointer => write!(f, "ptr"),
        }
    }
}

/// Result encoding of a native store-conditional.
///
/// Success and failure codes belong to the instruction contract, independently
/// of memory width and reservation state. Unspecified contracts yield no status
/// value. RISC-V failure is any nonzero value, not necessarily one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AtomicStatus {
    /// No status encoding is attested.
    #[default]
    Unspecified,
    /// Zero on success and any nonzero value on failure (RISC-V).
    ZeroNonzero,
    /// Zero on success and one on failure (Arm exclusive stores).
    ZeroOne,
    /// One on success and zero on failure (MIPS conditional stores).
    OneZero,
}

/// Atomic read-modify-write operation kind.
///
/// Each variant's discriminant, its declaration order, is its stable codec
/// number, the `u16` it converts to and from: append only, never reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum AtomicRmwOp {
    /// Exchange
    Xchg,
    /// Add
    Add,
    /// Sub
    Sub,
    /// And
    And,
    /// Or
    Or,
    /// Xor
    Xor,
    /// Min (signed)
    Min,
    /// Max (signed)
    Max,
    /// Bit-clear: `*p &= !value` (AArch64 `ldclr`/`stclr`).
    AndNot,
    /// Min (unsigned) (AArch64 `ldumin`/`stumin`).
    MinU,
    /// Max (unsigned) (AArch64 `ldumax`/`stumax`).
    MaxU,
    /// Negate: `*p = -*p` (x86 `lock neg`). The operation's value is unused.
    Neg,
    /// Float add: `*p += value` in the value's float format (AArch64 `ldfadd`,
    /// `ldbfadd`).
    FAdd,
    /// Float maximum, a NaN propagating (AArch64 `ldfmax`).
    FMax,
    /// Float minimum, a NaN propagating (AArch64 `ldfmin`).
    FMin,
    /// Float maximum, a quiet NaN losing to a number (AArch64 `ldfmaxnm`).
    FMaxNum,
    /// Float minimum, a quiet NaN losing to a number (AArch64 `ldfminnm`).
    FMinNum,
}

impl AtomicRmwOp {
    /// Returns whether the operation is float arithmetic, of the format its
    /// value's type states rather than one its width implies: a 16-bit
    /// `FAdd` is a half's or a bfloat16's.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::FAdd | Self::FMin | Self::FMax | Self::FMinNum | Self::FMaxNum
        )
    }

    /// Returns the IR name of this atomic read-modify-write operation, such as `atomic.exchange`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Signedness is part of the name where the enum has a signed and an unsigned value
    /// (`atomic.min.i`, `atomic.min.u`). Whether the old value is returned is a property of the
    /// operation that holds this one, not of the name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Xchg => "atomic.exchange",
            Self::Add => "atomic.add",
            Self::Sub => "atomic.sub",
            Self::And => "atomic.and",
            Self::Or => "atomic.or",
            Self::Xor => "atomic.xor",
            Self::Min => "atomic.min.i",
            Self::Max => "atomic.max.i",
            Self::AndNot => "atomic.and.not",
            Self::MinU => "atomic.min.u",
            Self::MaxU => "atomic.max.u",
            Self::Neg => "atomic.neg",
            Self::FAdd => "atomic.add.f",
            Self::FMax => "atomic.max.f",
            Self::FMin => "atomic.min.f",
            Self::FMaxNum => "atomic.max.num",
            Self::FMinNum => "atomic.min.num",
        }
    }
}
