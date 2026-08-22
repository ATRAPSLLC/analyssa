//! One shared intra-procedural address model for the analyses that need to
//! reason about *where* a memory access lands.
//!
//! Address chains reach the SSA in many shapes — a frontend may emit the whole
//! `[base + index*scale + disp]` form as a single [`SsaOp::PtrAdd`], or shred it
//! into a `Shl`/`Mul`/`Add`/`Add` chain, and normalization passes freely rewrite
//! between the two. Analyses that key on the *address value id* are therefore
//! unstable under GVN and LICM. This module is the single home for the decoded
//! form instead:
//!
//! - [`AddressExpr`] — the normalized `base + index*stride_bytes + offset_bits`
//!   form, recovered by [`normalize_address`] with one backward-recursive,
//!   [`get_definition`](crate::ir::SsaFunction::get_definition)-based,
//!   depth-bounded walk.
//! - [`AliasKey`] — a memory-cell identity key, derived as the projection of an
//!   [`AddressExpr`] onto plain SSA value ids.
//! - [`const_i64`] / [`const_u64`] — the single constant decoder, looking a
//!   value up by definition (no whole-function scan) and adopting the complete
//!   `True`/`False` mapping.
//!
//! The walk never fabricates a relationship: anything it cannot decompose
//! normalizes to its own identity, which is the sound conservative answer.

use std::collections::BTreeMap;

use crate::{
    ir::{ConstValue, SsaFunction, SsaOp, SsaVarId},
    pointer::PointerSize,
    target::Target,
};

/// Maximum backward-walk depth for [`normalize_address`].
///
/// Bounds the recursion over definition chains so a pathological (or cyclic
/// through a phi) chain cannot loop or blow the stack; eight levels cover every
/// realistic `base + index*stride + offset` addressing chain.
const MAX_ADDRESS_DEPTH: usize = 8;

/// Reduces a folded displacement into the target's pointer width.
///
/// The model accumulates offsets as `i64`, but address arithmetic wraps at the
/// target's pointer width, and the two lowerings of one address do not agree
/// until they are reduced into it: `lea`-style arithmetic sign-extends `-8`,
/// while `add reg, 0xFFFF_FFF8` lifts to an *unsigned* 32-bit constant that
/// [`const_value_i64`] zero-extends to `+4294967288`. On a 32-bit target both
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
/// large enough to wrap it cannot be constructed ([`const_value_i64`] rejects a
/// `u64` above `i64::MAX`, and the checked arithmetic bails).
fn canonical_offset_bits(offset_bits: i64, ptr_size: PointerSize) -> i64 {
    let bytes = offset_bits.div_euclid(8);
    let sub_byte = offset_bits.rem_euclid(8);
    ptr_size
        .mask_signed(bytes)
        .saturating_mul(8)
        .saturating_add(sub_byte)
}

/// A normalized address expression: `base + index*stride_bytes + offset`, where
/// `offset` is carried in bits.
///
/// Recovered by [`normalize_address`]. The `base` is the deepest value the walk
/// could not decompose further; `index`/`stride_bytes` capture a single scaled
/// index term (`i * 4`, `i << 3`) when present; `offset_bits` accumulates the
/// constant byte displacements, stored in bits so sub-byte member offsets stay
/// representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressExpr {
    /// Root value the address is measured from.
    pub base: SsaVarId,
    /// Scaled index term, when the address adds one.
    pub index: Option<SsaVarId>,
    /// Stride applied to [`Self::index`], in bytes.
    pub stride_bytes: u64,
    /// Constant displacement from the base, in bits.
    pub offset_bits: i64,
}

impl AddressExpr {
    /// Returns the trivial address `value + 0` — `value` is its own base with
    /// no index and no offset.
    #[must_use]
    fn identity(value: SsaVarId) -> Self {
        Self {
            base: value,
            index: None,
            stride_bytes: 0,
            offset_bits: 0,
        }
    }
}

/// Normalizes `value` into an [`AddressExpr`] by walking its definition chain.
///
/// Copy / `Conv` / `Bitcast` pass through to their source; [`SsaOp::PtrAdd`] is
/// read directly onto the recursively-normalized base; `Add` / `Sub` fold a
/// constant byte displacement into [`AddressExpr::offset_bits`]; an `Add` of a
/// scaled index (`Mul`/`Shl` by a constant) records the [`AddressExpr::index`].
/// Anything else — or a depth-exhausted / overflowing chain — terminates the
/// walk with that value as the [`AddressExpr::base`]. The walk never fabricates
/// a relationship: an undecodable value normalizes to its own identity.
#[must_use]
pub fn normalize_address<T: Target>(
    ir: &SsaFunction<T>,
    value: SsaVarId,
    ptr_size: PointerSize,
) -> AddressExpr {
    normalize_inner(ir, value, MAX_ADDRESS_DEPTH, ptr_size)
}

/// Recursively normalizes `value` into an [`AddressExpr`] of
/// `base + offset + scaled-index` form, descending at most `depth` levels
/// through the defining instructions.
fn normalize_inner<T: Target>(
    ir: &SsaFunction<T>,
    value: SsaVarId,
    depth: usize,
    ptr_size: PointerSize,
) -> AddressExpr {
    if depth == 0 {
        return AddressExpr::identity(value);
    }
    let Some(op) = ir.get_definition(value) else {
        return AddressExpr::identity(value);
    };
    match op {
        SsaOp::Copy { src, .. }
        | SsaOp::IntConv { operand: src, .. }
        | SsaOp::IntToPtr { operand: src, .. }
        | SsaOp::PtrToInt { operand: src, .. }
        | SsaOp::Bitcast { operand: src, .. } => {
            normalize_inner(ir, *src, depth.saturating_sub(1), ptr_size)
        }
        // `PtrAdd` *is* the normalized address, so read its fields directly onto
        // the recursively-normalized base. This yields exactly the `AddressExpr`
        // a shredded add-chain produces, so both lowerings decode identically.
        SsaOp::PtrAdd {
            base,
            index,
            stride,
            offset,
            ..
        } => {
            let base_expr = normalize_inner(ir, *base, depth.saturating_sub(1), ptr_size);
            let Some(off_bits) = offset.checked_mul(8) else {
                return AddressExpr::identity(value);
            };
            let mut expr = fold_offset(base_expr, off_bits, value, ptr_size);
            if expr.base == value {
                // `fold_offset` bailed to identity: the displacement is not one
                // this model will trust. Don't then graft an index onto it.
                return expr;
            }
            if let Some(index) = index {
                // The walk models a single scaled index term; if the base
                // already carried one, don't fabricate a second.
                if expr.index.is_some() {
                    return AddressExpr::identity(value);
                }
                expr.index = Some(*index);
                expr.stride_bytes = *stride;
            }
            expr
        }
        SsaOp::Add { left, right, .. } => normalize_add(ir, value, *left, *right, depth, ptr_size),
        SsaOp::Sub { left, right, .. } => {
            let Some(offset) = const_i64(ir, *right).and_then(|v| v.checked_mul(8)) else {
                return AddressExpr::identity(value);
            };
            let base = normalize_inner(ir, *left, depth.saturating_sub(1), ptr_size);
            let Some(negated) = offset.checked_neg() else {
                return AddressExpr::identity(value);
            };
            fold_offset(base, negated, value, ptr_size)
        }
        _ => AddressExpr::identity(value),
    }
}

/// Normalizes an `add` of `left + right` into an [`AddressExpr`], folding a
/// constant operand into the byte offset and recognising scaled-index forms.
fn normalize_add<T: Target>(
    ir: &SsaFunction<T>,
    value: SsaVarId,
    left: SsaVarId,
    right: SsaVarId,
    depth: usize,
    ptr_size: PointerSize,
) -> AddressExpr {
    // `base + const` / `const + base`: fold the constant byte offset.
    if let Some(offset) = const_i64(ir, right).and_then(|v| v.checked_mul(8)) {
        return fold_offset(
            normalize_inner(ir, left, depth.saturating_sub(1), ptr_size),
            offset,
            value,
            ptr_size,
        );
    }
    if let Some(offset) = const_i64(ir, left).and_then(|v| v.checked_mul(8)) {
        return fold_offset(
            normalize_inner(ir, right, depth.saturating_sub(1), ptr_size),
            offset,
            value,
            ptr_size,
        );
    }
    // `base + index*stride`: record the scaled index onto the other side's
    // base, but only when that side carries no index yet (a single index term).
    if let Some((index, stride)) = scaled_index(ir, right) {
        let mut base = normalize_inner(ir, left, depth.saturating_sub(1), ptr_size);
        if base.index.is_none() {
            base.index = Some(index);
            base.stride_bytes = stride;
            return base;
        }
    }
    if let Some((index, stride)) = scaled_index(ir, left) {
        let mut base = normalize_inner(ir, right, depth.saturating_sub(1), ptr_size);
        if base.index.is_none() {
            base.index = Some(index);
            base.stride_bytes = stride;
            return base;
        }
    }
    AddressExpr::identity(value)
}

/// Adds a constant byte `offset` to an address expression's accumulated
/// offset, tagging the result with the originating `value`.
fn fold_offset(
    mut base: AddressExpr,
    offset: i64,
    value: SsaVarId,
    ptr_size: PointerSize,
) -> AddressExpr {
    match base.offset_bits.checked_add(offset) {
        Some(folded) => {
            // Canonicalise into the target pointer width so the sign-extended
            // and zero-extended lowerings of one displacement agree.
            base.offset_bits = canonical_offset_bits(folded, ptr_size);
            base
        }
        None => AddressExpr::identity(value),
    }
}

/// Returns `Some((index, stride_bytes))` when `value` is a constant-scaled
/// index — a multiply or a left-shift by a constant — and `None` otherwise.
#[must_use]
pub fn scaled_index<T: Target>(ir: &SsaFunction<T>, value: SsaVarId) -> Option<(SsaVarId, u64)> {
    match ir.get_definition(value)? {
        SsaOp::Mul { left, right, .. } => {
            if let Some(stride) = const_u64(ir, *right) {
                return Some((*left, stride));
            }
            if let Some(stride) = const_u64(ir, *left) {
                return Some((*right, stride));
            }
            None
        }
        SsaOp::Shl { value, amount, .. } => {
            let amount = u32::try_from(const_u64(ir, *amount)?).ok()?;
            1_u64.checked_shl(amount).map(|stride| (*value, stride))
        }
        _ => None,
    }
}

/// Resolves `value` to a signed 64-bit constant by looking up its definition.
///
/// Returns `Some` only when the value is defined by a [`SsaOp::Const`] whose
/// payload converts to `i64`. The lookup is by definition (no whole-function
/// scan), so it is O(1) per call rather than O(n).
#[must_use]
pub fn const_i64<T: Target>(ir: &SsaFunction<T>, value: SsaVarId) -> Option<i64> {
    match ir.get_definition(value)? {
        SsaOp::Const { value, .. } => const_value_i64(value),
        _ => None,
    }
}

/// Resolves `value` to an unsigned 64-bit constant, returning `None` for a
/// negative or non-constant value.
#[must_use]
pub fn const_u64<T: Target>(ir: &SsaFunction<T>, value: SsaVarId) -> Option<u64> {
    u64::try_from(const_i64(ir, value)?).ok()
}

/// Converts a [`ConstValue`] to an `i64`, widening smaller integers and
/// fallibly narrowing larger ones.
///
/// Covers every integer and native-int width, the boolean `True` (`1`) /
/// `False` (`0`) constants, and a [`ConstValue::Symbol`] that denotes an
/// address; everything else returns `None`.
///
/// The symbol arm is what keeps this — and therefore
/// [`normalize_address`], [`AliasKey`], and every range/interval analysis built
/// on them — working once a host stops emitting materialised addresses as
/// integer constants. Without it a symbolic address would simply vanish from
/// the numeric analyses, which reads as "points-to got worse" rather than as
/// the representation change it actually is.
#[must_use]
pub fn const_value_i64<T: Target>(value: &ConstValue<T>) -> Option<i64> {
    match value {
        ConstValue::Symbol(symbol) => T::symbol_address(symbol),
        ConstValue::I8(value) => Some(i64::from(*value)),
        ConstValue::I16(value) => Some(i64::from(*value)),
        ConstValue::I32(value) => Some(i64::from(*value)),
        ConstValue::I64(value) => Some(*value),
        ConstValue::U8(value) => Some(i64::from(*value)),
        ConstValue::U16(value) => Some(i64::from(*value)),
        ConstValue::U32(value) => Some(i64::from(*value)),
        ConstValue::U64(value) => i64::try_from(*value).ok(),
        ConstValue::NativeInt(value) => Some(*value),
        ConstValue::NativeUInt(value) => i64::try_from(*value).ok(),
        ConstValue::True => Some(1),
        ConstValue::False => Some(0),
        _ => None,
    }
}

/// Memory-cell identity key for a `base + index*stride + offset` address.
///
/// The cell-identity projection of an [`AddressExpr`]: two addresses share a
/// key when they reduce to the same base, the same scaled index (by SSA value
/// identity and stride), and the same constant offset. Keeping the scaled-index
/// term lets array-element accesses participate: `arr[i]` loaded then stored, or
/// `ptr[i].field` chains, denote the same cell. It is sound as a cell identity
/// because in SSA the same index value-id denotes one value — distinct indices
/// (`arr[i]` vs `arr[j]`) keep distinct keys.
///
/// Note this is an *equality* key, not a may-alias relation: it carries no
/// access width, so two overlapping accesses of different widths at nearby
/// offsets get distinct keys. Alias reasoning that needs overlap belongs in
/// [`MemoryLocation`](crate::analysis::memory::MemoryLocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AliasKey {
    /// Root SSA value id.
    pub base_value: u32,
    /// Scaled-index SSA value id, when the address adds one.
    pub index: Option<u32>,
    /// Stride applied to [`Self::index`], in bytes (`0` when no index).
    pub stride_bytes: u64,
    /// Constant offset in bits from the root.
    pub offset_bits: i64,
}

/// Computes alias keys for the address chains in `ir`.
///
/// Each address-producing value (`Copy`/`Conv`/`Bitcast`/`Add`/`Sub`) is
/// normalized via [`normalize_address`]; a value that reduces past its own
/// trivial identity gets a key. Values without a simplified address fall back
/// to their own identity at lookup time (see [`alias_key_for_value`]).
#[must_use]
pub fn alias_keys_for_function<T: Target>(
    ir: &SsaFunction<T>,
    ptr_size: PointerSize,
) -> BTreeMap<u32, AliasKey> {
    let mut keys = BTreeMap::new();
    for block in ir.blocks() {
        for instruction in block.instructions() {
            let dest = match instruction.op() {
                SsaOp::Copy { dest, .. }
                | SsaOp::IntConv { dest, .. }
                | SsaOp::IntToPtr { dest, .. }
                | SsaOp::PtrToInt { dest, .. }
                | SsaOp::Bitcast { dest, .. }
                | SsaOp::Add { dest, .. }
                | SsaOp::Sub { dest, .. } => *dest,
                _ => continue,
            };
            let address = normalize_address(ir, dest, ptr_size);
            // A value that did not simplify past its own identity (no index, its
            // own base, zero offset) keeps the lookup fallback — no recorded key.
            // A `base + index*stride (+ offset)` chain DOES get a key, so
            // array-element cells share an identity.
            if address.index.is_none() && address.base == dest && address.offset_bits == 0 {
                continue;
            }
            keys.insert(
                dest.as_u32(),
                AliasKey {
                    base_value: address.base.as_u32(),
                    index: address.index.map(SsaVarId::as_u32),
                    stride_bytes: address.stride_bytes,
                    offset_bits: address.offset_bits,
                },
            );
        }
    }
    keys
}

/// Returns the alias key for `value`, falling back to the value's own identity
/// (`base = value`, `offset = 0`) when no simplified key was recorded.
#[must_use]
pub fn alias_key_for_value(value: SsaVarId, keys: &BTreeMap<u32, AliasKey>) -> Option<AliasKey> {
    let id = value.as_u32();
    keys.get(&id).copied().or(Some(AliasKey {
        base_value: id,
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
            SsaBlock, SsaInstruction,
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

    #[test]
    fn const_value_i64_covers_booleans_and_integers() {
        assert_eq!(const_value_i64(&ConstValue::<MockTarget>::True), Some(1));
        assert_eq!(const_value_i64(&ConstValue::<MockTarget>::False), Some(0));
        assert_eq!(
            const_value_i64(&ConstValue::<MockTarget>::I32(-7)),
            Some(-7)
        );
        assert_eq!(
            const_value_i64(&ConstValue::<MockTarget>::U64(u64::MAX)),
            None,
            "an out-of-range unsigned constant does not convert"
        );
    }

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

        let address = normalize_address(&ir, addr, PointerSize::Bit64);
        assert_eq!(address.base, base);
        assert_eq!(address.index, None);
        // 4 bytes folds to 32 bits.
        assert_eq!(address.offset_bits, 32);
    }

    /// The two lowerings of one displacement must decode to one address.
    ///
    /// A 32-bit target reaches `[base - 8]` either as a sign-extended `-8` or as
    /// `add reg, 0xFFFF_FFF8`, whose unsigned constant `const_value_i64`
    /// zero-extends to `+4294967288`. Both name the same cell. Left as `i64`
    /// they sit 4 GiB apart, and `IndirectLocation::may_alias` would then prove
    /// them *disjoint* — a false NoAlias, which lets a stale value survive a
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
            normalize_address(&ir, addr, ptr_size)
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

    /// Ordinary displacements — including genuinely negative ones — must keep
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

            let address = normalize_address(&ir, addr, PointerSize::Bit64);
            assert_eq!(
                address.base, base,
                "{value:?} should decompose onto the base"
            );
            assert_eq!(address.offset_bits, expected_bits, "for {value:?}");
        }
    }

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

        let address = normalize_address(&ir, addr, PointerSize::Bit64);
        assert_eq!(address.base, base);
        assert_eq!(address.index, Some(index));
        assert_eq!(address.stride_bytes, 8);
        assert_eq!(address.offset_bits, 0);
    }

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

        let address = normalize_address(&ir, addr, PointerSize::Bit64);
        assert_eq!(address.base, base);
        assert_eq!(address.index, Some(index));
        assert_eq!(address.stride_bytes, 4);
        // 8 bytes → 64 bits, matching the chain-based normalization.
        assert_eq!(address.offset_bits, 64);
    }

    #[test]
    fn normalize_unknown_value_is_identity() {
        let ir = function();
        let value = SsaVarId::from_index(7);
        let address = normalize_address(&ir, value, PointerSize::Bit64);
        assert_eq!(address.base, value);
        assert_eq!(address.index, None);
        assert_eq!(address.offset_bits, 0);
    }

    #[test]
    fn alias_key_for_value_falls_back_to_identity() {
        let keys = BTreeMap::new();
        let value = SsaVarId::from_index(3);
        assert_eq!(
            alias_key_for_value(value, &keys),
            Some(AliasKey {
                base_value: 3,
                index: None,
                stride_bytes: 0,
                offset_bits: 0,
            })
        );
    }

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

        let keys = alias_keys_for_function(&ir, PointerSize::Bit64);
        assert_eq!(
            keys.get(&addr.as_u32()),
            Some(&AliasKey {
                base_value: base.as_u32(),
                index: Some(index.as_u32()),
                stride_bytes: 8,
                offset_bits: 0,
            })
        );
    }
}
