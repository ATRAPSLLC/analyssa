//! Integer conversion-chain collapse.
//!
//! A lift emits one conversion per width change, so a value that is narrowed,
//! widened and narrowed again arrives as a chain of `IntConv` operations. This
//! module holds the single rule that decides when the outer conversion may read
//! straight past the ones inside it, and how far past.
//!
//! The rule lives here rather than in [`crate::analysis::algebraic`] because it
//! walks the function: resolving a link needs the definition of the operand,
//! and the walk repeats while links keep falling away. `analysis::algebraic`
//! promises O(1) per operation with no recursion or iteration, which a
//! function-walking rule cannot honour.
//!
//! # Complexity
//!
//! O(d) per candidate conversion, where `d` is the depth reached. Each step is
//! one guarded, scan-free def-site dereference
//! ([`SsaFunction::recorded_definition`]), so a pass that applies this to every
//! conversion in a function costs O(n + C·d) overall, for `C` conversions.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    PointerSize,
    ir::{
        function::SsaFunction,
        integer::BitWidth,
        ops::{CmpKind, SsaOp},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// Collapses an integer conversion whose operand is another integer
/// conversion, where the intermediate one cannot be observed.
///
/// A lift emits one conversion per width change, so a value that is narrowed,
/// widened and narrowed again arrives as a chain - and the chain is what a
/// reader sees as `(uint16_t)(uint64_t)(uint32_t)x`. Measured over the
/// committed fixtures, **40.8%** of `x86_64` integer conversions take another
/// conversion as their operand, and 28.1% on `arm_64`.
///
/// # When the intermediate is unobservable
///
/// A conversion to width `w` yields the low `w` bits of its operand, extended
/// above `w` when the operand is narrower - by sign or by zero, according to
/// its own `unsigned` flag. So the outer conversion reads only bits the source
/// supplied directly, and the intermediate changes nothing, exactly when the
/// outer width is no wider than *both* the intermediate width and the source
/// width.
///
/// Both halves of that condition are needed, and the second is the one that is
/// easy to lose. `(uint32_t)(int64_t)x` on an 8-bit `x` is **not**
/// `(uint32_t)x`: the intermediate sign-extends, so bits 8..32 of the result
/// are copies of the sign bit, while the collapsed form would zero them. That
/// is a different number, not a different spelling of one.
///
/// Where the result *is* wider than the source, the bits between them are an
/// extension either way - the intermediate's before collapsing, the surviving
/// conversion's after - so the two are interchangeable exactly when they agree
/// on sign or zero. That is the second arm, and it is what lets
/// `(uint32_t)(uint64_t)x` collapse where `(uint32_t)(int64_t)x` may not.
///
/// An intermediate at the source's own width - a restatement of the same bits
/// under another sign, as a lift writes when a signed result lands in a
/// register it types unsigned - is unobservable whatever the result's width:
/// it keeps every bit the source has and extends nothing, so the result reads
/// the source's bits and extends them by its own flag either way. It is read
/// through only when that makes the extension agree with the type it extends:
/// a sign-extension of a 32-bit signed `x` restated unsigned is the
/// sign-extension of `x`, and the restatement said only the opposite of what
/// the extension does. A restatement that *agrees* with the extension - RV64's
/// `sext.w`, an unsigned value restated signed and then sign-extended - is
/// where the value's sign is stated at all, and stays.
///
/// A conversion that checks overflow is never removed: the check is the point
/// of it.
///
/// A float conversion ([`SsaOp::FloatConv`]) is removed only where it converts
/// to the format its operand already has, which changes no value: a lift that
/// widens what it reads before it computes cannot always know the width of a
/// value another block supplies, and widens it anyway. Its chains are left
/// alone, since a narrowing rounds and a widening of a signalling NaN quiets
/// it.
///
/// # Widths
///
/// A link's width is read from the conversion's own `target`, and the source's
/// from the source variable's declared type: a variable's declared type is the
/// truth about its width. IR whose typing contradicts its conversions is out of
/// scope here, as it is for every other consumer.
///
/// # Returns
///
/// The operation that should replace `op`, or `None` when nothing can be
/// removed. Two shapes come back:
///
/// - [`SsaOp::Copy`] when `op`, an integer or a float conversion, converts to
///   the type its operand already has, so the conversion converts nothing;
/// - [`SsaOp::IntConv`] reading whichever variable the walk reached, when one
///   or more links fell away.
///
/// `None` also covers every refusal - an overflow check, a width the rule
/// cannot clear, a link whose definition is not recoverable from its def site,
/// and a def-use chain long enough to be cyclic.
///
/// # Correctness
///
/// Each link is resolved with
/// [`SsaFunction::recorded_definition`](SsaFunction::recorded_definition), the
/// guarded lookup, so a def site an edit left stale stops the walk instead of
/// importing an unrelated conversion's widths - a missed optimization rather
/// than a rewrite to an operand that never fed this conversion.
///
/// The walk refuses outright once it has taken more steps than the function has
/// variable ids ([`SsaFunction::var_id_bound`]). A chain that long has to
/// revisit a variable, which is not SSA, so the bound is exact by pigeonhole
/// and costs no allocation and no visited set to enforce.
///
/// # Examples
///
/// ```
/// use analyssa::{
///     analysis::convert::collapse_conversion_chain,
///     ir::ops::SsaOp,
///     testing::{MockConvLink, MockConversionChain, MockType},
/// };
///
/// // x: 64 bits, widened to 64, narrowed to 32. The intermediate touches
/// // nothing the outer conversion reads.
/// let chain = MockConversionChain::build(
///     MockType::I64,
///     &[
///         MockConvLink::new(MockType::I64, false),
///         MockConvLink::new(MockType::I32, false),
///     ],
/// );
///
/// let collapsed = collapse_conversion_chain(chain.function(), &chain.outer_op())
///     .expect("the intermediate is unobservable");
/// assert!(matches!(
///     collapsed,
///     SsaOp::IntConv { operand, .. } if operand == chain.source()
/// ));
/// ```
#[must_use]
pub fn collapse_conversion_chain<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
) -> Option<SsaOp<T>> {
    if let SsaOp::FloatConv {
        dest,
        operand,
        target,
        ..
    } = op
    {
        return ssa
            .variable(*operand)
            .is_some_and(|variable| variable.var_type() == target)
            .then_some(SsaOp::Copy {
                dest: *dest,
                src: *operand,
            });
    }
    let SsaOp::IntConv {
        dest,
        operand,
        target,
        overflow_check,
        unsigned,
    } = op
    else {
        return None;
    };
    if *overflow_check {
        return None;
    }

    // A conversion to the type the operand already has converts nothing. The
    // lift cannot always see this: it decides from the *register view's* widths,
    // and whether the value in that register carries the narrower type is
    // settled later, by the copy propagation that replaces a wide operand with
    // an already-narrow one. So the identity appears here and nowhere earlier -
    // and every one of them is a cast in front of a reader.
    if ssa
        .variable(*operand)
        .is_some_and(|variable| variable.var_type() == target)
    {
        return Some(SsaOp::Copy {
            dest: *dest,
            src: *operand,
        });
    }

    let outer_width = T::bit_width(target)?;

    // Walk inward as far as the chain allows, rather than one link per pass. A
    // value narrowed, widened and narrowed again is three conversions deep, and
    // peeling one at a time would need the whole pass to run three times to say
    // what one walk says once.
    let mut reached = *operand;
    let mut steps: usize = 0;
    while let Some(definition) = ssa.recorded_definition(reached) {
        let SsaOp::IntConv {
            operand: source,
            target: link_target,
            overflow_check: link_overflow,
            unsigned: link_unsigned,
            ..
        } = definition.op()
        else {
            break;
        };

        if *link_overflow {
            break;
        }
        let Some(link_width) = T::bit_width(link_target) else {
            break;
        };
        let Some(source_width) = ssa.bit_width(*source) else {
            break;
        };

        // A restatement keeps every bit of its source and extends none, so the
        // result reads through it at any width - when the source then states
        // the sign the result extends by, which the restatement contradicted.
        let restates = link_width == source_width
            && outer_width > source_width
            && ssa
                .variable(*source)
                .is_some_and(|variable| T::is_signed(variable.var_type()) != *unsigned)
            && T::is_signed(link_target) == *unsigned;
        // The result must not read past what this link kept.
        if !restates && outer_width > link_width {
            break;
        }
        // Below the source width, every bit the result keeps came from the
        // source directly. At or above it, the bits between the source and the
        // result are this link's extension, and the collapsed form supplies its
        // own - so the two must agree on which.
        if !restates && outer_width > source_width && *link_unsigned != *unsigned {
            break;
        }

        // One step per link consumed. A well-formed chain visits each variable
        // at most once, so exceeding the id bound means the def-use chain is
        // cyclic and the IR is malformed; refuse rather than spin.
        steps = steps.saturating_add(1);
        if steps > ssa.var_id_bound() {
            return None;
        }
        reached = *source;
    }

    if reached == *operand {
        return None;
    }

    Some(SsaOp::IntConv {
        dest: *dest,
        operand: reached,
        target: target.clone(),
        overflow_check: *overflow_check,
        unsigned: *unsigned,
    })
}

/// Drops a disjunct a narrowing conversion throws away.
///
/// A **byte insert** is how a target with no sub-register write spells
/// "replace the low byte": keep the rest of the register, mask the new byte in,
/// then read the register back at byte width. x86's `setcc %al` inside a 64-bit
/// register is exactly that, and so is a `bfi` fed to a narrow read. What
/// arrives is
///
/// ```text
/// t = (wide & ~0xff) | (narrow & 0xff)
/// b = (u8) t
/// ```
///
/// and the first disjunct contributes nothing to `b`: every bit it can set lies
/// above the conversion's width. Reading it anyway makes the result look like
/// it could be anything the wide value holds, which is what leaves a
/// `setcc`-derived value with a 256-wide range instead of the two values it can
/// take - and a dispatcher whose state is written that way unanalysable.
///
/// So: **a narrowing conversion of an `Or` reads only the disjuncts whose bits
/// it keeps.** A disjunct is dropped when it is masked by a constant with no
/// bit inside the conversion's width, or shifted left by at least that width,
/// the two forms that prove it without a bit-level domain. The second is the
/// high half of a pair composed `(hi << n) | lo` and cut back to its low half:
/// a result beside a value the body left in the register after it.
///
/// # Arguments
///
/// * `ssa` - The function, for the disjuncts' defining operations.
/// * `op` - The conversion to simplify.
///
/// # Returns
///
/// The same conversion reading the surviving disjunct, or `None` when the
/// operand is not an `Or`, when neither side is provably outside the width, or
/// when both are (which is a zero this rule declines to assert).
#[must_use]
pub fn drop_discarded_disjunct<T: Target>(ssa: &SsaFunction<T>, op: &SsaOp<T>) -> Option<SsaOp<T>> {
    let SsaOp::IntConv {
        dest,
        operand,
        target,
        overflow_check,
        unsigned,
    } = op
    else {
        return None;
    };
    if *overflow_check {
        return None;
    }
    let width = T::bit_width(target).filter(|width| *width >= 1)?;
    // The bits a 64-bit mask keeps inside the width; a mask holds no bit
    // above a width of 64 or more.
    let kept = (width < 64).then(|| BitWidth::new(width).mask());

    let SsaOp::Or { left, right, .. } = ssa.recorded_definition(*operand)?.instruction().op()
    else {
        return None;
    };
    let (left, right) = (*left, *right);
    let constant = |value: SsaVarId| match ssa
        .recorded_definition(value)
        .map(|definition| definition.instruction().op())
    {
        // The constant's *bit pattern*, which is what "outside the width"
        // is a question about: `-256` and `0xffff_ff00` name the same set of
        // bits and the lift may carry either.
        Some(SsaOp::Const { value, .. }) => value
            .as_u64()
            .or_else(|| value.as_i64().map(|signed| signed as u64)),
        _ => None,
    };
    let outside = |value: SsaVarId| -> bool {
        let definition = ssa
            .recorded_definition(value)
            .map(|definition| definition.instruction().op());
        // Shifted left past the width: no bit of it is left inside.
        if let Some(SsaOp::Shl { amount, .. }) = definition {
            return constant(*amount).is_some_and(|amount| amount >= u64::from(width));
        }
        let Some(SsaOp::And {
            left: masked,
            right: mask,
            ..
        }) = definition
        else {
            return false;
        };
        // Either operand may carry the literal; `x & m` and `m & x` are the
        // same mask.
        [*masked, *mask]
            .into_iter()
            .filter_map(constant)
            .any(|literal| kept.is_some_and(|kept| literal & kept == 0))
    };

    let survivor = match (outside(left), outside(right)) {
        (true, false) => right,
        (false, true) => left,
        // Both outside would make the conversion a zero, which is a claim this
        // rule does not need to make and the constant folder can reach on its
        // own; neither outside means the `Or` is doing real work.
        _ => return None,
    };
    Some(SsaOp::IntConv {
        dest: *dest,
        operand: survivor,
        target: target.clone(),
        overflow_check: false,
        unsigned: *unsigned,
    })
}

/// A value a narrowed operation reads that the function does not yet hold at
/// the narrow width, and how the caller defines it there.
#[derive(Debug, Clone)]
pub enum NarrowValue<T: Target> {
    /// A constant, at the narrow width.
    Constant(ConstValue<T>),
    /// A value narrower still, extended to the narrow width as the wide
    /// operand extended it (zero-extended when `unsigned`).
    Extension {
        /// The narrower value.
        source: SsaVarId,
        /// Whether the extension reads it unsigned.
        unsigned: bool,
    },
    /// A value the function is entered with, read at the narrow width: its
    /// low bits.
    Truncation {
        /// The value entered with.
        source: SsaVarId,
    },
    /// An index scaled by an address's stride, at the narrow width: the
    /// index's narrow value (an earlier entry's, or a value already that
    /// wide) times the stride's low bits.
    Scaled {
        /// The narrow index, or the wide value an earlier entry narrows.
        index: SsaVarId,
        /// The stride, at the narrow width.
        stride: ConstValue<T>,
    },
}

/// An operation [`narrow_truncated_operation`] or
/// [`narrow_extended_comparison`] narrowed, with the values it reads that need
/// defining at the narrow width.
#[derive(Debug, Clone)]
pub struct NarrowedOperation<T: Target> {
    /// The operation at the narrow width: defining the conversion's
    /// destination, or the comparison's.
    pub op: SsaOp<T>,
    /// The wide operands [`Self::op`] reads that have no value at the narrow
    /// width yet, each with how to define one: the caller defines one value
    /// per entry, of type [`Self::ty`], and has the operation read it
    /// instead.
    pub inserted: Vec<(SsaVarId, NarrowValue<T>)>,
    /// The type of the narrow width: the conversion's target, or the type of
    /// the values a comparison's operands extend.
    pub ty: T::Type,
}

/// Narrows a truncated operation on extended values to the operation at the
/// narrow width.
///
/// A 64-bit machine computing on 32-bit values often runs part of a
/// computation at the register's width: RISC-V has no 32-bit `xor` or `and`,
/// so `(x ^ y) + 2 * (x & y)` on `int` arrives as
///
/// ```text
/// t = (int32) ((int64) x ^ (int64) y)
/// ```
///
/// beside the 32-bit `addw` and `slliw` - one expression at two widths, which
/// every rule that matches it as a whole (the MBA identities, the byte-swap
/// and rotate idioms) cannot see as one. The low `n` bits of a sum,
/// difference, product, conjunction, disjunction, exclusive or or left shift
/// depend only on the low `n` bits of its operands, and the low `n` bits of a
/// value extended from `n` bits are that value. So: **a conversion to `n` bits
/// of such an operation on two values extended from `n` bits is the operation
/// on those values**, at `n` bits, defining the conversion's result. An operand
/// may also be a constant (the `-1` of RISC-V's `not`, `xori rd, rs, -1`), or
/// an extension from fewer than `n` bits (a loaded byte): the narrow operation
/// reads its low `n` bits, as a constant or extension at the narrow width the
/// caller defines ([`NarrowedOperation::inserted`]). So may a value the
/// function is entered with, read at the narrow width: `t_i32(int a, int b)`
/// on x86-64 is `lea eax, [rdi + rsi]`, a 64-bit sum of the whole argument
/// registers cut to 32 bits, and the sum of their low halves is the same
/// value, read where a parameter is the part of its register its reads take
/// (an entry value every read of which is its low bits is those bits). At
/// least one operand must be an extension from exactly `n` bits or a value
/// entered with, so an operation on constants alone is left to constant
/// folding and one that would only add conversions of computed values is
/// left as it is. A left shift qualifies only by a constant below `n`,
/// where the narrow shift agrees with the wide one.
///
/// # Arguments
///
/// * `ssa` - The function, for the operands' defining operations.
/// * `op` - The conversion to narrow.
/// * `ptr_size` - Target pointer size, the width of the native integers.
///
/// # Returns
///
/// The operation at the narrow width, defining the conversion's destination,
/// with the constants it reads narrowed, or `None` when `op` is not a plain
/// narrowing conversion of such an operation on extended values.
#[must_use]
pub fn narrow_truncated_operation<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    ptr_size: PointerSize,
) -> Option<NarrowedOperation<T>> {
    // A narrowing conversion: of an integer, or of an address back to an
    // integer, which is how a lift spells `lea` computing a sum.
    let (dest, operand, target) = match op {
        SsaOp::IntConv {
            dest,
            operand,
            target,
            overflow_check: false,
            ..
        }
        | SsaOp::PtrToInt {
            dest,
            operand,
            target,
        } => (dest, operand, target),
        _ => return None,
    };
    let width = T::bit_width(target)?;
    // A pointer is as wide as the target's pointers, where its type states no
    // width of its own.
    let width_of = |value: SsaVarId| {
        ssa.bit_width(value)
            .or_else(|| T::is_pointer(ssa.variable(value)?.var_type()).then(|| ptr_size.bits()))
    };
    if width_of(*operand)? <= width {
        return None;
    }
    // The value a wide operand extends from `width` bits, or a constant, whose
    // low `width` bits are all the narrow operation reads.
    let inserted = std::cell::RefCell::new(Vec::new());
    let extended = std::cell::Cell::new(false);
    let narrow = |value: SsaVarId| -> Option<SsaVarId> {
        // A value the function is entered with - one no instruction and no
        // phi defines, a native entry register's, or an argument a bytecode
        // loads - has low bits as much a value as an extension's.
        let entered = |value: SsaVarId| -> Option<SsaVarId> {
            if width_of(value)? <= width || !T::is_integer(ssa.variable(value)?.var_type()) {
                return None;
            }
            extended.set(true);
            inserted
                .borrow_mut()
                .push((value, NarrowValue::Truncation { source: value }));
            Some(value)
        };
        let Some(definition) = ssa.recorded_definition(value) else {
            if ssa.find_phi_defining(value).is_some() {
                return None;
            }
            return entered(value);
        };
        match definition.instruction().op() {
            SsaOp::LoadArg { .. } => entered(value),
            SsaOp::IntConv {
                operand: source,
                overflow_check: false,
                ..
            } if width_of(*source) == Some(width) => {
                extended.set(true);
                Some(*source)
            }
            SsaOp::IntConv {
                operand: source,
                overflow_check: false,
                unsigned,
                ..
            } if width_of(*source)? < width => {
                inserted.borrow_mut().push((
                    value,
                    NarrowValue::Extension {
                        source: *source,
                        unsigned: *unsigned,
                    },
                ));
                Some(value)
            }
            SsaOp::Const {
                value: constant, ..
            } => {
                let (bits, _) = constant.integer_pattern(ptr_size)?;
                let narrowed = ConstValue::from_bits_of_type(bits, target)?;
                inserted
                    .borrow_mut()
                    .push((value, NarrowValue::Constant(narrowed)));
                Some(value)
            }
            _ => None,
        }
    };
    let dest = *dest;
    let narrowed = match ssa.recorded_definition(*operand)?.instruction().op() {
        // An address that only sums integers, read back as an integer: `lea
        // eax, [rdi + rsi]`, `lea eax, [rdi + 5]`, and `lea eax, [rdi +
        // rsi*4]`, whose scaled index the caller defines at the narrow width
        // ahead of the sum. A sum of three terms is more than that.
        SsaOp::PtrAdd {
            base,
            index,
            stride,
            offset,
            ..
        } if matches!(op, SsaOp::PtrToInt { .. }) => match (index, stride, offset) {
            (Some(index), 1, 0) => SsaOp::Add {
                dest,
                left: narrow(*base)?,
                right: narrow(*index)?,
                flags: None,
            },
            (Some(index), stride, 0) => {
                let left = narrow(*base)?;
                let index = narrow(*index)?;
                // The address itself stands for the scaled index until the
                // caller defines it: the sum reads nothing else of it.
                let stride = ConstValue::from_bits_of_type(*stride, target)?;
                inserted
                    .borrow_mut()
                    .push((*operand, NarrowValue::Scaled { index, stride }));
                SsaOp::Add {
                    dest,
                    left,
                    right: *operand,
                    flags: None,
                }
            }
            (None, _, displacement) if *displacement != 0 => {
                let left = narrow(*base)?;
                let (bits, _) =
                    ConstValue::<T>::integer(64, true, *displacement)?.integer_pattern(ptr_size)?;
                // The address itself stands for the narrow constant until the
                // caller defines it: the sum reads nothing else of it.
                inserted.borrow_mut().push((
                    *operand,
                    NarrowValue::Constant(ConstValue::from_bits_of_type(bits, target)?),
                ));
                SsaOp::Add {
                    dest,
                    left,
                    right: *operand,
                    flags: None,
                }
            }
            _ => return None,
        },
        SsaOp::Add {
            left,
            right,
            flags: None,
            ..
        } => SsaOp::Add {
            dest,
            left: narrow(*left)?,
            right: narrow(*right)?,
            flags: None,
        },
        SsaOp::Sub {
            left,
            right,
            flags: None,
            ..
        } => SsaOp::Sub {
            dest,
            left: narrow(*left)?,
            right: narrow(*right)?,
            flags: None,
        },
        SsaOp::Mul {
            left,
            right,
            flags: None,
            ..
        } => SsaOp::Mul {
            dest,
            left: narrow(*left)?,
            right: narrow(*right)?,
            flags: None,
        },
        SsaOp::And {
            left,
            right,
            flags: None,
            ..
        } => SsaOp::And {
            dest,
            left: narrow(*left)?,
            right: narrow(*right)?,
            flags: None,
        },
        SsaOp::Or {
            left,
            right,
            flags: None,
            ..
        } => SsaOp::Or {
            dest,
            left: narrow(*left)?,
            right: narrow(*right)?,
            flags: None,
        },
        SsaOp::Xor {
            left,
            right,
            flags: None,
            ..
        } => SsaOp::Xor {
            dest,
            left: narrow(*left)?,
            right: narrow(*right)?,
            flags: None,
        },
        SsaOp::Shl {
            value,
            amount,
            flags: None,
            ..
        } => {
            let SsaOp::Const { value: count, .. } =
                ssa.recorded_definition(*amount)?.instruction().op()
            else {
                return None;
            };
            if count.as_u64()? >= u64::from(width) {
                return None;
            }
            SsaOp::Shl {
                dest,
                value: narrow(*value)?,
                amount: *amount,
                flags: None,
            }
        }
        _ => return None,
    };
    if !extended.get() {
        return None;
    }
    Some(NarrowedOperation {
        op: narrowed,
        inserted: inserted.into_inner(),
        ty: target.clone(),
    })
}

/// How a value is extended from a narrower width: the width, and whether by
/// zeros rather than by its sign.
///
/// What [`narrow_extended_comparison`] and [`narrowable_phis`] read an
/// extension as, and what a host reads a convention's extension of a narrow
/// value as: RISC-V and MIPS return a narrow result extended to its register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extension {
    /// The width extended from.
    pub from: u32,
    /// Whether the extension fills with zeros rather than the sign.
    pub unsigned: bool,
}

impl Extension {
    /// The extension a plain integer conversion of `value` to a wider width
    /// is, with the value it extends; `None` where `value` is defined by no
    /// such conversion of an integer.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> Option<(SsaVarId, Self)> {
        let SsaOp::IntConv {
            operand: source,
            overflow_check: false,
            unsigned,
            ..
        } = ssa.recorded_definition(value)?.instruction().op()
        else {
            return None;
        };
        let from = ssa
            .bit_width(*source)
            .filter(|from| ssa.bit_width(value).is_some_and(|wide| *from < wide))?;
        T::is_integer(ssa.variable(*source)?.var_type()).then_some((
            *source,
            Self {
                from,
                unsigned: *unsigned,
            },
        ))
    }

    /// The constant `constant`, `wide` bits wide, at the narrow width as
    /// `ty`, where this extension of that gives `constant` back; `None` where
    /// it does not, and no narrow value extends to it.
    #[must_use]
    pub fn narrowed<T: Target>(
        self,
        constant: &ConstValue<T>,
        wide: u32,
        ty: &T::Type,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        let (bits, constant_width) = constant.integer_pattern(ptr_size)?;
        let shift = 64u32.checked_sub(self.from)?;
        let extended = if self.unsigned {
            bits.wrapping_shl(shift).wrapping_shr(shift)
        } else {
            (bits.wrapping_shl(shift).cast_signed() >> shift).cast_unsigned()
        };
        let kept = BitWidth::new(constant_width).mask();
        if constant_width != wide || extended & kept != bits {
            return None;
        }
        ConstValue::from_bits_of_type(bits, ty)
    }
}

/// Narrows a comparison of values extended from one narrower width to the
/// comparison of those values.
///
/// An extension keeps the order of the values it extends: a sign extension
/// both their signed and their unsigned order, a zero extension their unsigned
/// order, which is the extended values' signed order too, since none of them
/// is negative. So **a comparison of two values extended from `n` bits by one
/// kind of extension is the comparison of the values extended**, at `n` bits:
/// signed or unsigned as the wide one is for a sign extension, unsigned for a
/// zero extension; equality is equality either way. So is a comparison of
/// such a value and a constant the extension gives back, the constant at the
/// narrow width ([`NarrowedOperation::inserted`]); a constant no narrow value
/// extends to (`0x1_0000_0000` against a sign-extended word) is left to the
/// wide comparison, which decides it. A 64-bit machine without narrow
/// comparisons compares its 32-bit values whole: RV64 has no 32-bit branch,
/// so `if (i < n)` over two `int`s the `addw`s left sign-extended is a 64-bit
/// `blt` of their extensions, which a reader should see as the `int`s
/// compared.
///
/// # Arguments
///
/// * `ssa` - The function, for the operands' defining operations.
/// * `op` - The comparison to narrow: [`SsaOp::Ceq`], [`SsaOp::Clt`],
///   [`SsaOp::Cgt`] or [`SsaOp::BranchCmp`].
/// * `ptr_size` - Target pointer size, the width of the native integers.
///
/// # Returns
///
/// The comparison at the narrow width, with the constant it reads narrowed,
/// or `None` when `op` is no comparison of such values.
#[must_use]
pub fn narrow_extended_comparison<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    ptr_size: PointerSize,
) -> Option<NarrowedOperation<T>> {
    let (SsaOp::Ceq { left, right, .. }
    | SsaOp::Clt { left, right, .. }
    | SsaOp::Cgt { left, right, .. }
    | SsaOp::BranchCmp { left, right, .. }) = op
    else {
        return None;
    };
    let wide = ssa
        .bit_width(*left)
        .filter(|wide| ssa.bit_width(*right) == Some(*wide))?;
    let (left_extension, right_extension) = (Extension::of(ssa, *left), Extension::of(ssa, *right));
    let (source, extension) = match (left_extension, right_extension) {
        (Some((source, extension)), Some((_, other))) if extension == other => (source, extension),
        (Some(extended), None) | (None, Some(extended)) => extended,
        _ => return None,
    };
    let ty = ssa.variable(source)?.var_type().clone();
    let mut inserted = Vec::new();
    let mut narrow = |value: SsaVarId, extended: Option<(SsaVarId, Extension)>| {
        if let Some((source, _)) = extended {
            return Some(source);
        }
        let SsaOp::Const {
            value: constant, ..
        } = ssa.recorded_definition(value)?.instruction().op()
        else {
            return None;
        };
        let narrowed = extension.narrowed(constant, wide, &ty, ptr_size)?;
        inserted.push((value, NarrowValue::Constant(narrowed)));
        Some(value)
    };
    let (narrow_left, narrow_right) = (
        narrow(*left, left_extension)?,
        narrow(*right, right_extension)?,
    );
    // A zero extension's values are compared by their unsigned order whatever
    // the wide comparison read them as.
    let ordered = |unsigned: bool| unsigned || extension.unsigned;
    let mut narrowed = op.clone();
    match &mut narrowed {
        SsaOp::Ceq { left, right, .. } => (*left, *right) = (narrow_left, narrow_right),
        SsaOp::Clt {
            left,
            right,
            unsigned,
            ..
        }
        | SsaOp::Cgt {
            left,
            right,
            unsigned,
            ..
        } => {
            (*left, *right) = (narrow_left, narrow_right);
            *unsigned = ordered(*unsigned);
        }
        SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned,
            ..
        } => {
            (*left, *right) = (narrow_left, narrow_right);
            if !matches!(cmp, CmpKind::Eq | CmpKind::Ne) {
                *unsigned = ordered(*unsigned);
            }
        }
        _ => return None,
    }
    Some(NarrowedOperation {
        op: narrowed,
        inserted,
        ty,
    })
}

/// One operand of a phi [`narrowable_phis`] narrows, as the narrow phi reads
/// it.
#[derive(Debug, Clone)]
pub enum NarrowPhiOperand<T: Target> {
    /// A value already held at the narrow width: what the wide operand
    /// extended.
    Value(SsaVarId),
    /// A constant, at the narrow width, which the caller defines in the
    /// predecessor.
    Constant(ConstValue<T>),
    /// A phi narrowed with this one, by its wide result, the phi itself round
    /// a loop among them: the narrow phi reads that phi's narrow one.
    Joined(SsaVarId),
    /// A value of the wide width that is no extension, whose low bits the
    /// caller truncates to at the end of the predecessor: allowed only where
    /// nothing reads the phi past its low bits.
    Truncated(SsaVarId),
}

/// A phi whose every value is the extension of one narrower value, and the
/// narrower phi it is the extension of.
#[derive(Debug, Clone)]
pub struct PhiNarrowing<T: Target> {
    /// The block the phi is in.
    pub block: usize,
    /// The wide phi's result.
    pub result: SsaVarId,
    /// The type of the narrow phi.
    pub narrow_type: T::Type,
    /// Whether the wide phi is the zero-extension of the narrow one, rather
    /// than its sign-extension.
    pub unsigned: bool,
    /// The type of the narrow width the narrow phi is reinterpreted as before
    /// it is extended, when the joined values were: a signed reading of an
    /// unsigned value (`(int64_t)(int32_t)x` over a `uint32_t x`), which the
    /// narrow phi joins unsigned and the caller converts back before the
    /// extension, so the extension still reads its source signed.
    pub reinterpreted: Option<T::Type>,
    /// The narrow phi's operands, by predecessor block.
    pub operands: Vec<(usize, NarrowPhiOperand<T>)>,
}

/// Who reads each value: the operations reading it and whether a phi does, by
/// one scan of a function - the basis of [`narrowable_phis`]'s check that a
/// value's readers want only its low bits.
#[derive(Debug, Default)]
pub struct Readers<'a, T: Target> {
    /// The operations reading each value.
    operations: BTreeMap<SsaVarId, Vec<&'a SsaOp<T>>>,
    /// The phis reading each value, by their results.
    phis: BTreeMap<SsaVarId, Vec<SsaVarId>>,
}

impl<'a, T: Target> Readers<'a, T> {
    /// Whether phis among `phis`, by their results, are all that reads `var`.
    ///
    /// # Arguments
    ///
    /// * `var` - The value read.
    /// * `phis` - The results of the phis that may be its only readers.
    #[must_use]
    pub fn read_only_by(&self, var: SsaVarId, phis: &BTreeSet<SsaVarId>) -> bool {
        self.operations.get(&var).is_none_or(Vec::is_empty)
            && self
                .phis
                .get(&var)
                .is_some_and(|readers| readers.iter().all(|reader| phis.contains(reader)))
    }

    /// The operations reading `var`, an operation once for each of its
    /// operands `var` is; the phis reading it are not among them.
    #[must_use]
    pub fn operations(&self, var: SsaVarId) -> &[&'a SsaOp<T>] {
        self.operations.get(&var).map_or(&[], Vec::as_slice)
    }

    /// The results of the phis reading `var`.
    #[must_use]
    pub fn phis(&self, var: SsaVarId) -> &[SsaVarId] {
        self.phis.get(&var).map_or(&[], Vec::as_slice)
    }

    /// The readers of every value of `ssa`.
    #[must_use]
    pub fn of(ssa: &'a SsaFunction<T>) -> Self {
        let mut operations: BTreeMap<SsaVarId, Vec<&'a SsaOp<T>>> = BTreeMap::new();
        let mut phis: BTreeMap<SsaVarId, Vec<SsaVarId>> = BTreeMap::new();
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    phis.entry(operand.value()).or_default().push(phi.result());
                }
            }
            for instruction in block.instructions() {
                let op = instruction.op();
                op.for_each_use(|var| operations.entry(var).or_default().push(op));
            }
        }
        Self { operations, phis }
    }

    /// Whether every reader of `var` reads only its low `width` bits: a
    /// conversion to at most `width` bits, an operation whose low bits
    /// depend only on its operands' low bits (`+`, `-`, `*`, `&`, `|`, `^`) and
    /// whose every reader is such a conversion, or an operation `compared`
    /// says reads `var` at the narrow width. A phi reading `var` counts only
    /// when it is one of `narrowed`, the phis narrowed with `var`'s.
    fn want_low_bits(
        &self,
        var: SsaVarId,
        width: u32,
        narrowed: &BTreeSet<SsaVarId>,
        compared: impl Fn(&SsaOp<T>) -> bool,
    ) -> bool {
        let narrowing = |op: &SsaOp<T>| {
            matches!(op, SsaOp::IntConv { target, overflow_check: false, .. }
                if T::bit_width(target).is_some_and(|bits| bits <= width))
        };
        let read_narrow = |dest: SsaVarId| {
            self.phis.get(&dest).is_none_or(Vec::is_empty)
                && self
                    .operations
                    .get(&dest)
                    .is_some_and(|readers| readers.iter().all(|op| narrowing(op)))
        };
        if self
            .phis
            .get(&var)
            .is_some_and(|phis| phis.iter().any(|phi| !narrowed.contains(phi)))
        {
            return false;
        }
        self.operations.get(&var).is_none_or(|readers| {
            readers.iter().all(|op| {
                narrowing(op)
                    || compared(op)
                    || match op {
                        SsaOp::Add {
                            dest, flags: None, ..
                        }
                        | SsaOp::Sub {
                            dest, flags: None, ..
                        }
                        | SsaOp::Mul {
                            dest, flags: None, ..
                        }
                        | SsaOp::And {
                            dest, flags: None, ..
                        }
                        | SsaOp::Or {
                            dest, flags: None, ..
                        }
                        | SsaOp::Xor {
                            dest, flags: None, ..
                        } => read_narrow(*dest),
                        _ => false,
                    }
            })
        })
    }
}

/// What one phi of a web joins, as the narrowing reads it.
struct Joining<T: Target> {
    /// The narrow phi's operands, by predecessor; a constant's is a
    /// placeholder until the narrow type is known.
    operands: Vec<(usize, NarrowPhiOperand<T>)>,
    /// The constants joined, by their operand's position.
    constants: Vec<(usize, ConstValue<T>)>,
    /// Per extension and truncated value joined, the value of the same width
    /// a conversion reinterpreted into the narrow source, with that value's
    /// type.
    peeled: Vec<Option<(SsaVarId, T::Type)>>,
    /// The width, kind and source type of the extensions joined, where any
    /// are.
    narrow: Option<(u32, bool, T::Type)>,
    /// How many extensions it joins.
    extensions: usize,
    /// How many of those nothing but the narrowed phis reads, which the
    /// narrowing leaves dead.
    dying: usize,
    /// How many values it joins truncated.
    truncated: usize,
}

impl<T: Target> Joining<T> {
    /// What `phi`, `wide` bits wide, joins, where each value is a phi of
    /// `narrowed`, an extension, a constant, or a wide value no phi defines;
    /// `None` where one is anything else, or the extensions differ.
    fn of(
        ssa: &SsaFunction<T>,
        phi: &PhiNode,
        narrowed: &BTreeSet<SsaVarId>,
        readers: &Readers<'_, T>,
        wide: u32,
    ) -> Option<Self> {
        let width_of = |value: SsaVarId| ssa.bit_width(value);
        // Whether `value` is the result of a phi of its defining block.
        let joined_by_phi = |value: SsaVarId| ssa.find_phi_defining(value).is_some();
        let mut joining = Self {
            operands: Vec::with_capacity(phi.operand_count()),
            constants: Vec::new(),
            peeled: Vec::new(),
            narrow: None,
            extensions: 0,
            dying: 0,
            truncated: 0,
        };
        for operand in phi.operands() {
            if operand.is_exceptional() {
                return None;
            }
            let value = operand.value();
            if narrowed.contains(&value) {
                joining
                    .operands
                    .push((operand.predecessor(), NarrowPhiOperand::Joined(value)));
                continue;
            }
            match ssa
                .recorded_definition(value)
                .map(|definition| definition.instruction().op())
            {
                Some(SsaOp::IntConv {
                    operand: source,
                    overflow_check: false,
                    unsigned,
                    ..
                }) if width_of(*source).is_some_and(|width| width < wide) => {
                    let width = width_of(*source)?;
                    let ty = ssa.variable(*source)?.var_type().clone();
                    match &joining.narrow {
                        Some((known, kind, known_type))
                            if *known != width || *kind != *unsigned || *known_type != ty =>
                        {
                            return None;
                        }
                        Some(_) => {}
                        None => joining.narrow = Some((width, *unsigned, ty)),
                    }
                    joining.extensions = joining.extensions.saturating_add(1);
                    if readers.read_only_by(value, narrowed) {
                        joining.dying = joining.dying.saturating_add(1);
                    }
                    joining.peeled.push(
                        match ssa
                            .recorded_definition(*source)
                            .map(|d| d.instruction().op())
                        {
                            Some(SsaOp::IntConv {
                                operand: inner,
                                overflow_check: false,
                                ..
                            }) if width_of(*inner) == Some(width) => ssa
                                .variable(*inner)
                                .map(|variable| (*inner, variable.var_type().clone())),
                            _ => None,
                        },
                    );
                    joining
                        .operands
                        .push((operand.predecessor(), NarrowPhiOperand::Value(*source)));
                }
                Some(SsaOp::Const {
                    value: constant, ..
                }) => {
                    joining
                        .constants
                        .push((joining.operands.len(), constant.clone()));
                    // Filled once the narrow type is known.
                    joining
                        .operands
                        .push((operand.predecessor(), NarrowPhiOperand::Joined(value)));
                }
                // Not the value of a phi left wide: the phis a value flows
                // through are one web the lowering gives one variable, and
                // truncating where one enters this one would split it, the
                // copy this saves moved to the edge it enters by.
                _ if width_of(value) == Some(wide) && !joined_by_phi(value) => {
                    joining.truncated = joining.truncated.saturating_add(1);
                    joining.peeled.push(None);
                    joining
                        .operands
                        .push((operand.predecessor(), NarrowPhiOperand::Truncated(value)));
                }
                _ => return None,
            }
        }
        Some(joining)
    }
}

/// Narrows the webs of phis of values extended from one narrower width.
///
/// A 64-bit machine holds a 32-bit loop variable in a 64-bit register,
/// sign-extended: RV64's `int` accumulator enters the loop as the constant
/// `0xffffffff811c9dc5` its `lui`/`addi` build, and leaves each iteration as
/// the `addw`/`mulw` result's sign extension; AArch64 carries a `w` counter
/// zero-extended through every join a loop copied for each way it could go.
/// The phi joining them is a 64-bit value, so every reader sees the register
/// rather than the `int`, and the rules that narrow an operation (see
/// [`narrow_truncated_operation`]) stop at it. When every value a phi joins is
/// the extension of an `n`-bit value by one kind of extension, sign or zero,
/// the phi is that extension of the phi joining the `n`-bit values: a
/// constant joins as its `n`-bit value when the extension gives it back, and
/// a phi narrowed with it as that phi's narrow one.
///
/// The phis joining each other are one web, and the phis narrowed are the
/// greatest set of the web each of whose values is a phi of the set or one
/// of those values from outside it: a phi joining one left wide stays wide,
/// and a phi it enters joins its extension. A set joining only constants
/// and phis, or whose extensions differ in kind or width, is left as it is,
/// and so is one with no phi that operations read, and read only at its low
/// `n` bits, where the narrow phis would only add the conversions back: a phi of the set
/// something reads past those bits (a 64-bit compare of the wide value, a
/// zero-extended byte a loop compares) is narrowed with the others all the
/// same, its readers reading its extension, so one variable holds the value
/// round the loop. A phi with an exceptional edge stays wide.
///
/// A joined value of the wide width that is no extension - a loop counter
/// entering as a whole register, and carried back as its `addw`'s sign
/// extension - joins truncated to its low `n` bits when nothing reads any
/// phi of the set past them, a comparison included: the narrow phis then
/// hold every bit a reader sees, and the extensions they are redefined as
/// only differ from the wide phis in bits nobody reads. Otherwise both widths
/// would be live across the back edge, and the lowering would copy one into
/// the other there. A truncation is a row the narrowing adds, so it is made
/// only where every extension joined dies with the wide phis, nothing else
/// reading it, and at least one for each truncation; and never of the value
/// of a phi left wide, whose web the lowering gives one variable a
/// truncation would split.
///
/// # Arguments
///
/// * `ssa` - The function, for the joined values' definitions and widths.
/// * `readers` - Who reads each value of `ssa` ([`Readers::of`]).
/// * `ptr_size` - Target pointer size, the width of the native integers.
/// * `reads_narrow` - Operations the caller knows read only the narrow bits
///   of an extended value, beyond the ones every target does: a return, where
///   the convention extends a narrow result, reads the value it returns
///   exactly as well through the narrow phi's extension.
///
/// # Returns
///
/// How to narrow each phi narrowed, the phis of one web together.
#[must_use]
pub fn narrowable_phis<T: Target>(
    ssa: &SsaFunction<T>,
    readers: &Readers<'_, T>,
    ptr_size: PointerSize,
    reads_narrow: impl Fn(&SsaOp<T>) -> bool,
) -> Vec<PhiNarrowing<T>> {
    let mut phis: BTreeMap<SsaVarId, (usize, &PhiNode)> = BTreeMap::new();
    for (block_idx, block) in ssa.blocks().iter().enumerate() {
        for phi in block.phi_nodes() {
            phis.insert(phi.result(), (block_idx, phi));
        }
    }
    let mut seen = BTreeSet::new();
    let mut narrowings = Vec::new();
    for start in phis.keys() {
        if !seen.insert(*start) {
            continue;
        }
        // The phis joining each other, either way, are one web.
        let mut web = vec![*start];
        let mut at = 0;
        while let Some(&result) = web.get(at) {
            at = at.saturating_add(1);
            let Some((_, phi)) = phis.get(&result) else {
                continue;
            };
            for neighbour in phi
                .operands()
                .iter()
                .map(PhiOperand::value)
                .chain(readers.phis(result).iter().copied())
            {
                if phis.contains_key(&neighbour) && seen.insert(neighbour) {
                    web.push(neighbour);
                }
            }
        }
        narrowings.extend(narrowed_web(
            ssa,
            readers,
            &phis,
            &web,
            ptr_size,
            &reads_narrow,
        ));
    }
    narrowings
}

/// How to narrow the phis of `web` [`narrowable_phis`] narrows.
fn narrowed_web<T: Target>(
    ssa: &SsaFunction<T>,
    readers: &Readers<'_, T>,
    phis: &BTreeMap<SsaVarId, (usize, &PhiNode)>,
    web: &[SsaVarId],
    ptr_size: PointerSize,
    reads_narrow: &impl Fn(&SsaOp<T>) -> bool,
) -> Vec<PhiNarrowing<T>> {
    let Some(wide) = web.first().and_then(|first| ssa.bit_width(*first)) else {
        return Vec::new();
    };
    // The greatest set each of whose phis joins only phis of the set and
    // values narrowable from outside it.
    let mut narrowed: BTreeSet<SsaVarId> = web
        .iter()
        .copied()
        .filter(|result| ssa.bit_width(*result) == Some(wide))
        .collect();
    let mut joinings: BTreeMap<SsaVarId, Joining<T>> = BTreeMap::new();
    loop {
        joinings.clear();
        let mut failing = Vec::new();
        for result in &narrowed {
            match phis
                .get(result)
                .and_then(|(_, phi)| Joining::of(ssa, phi, &narrowed, readers, wide))
            {
                Some(joining) => {
                    joinings.insert(*result, joining);
                }
                None => failing.push(*result),
            }
        }
        if failing.is_empty() {
            break;
        }
        for result in failing {
            narrowed.remove(&result);
        }
    }
    // One extension over the set.
    let mut narrow: Option<&(u32, bool, T::Type)> = None;
    for found in joinings
        .values()
        .filter_map(|joining| joining.narrow.as_ref())
    {
        match narrow {
            Some(known) if known != found => return Vec::new(),
            Some(_) => {}
            None => narrow = Some(found),
        }
    }
    let Some((narrow_width, unsigned, source_type)) = narrow.cloned() else {
        return Vec::new();
    };
    let extension = Extension {
        from: narrow_width,
        unsigned,
    };
    let total = |count: fn(&Joining<T>) -> usize| {
        joinings
            .values()
            .map(count)
            .fold(0_usize, usize::saturating_add)
    };
    let (truncated, dying, extensions) = (
        total(|joining| joining.truncated),
        total(|joining| joining.dying),
        total(|joining| joining.extensions),
    );
    // A truth value holds 0 or 1 however many bits store it: a wide value
    // truncated to its width is no truth value, and the narrow phis could not
    // be typed as one.
    if truncated > 0 && T::is_boolean(&source_type) {
        return Vec::new();
    }
    // A comparison of a phi with a constant its extension gives back, with
    // another value extended the same way, or with a phi of the set, reads it
    // at the narrow width: narrowed, it is a comparison of extensions, which
    // compares the narrow values ([`narrow_extended_comparison`]). Not with a
    // value joined truncated: its high bits are not the narrow value's
    // extension, and a comparison of the wide phi reads them.
    let compared = |joined: SsaVarId, op: &SsaOp<T>| {
        if truncated > 0 {
            return false;
        }
        if reads_narrow(op) {
            return true;
        }
        let (SsaOp::Ceq { left, right, .. }
        | SsaOp::Clt { left, right, .. }
        | SsaOp::Cgt { left, right, .. }
        | SsaOp::BranchCmp { left, right, .. }) = op
        else {
            return false;
        };
        [(*left, *right), (*right, *left)]
            .into_iter()
            .filter(|(phi, _)| *phi == joined)
            .any(|(_, other)| {
                narrowed.contains(&other)
                    || Extension::of(ssa, other).is_some_and(|(_, other)| other == extension)
                    || matches!(
                        ssa.recorded_definition(other).map(|definition| definition.instruction().op()),
                        Some(SsaOp::Const { value, .. })
                            if extension.narrowed(value, wide, &source_type, ptr_size).is_some()
                    )
            })
    };
    let read_narrow: Vec<bool> = narrowed
        .iter()
        .map(|result| {
            readers.want_low_bits(*result, narrow_width, &narrowed, |op| compared(*result, op))
        })
        .collect();
    // Some operation must read a phi at the narrow width: a phi only phis of
    // the set read wants its low bits of nobody.
    let profits = narrowed
        .iter()
        .zip(&read_narrow)
        .any(|(result, narrow)| *narrow && !readers.operations(*result).is_empty());
    if !profits {
        return Vec::new();
    }
    // A truncation is a row the narrowing adds, so it pays only where the
    // extensions joined die with the wide phis, at least one for each, and
    // where no phi is read past the narrow bits its truncated values lose.
    if truncated > 0
        && (!read_narrow.iter().all(|narrow| *narrow) || dying < extensions || dying < truncated)
    {
        return Vec::new();
    }
    // When every joined value reinterprets one value type at the same width,
    // the narrow phis join those, and are reinterpreted back before the
    // extension.
    let peeled: Vec<&Option<(SsaVarId, T::Type)>> = joinings
        .values()
        .flat_map(|joining| &joining.peeled)
        .collect();
    let peeled_type = peeled
        .first()
        .and_then(|entry| entry.as_ref())
        .map(|(_, ty)| ty.clone());
    let reinterpreted = match &peeled_type {
        Some(ty)
            if *ty != source_type
                && peeled
                    .iter()
                    .all(|entry| entry.as_ref().is_some_and(|(_, t)| t == ty)) =>
        {
            Some(source_type.clone())
        }
        _ => None,
    };
    let narrow_type = match (&reinterpreted, peeled_type) {
        (Some(_), Some(ty)) => ty,
        _ => source_type,
    };
    let mut narrowings = Vec::with_capacity(joinings.len());
    for (result, mut joining) in joinings {
        if reinterpreted.is_some() {
            let mut values = joining.peeled.iter().flatten();
            for (_, operand) in &mut joining.operands {
                if let NarrowPhiOperand::Value(value) = operand {
                    let Some((inner, _)) = values.next() else {
                        return Vec::new();
                    };
                    *value = *inner;
                }
            }
        }
        // The extension must give each constant back, or the join of the
        // narrow values is not the join of the wide ones.
        for (at, constant) in &joining.constants {
            let Some(narrowed) = extension.narrowed(constant, wide, &narrow_type, ptr_size) else {
                return Vec::new();
            };
            let Some(slot) = joining.operands.get_mut(*at) else {
                return Vec::new();
            };
            slot.1 = NarrowPhiOperand::Constant(narrowed);
        }
        let Some((block, _)) = phis.get(&result) else {
            return Vec::new();
        };
        narrowings.push(PhiNarrowing {
            block: *block,
            result,
            narrow_type: narrow_type.clone(),
            unsigned,
            reinterpreted: reinterpreted.clone(),
            operands: joining.operands,
        });
    }
    narrowings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            block::SsaBlock,
            function::SsaFunction,
            instruction::SsaInstruction,
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockConvLink, MockConversionChain, MockTarget, MockType},
    };

    /// The four integer widths the mock target offers, which is what makes the
    /// (width, signedness) space a space rather than a pair.
    const WIDTHS: [MockType; 4] = [MockType::I8, MockType::I16, MockType::I32, MockType::I64];

    /// Returns the variable the collapsed conversion reads, or `None` when the
    /// rule refused or answered with a copy.
    fn collapsed_operand(op: Option<&SsaOp<MockTarget>>) -> Option<SsaVarId> {
        match op {
            Some(SsaOp::IntConv { operand, .. }) => Some(*operand),
            _ => None,
        }
    }

    /// A value narrowed, widened and narrowed again is three conversions deep,
    /// and one walk must say what peeling a link per pass would need three
    /// passes to say. This is the only assertion that fails if the loop is
    /// reduced to a single peel.
    #[test]
    fn the_walk_reaches_the_source_in_one_pass() {
        // I32 → I64 → I64 → I16, all zero-extending: every link is at least as
        // wide as the result, so none of them touches a bit the result keeps.
        let chain = MockConversionChain::build(
            MockType::I32,
            &[
                MockConvLink::new(MockType::I64, true),
                MockConvLink::new(MockType::I64, true),
                MockConvLink::new(MockType::I16, true),
            ],
        );

        let collapsed = collapse_conversion_chain(chain.function(), &chain.outer_op())
            .expect("three transparent links collapse");

        assert_eq!(
            collapsed_operand(Some(&collapsed)),
            Some(chain.source()),
            "one walk must reach var(0); stopping at var(1) is a single peel"
        );
    }

    /// The walk is not "collapse the chain", it is "collapse the links that
    /// change nothing" - so it stops at the first link that does, and leaves
    /// every link inside that one in place.
    #[test]
    fn the_walk_stops_at_the_first_observable_link() {
        // I32 → I8 → I64 → I16. The I64 link is transparent; the I8 link below
        // it destroys bits 8..16 that the I16 result keeps.
        let chain = MockConversionChain::build(
            MockType::I32,
            &[
                MockConvLink::new(MockType::I8, true),
                MockConvLink::new(MockType::I64, true),
                MockConvLink::new(MockType::I16, true),
            ],
        );

        let collapsed = collapse_conversion_chain(chain.function(), &chain.outer_op())
            .expect("the outermost link is transparent");

        assert_eq!(
            collapsed_operand(Some(&collapsed)),
            Some(chain.var(1)),
            "the walk must stop on the narrowing link, not walk through it"
        );
    }

    /// A conversion to the type its operand already has converts nothing, and
    /// the answer is a copy rather than a shorter conversion.
    #[test]
    fn a_conversion_to_the_operands_own_type_becomes_a_copy() {
        let chain = MockConversionChain::build(
            MockType::I64,
            &[
                MockConvLink::new(MockType::I32, false),
                MockConvLink::new(MockType::I32, false),
            ],
        );

        let collapsed = collapse_conversion_chain(chain.function(), &chain.outer_op())
            .expect("converting to the operand's own type is an identity");

        assert!(
            matches!(collapsed, SsaOp::Copy { src, .. } if src == chain.var(1)),
            "expected a copy of the operand, got {collapsed:?}"
        );
    }

    /// A float conversion to the format its operand already has converts
    /// nothing and becomes a copy; one that changes the format stays, and so
    /// does a narrowing of a widening, which rounds nothing here but would
    /// quiet a signalling NaN on the way.
    #[test]
    fn a_float_conversion_to_its_operands_own_format_becomes_a_copy() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let live_in = |ssa: &mut SsaFunction<MockTarget>, ty: MockType| {
            ssa.create_variable(VariableOrigin::EntryLiveIn, 0, DefSite::entry(), ty)
        };
        let double = live_in(&mut ssa, MockType::F64);
        let extended = live_in(&mut ssa, MockType::F80);
        let define = |ssa: &mut SsaFunction<MockTarget>, index: usize, ty: MockType| {
            ssa.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(0, index), ty)
        };
        let same = define(&mut ssa, 0, MockType::F80);
        let widened = define(&mut ssa, 1, MockType::F80);
        let narrowed = define(&mut ssa, 2, MockType::F64);
        let identity = SsaOp::FloatConv {
            dest: same,
            operand: extended,
            target: MockType::F80,
            rounding: None,
            source: None,
            environment: None,
        };
        let widening = SsaOp::FloatConv {
            dest: widened,
            operand: double,
            target: MockType::F80,
            rounding: None,
            source: None,
            environment: None,
        };
        let round_trip = SsaOp::FloatConv {
            dest: narrowed,
            operand: widened,
            target: MockType::F64,
            rounding: None,
            source: None,
            environment: None,
        };
        let mut block = SsaBlock::new(0);
        for op in [identity.clone(), widening.clone(), round_trip.clone()] {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.refresh_def_sites();
        ssa.recompute_uses();

        assert!(
            matches!(
                collapse_conversion_chain(&ssa, &identity),
                Some(SsaOp::Copy { src, .. }) if src == extended
            ),
            "a conversion to the operand's own format is a copy"
        );
        assert_eq!(collapse_conversion_chain(&ssa, &widening), None);
        assert_eq!(collapse_conversion_chain(&ssa, &round_trip), None);
    }

    /// The condition is about the *source* width as well as the intermediate's,
    /// and dropping that half is how `(uint32_t)(int64_t)x` on an 8-bit `x`
    /// silently becomes a different number: the intermediate sign-extends, and
    /// the collapsed form would zero those bits instead.
    #[test]
    fn a_conversion_wider_than_its_source_is_never_collapsed_through() {
        // x: 8 bits, sign-extended to 64, read back as an unsigned 32. The
        // result's bits 8..32 are sign copies; the collapsed form would zero
        // them.
        let chain = MockConversionChain::build(
            MockType::I8,
            &[
                MockConvLink::new(MockType::I64, false),
                MockConvLink::new(MockType::I32, true),
            ],
        );

        assert!(
            collapse_conversion_chain(chain.function(), &chain.outer_op()).is_none(),
            "the extension is observable in the result"
        );
    }

    /// Above the source width the bits are an extension either way, so the two
    /// conversions are interchangeable when they agree on which extension.
    #[test]
    fn a_widening_pair_collapses_when_they_agree_on_the_extension() {
        // x: 8 bits, zero-extended to 64 and read back as an unsigned 32.
        let chain = MockConversionChain::build(
            MockType::I8,
            &[
                MockConvLink::new(MockType::I64, true),
                MockConvLink::new(MockType::I32, true),
            ],
        );

        assert_eq!(
            collapsed_operand(
                collapse_conversion_chain(chain.function(), &chain.outer_op()).as_ref()
            ),
            Some(chain.source()),
            "two zero-extensions to the same value are one zero-extension"
        );
    }

    /// A restatement at the source's width keeps every bit and extends none,
    /// so a wider result reads through it and extends by its own flag: a
    /// signed 32-bit value restated unsigned and then sign-extended is its own
    /// sign-extension. The control is a link one width narrower, which the
    /// same result may not read through.
    #[test]
    fn a_restatement_at_the_sources_width_is_read_through_at_any_width() {
        let chain = MockConversionChain::build(
            MockType::I32,
            &[
                MockConvLink::new(MockType::U32, true),
                MockConvLink::new(MockType::I64, false),
            ],
        );
        assert_eq!(
            collapsed_operand(
                collapse_conversion_chain(chain.function(), &chain.outer_op()).as_ref()
            ),
            Some(chain.source()),
        );

        let control = MockConversionChain::build(
            MockType::I32,
            &[
                MockConvLink::new(MockType::I16, true),
                MockConvLink::new(MockType::I64, false),
            ],
        );
        assert!(collapse_conversion_chain(control.function(), &control.outer_op()).is_none());
    }

    /// A restatement that states the sign the result extends by is where that
    /// sign is said, and stays: an unsigned 32-bit value restated signed and
    /// then sign-extended (RV64's `sext.w`) keeps its restatement.
    #[test]
    fn a_restatement_agreeing_with_the_extension_stays() {
        let chain = MockConversionChain::build(
            MockType::U32,
            &[
                MockConvLink::new(MockType::I32, true),
                MockConvLink::new(MockType::I64, false),
            ],
        );
        assert!(collapse_conversion_chain(chain.function(), &chain.outer_op()).is_none());
    }

    /// An intermediate that narrows below the result destroys bits the result
    /// keeps, so it is not an intermediate at all.
    #[test]
    fn a_narrowing_conversion_is_never_collapsed_through() {
        let chain = MockConversionChain::build(
            MockType::I64,
            &[
                MockConvLink::new(MockType::I32, false),
                MockConvLink::new(MockType::I64, false),
            ],
        );

        assert!(
            collapse_conversion_chain(chain.function(), &chain.outer_op()).is_none(),
            "the narrowing is what the outer conversion reads"
        );
    }

    /// The overflow check is the point of a checked conversion, so the checked
    /// conversion is never the one removed - not even when its widths would
    /// otherwise make it transparent.
    #[test]
    fn an_overflow_checked_outer_conversion_is_refused() {
        let chain = MockConversionChain::build(
            MockType::I64,
            &[
                MockConvLink::new(MockType::I64, false),
                MockConvLink::checked(MockType::I32, false),
            ],
        );

        assert!(
            collapse_conversion_chain(chain.function(), &chain.outer_op()).is_none(),
            "removing a checked conversion removes an observable effect"
        );
    }

    /// A checked link stops the walk without being removed: the links outside
    /// it still collapse, and the check itself stays in the function feeding
    /// the survivor.
    #[test]
    fn an_overflow_checked_link_stops_the_walk_without_removing_it() {
        let chain = MockConversionChain::build(
            MockType::I64,
            &[
                MockConvLink::checked(MockType::I64, false),
                MockConvLink::new(MockType::I64, false),
                MockConvLink::new(MockType::I32, false),
            ],
        );

        let collapsed = collapse_conversion_chain(chain.function(), &chain.outer_op())
            .expect("the unchecked link above the check still collapses");

        assert_eq!(
            collapsed_operand(Some(&collapsed)),
            Some(chain.var(1)),
            "the walk stops on the checked link"
        );
        assert!(
            chain.function().all_instructions().any(|instr| matches!(
                instr.op(),
                SsaOp::IntConv {
                    overflow_check: true,
                    ..
                }
            )),
            "the checked conversion is still there to be executed"
        );
    }

    /// A def site is variable *metadata*, and an edit can leave it naming an
    /// instruction that defines something else. Reading that instruction's
    /// widths rewrites the outer conversion to an operand that never fed it -
    /// silent wrong code, not a missed optimization - so the guarded lookup
    /// stops the walk instead.
    #[test]
    fn a_stale_def_site_stops_the_walk() {
        // 0: v0 = const (I64)
        // 1: v1 = (I64) v0      -- transparent, and *not* v2's definition
        // 2: v2 = (I32) v1      -- v2's real definition; narrows below the result
        // 3: v3 = (I64) v2      -- the outer conversion
        let mut chain = MockConversionChain::build(
            MockType::I64,
            &[
                MockConvLink::new(MockType::I64, false),
                MockConvLink::new(MockType::I32, false),
                MockConvLink::new(MockType::I64, false),
            ],
        );
        let (intermediate, source) = (chain.var(2), chain.source());

        assert!(
            collapse_conversion_chain(chain.function(), &chain.outer_op()).is_none(),
            "undamaged, the narrowing link already refuses"
        );

        // Stale by one: v2 now names the instruction that defines v1.
        chain
            .function_mut()
            .variable_mut(intermediate)
            .expect("the intermediate exists")
            .set_def_site(DefSite::instruction(0, 1));

        let collapsed = collapse_conversion_chain(chain.function(), &chain.outer_op());
        assert!(
            collapsed.is_none(),
            "a stale def site must stop the walk; reading v1's widths would rewrite \
             the conversion to read {source:?} and skip the narrowing entirely: {collapsed:?}"
        );
    }

    /// Malformed IR can hand the walk a def-use cycle, where every def site is
    /// exact and following them never ends. The step bound is
    /// `var_id_bound()`, which a well-formed chain cannot reach because it
    /// would have to revisit a variable.
    ///
    /// Without the bound this test does not fail - it hangs.
    #[test]
    fn a_cyclic_def_use_chain_terminates_and_refuses() {
        let mut chain = MockConversionChain::build(
            MockType::I64,
            &[
                MockConvLink::new(MockType::I64, false),
                MockConvLink::new(MockType::I64, false),
                MockConvLink::new(MockType::I32, false),
            ],
        );
        let (source, second) = (chain.source(), chain.var(2));

        // v1 = (I64) v2 while v2 = (I64) v1: both def sites still exact.
        let replaced = chain
            .function_mut()
            .block_mut(0)
            .expect("the block exists")
            .instruction_mut(1)
            .expect("the first link exists")
            .op_mut()
            .replace_uses(source, second);
        assert_eq!(replaced, 1, "the first link now reads the second's result");

        assert!(
            collapse_conversion_chain(chain.function(), &chain.outer_op()).is_none(),
            "a cyclic def-use chain must terminate and refuse"
        );
    }

    /// Low `width` bits set.
    fn mask(width: u32) -> u64 {
        if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        }
    }

    /// Models one `IntConv`: the low `to` bits of the operand, extended above
    /// the operand's own width by sign or by zero.
    fn convert(value: u64, from: u32, to: u32, unsigned: bool) -> u64 {
        if to <= from || unsigned {
            return value & mask(to);
        }
        let sign_bit = 1u64 << (from - 1);
        let extended = if value & sign_bit == 0 {
            value
        } else {
            value | !mask(from)
        };
        extended & mask(to)
    }

    /// Bit patterns that sit on the boundaries a conversion can move: zero,
    /// one, the sign bit, the largest positive value, all ones, and two
    /// alternating patterns, at every narrower width as well as the source's.
    fn probes(width: u32) -> Vec<u64> {
        let mut values = vec![
            0,
            1,
            mask(width),
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
        ];
        for boundary in [8u32, 16, 32, 64] {
            if boundary <= width {
                values.push(1u64 << (boundary - 1));
                values.push(mask(boundary - 1));
                values.push(mask(boundary));
                values.push((1u64 << (boundary - 1)) | 1);
            }
        }
        values.iter().map(|value| value & mask(width)).collect()
    }

    fn width_of(ty: MockType) -> u32 {
        MockTarget::bit_width(&ty).expect("every matrix type has a width")
    }

    /// The gate: over every three-link shape the mock target can express, a
    /// collapse the rule accepts must compute what the chain computed, on every
    /// probe.
    ///
    /// 4 source widths × (4 targets × 2 signednesses)³ = 2048 shapes. The
    /// hand-written tests above pin the interesting shapes; this pins the space,
    /// which is what the two-link ceiling escaped for as long as it did.
    ///
    /// The dual assertion is about depth, not values: whenever the outer width
    /// clears every link and the source, and the identity copy does not
    /// intercept, the walk has to arrive at the source. That is false under a
    /// single peel.
    #[test]
    fn every_accepted_collapse_agrees_with_the_chain_over_the_width_matrix() {
        let links: Vec<MockConvLink> = WIDTHS
            .iter()
            .flat_map(|target| {
                [
                    MockConvLink::new(*target, false),
                    MockConvLink::new(*target, true),
                ]
            })
            .collect();

        for source in WIDTHS {
            let source_width = width_of(source);
            let probe_values = probes(source_width);

            for outer in &links {
                for middle in &links {
                    for inner in &links {
                        let shape = [*inner, *middle, *outer];
                        let chain = MockConversionChain::build(source, &shape);
                        let Some(collapsed) =
                            collapse_conversion_chain(chain.function(), &chain.outer_op())
                        else {
                            continue;
                        };

                        // Widths of v0..v3, and the variable each one is.
                        let mut widths = vec![source_width];
                        widths.extend(shape.iter().map(|link| width_of(link.target)));
                        let vars: Vec<SsaVarId> = (0..=shape.len()).map(|i| chain.var(i)).collect();

                        for probe in &probe_values {
                            // Evaluate the chain link by link.
                            let mut values = vec![*probe];
                            for (index, link) in shape.iter().enumerate() {
                                let value = convert(
                                    values[index],
                                    widths[index],
                                    widths[index + 1],
                                    link.unsigned,
                                );
                                values.push(value);
                            }
                            let expected = *values.last().expect("the chain has a result");

                            let actual = match &collapsed {
                                SsaOp::Copy { src, .. } => {
                                    let index = vars
                                        .iter()
                                        .position(|var| var == src)
                                        .expect("a copy reads a chain variable");
                                    values[index]
                                }
                                SsaOp::IntConv {
                                    operand,
                                    target,
                                    unsigned,
                                    ..
                                } => {
                                    let index = vars
                                        .iter()
                                        .position(|var| var == operand)
                                        .expect("the collapse reads a chain variable");
                                    convert(
                                        values[index],
                                        widths[index],
                                        width_of(*target),
                                        *unsigned,
                                    )
                                }
                                other => panic!("unexpected collapse shape {other:?}"),
                            };

                            assert_eq!(
                                actual, expected,
                                "collapse disagrees with the chain: source {source:?}, \
                                 links {shape:?}, probe {probe:#x}, collapsed {collapsed:?}"
                            );
                        }

                        // Depth dual.
                        let outer_width = width_of(outer.target);
                        let clears_every_link = outer_width <= source_width
                            && outer_width <= width_of(inner.target)
                            && outer_width <= width_of(middle.target);
                        if clears_every_link && middle.target != outer.target {
                            assert_eq!(
                                collapsed_operand(Some(&collapsed)),
                                Some(chain.source()),
                                "every link is transparent, so the walk must reach the source: \
                                 source {source:?}, links {shape:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    /// A narrowing conversion of a byte insert reads only the byte.
    ///
    /// `(u8)((wide & ~0xff) | (narrow & 0xff))` is how a target with no
    /// sub-register write spells "replace the low byte", and the wide disjunct
    /// contributes nothing the conversion keeps. Reading it anyway makes the
    /// result look as wide as the register it was inserted into.
    #[test]
    fn a_narrowing_conversion_drops_the_disjunct_it_discards() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let define = |ssa: &mut SsaFunction<MockTarget>, index: usize, ty: MockType| {
            ssa.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(0, index), ty)
        };
        // The two values being merged are caller-supplied, not literals: the
        // rule reads the *mask* of each conjunct, and a literal on the other
        // side would answer for it.
        let live_in = |ssa: &mut SsaFunction<MockTarget>| {
            ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::I32,
            )
        };
        let wide = live_in(&mut ssa);
        let narrow = live_in(&mut ssa);
        let high_mask = define(&mut ssa, 0, MockType::I32);
        let low_mask = define(&mut ssa, 1, MockType::I32);
        let kept_high = define(&mut ssa, 2, MockType::I32);
        let kept_low = define(&mut ssa, 3, MockType::I32);
        let merged = define(&mut ssa, 4, MockType::I32);
        let narrowed = define(&mut ssa, 5, MockType::I8);

        let mut block = SsaBlock::new(0);
        for op in [
            SsaOp::Const {
                dest: high_mask,
                value: ConstValue::I32(-256),
            },
            SsaOp::Const {
                dest: low_mask,
                value: ConstValue::I32(255),
            },
            SsaOp::And {
                dest: kept_high,
                left: wide,
                right: high_mask,
                flags: None,
            },
            SsaOp::And {
                dest: kept_low,
                left: narrow,
                right: low_mask,
                flags: None,
            },
            SsaOp::Or {
                dest: merged,
                left: kept_high,
                right: kept_low,
                flags: None,
            },
        ] {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        let conversion = SsaOp::IntConv {
            dest: narrowed,
            operand: merged,
            target: MockType::I8,
            overflow_check: false,
            unsigned: true,
        };
        block.add_instruction(SsaInstruction::synthetic(conversion.clone()));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.refresh_def_sites();
        ssa.recompute_uses();

        assert_eq!(
            collapsed_operand(drop_discarded_disjunct(&ssa, &conversion).as_ref()),
            Some(kept_low),
            "the byte the insert wrote is the whole of an 8-bit read"
        );

        // A wider read keeps both: at 16 bits the high mask still names bits
        // the conversion returns.
        let wider = SsaOp::IntConv {
            dest: narrowed,
            operand: merged,
            target: MockType::I16,
            overflow_check: false,
            unsigned: true,
        };
        assert_eq!(
            drop_discarded_disjunct(&ssa, &wider),
            None,
            "a disjunct whose bits the conversion keeps is not discarded"
        );
    }

    /// **A narrowing conversion of a composed pair reads only its low half.**
    /// `(u64)(((u128)hi << 64) | (u128)lo)` is a result returned beside a
    /// value the body left in the register after it, narrowed back to the
    /// first register: the shifted half has no bit below 64. Shifted by less,
    /// it keeps bits the conversion returns and is read.
    #[test]
    fn a_narrowing_conversion_drops_a_disjunct_shifted_past_it() {
        let build = |shift: u8| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
            let define = |ssa: &mut SsaFunction<MockTarget>, index: usize, ty: MockType| {
                ssa.create_variable(VariableOrigin::Phi, 0, DefSite::instruction(0, index), ty)
            };
            let live_in = |ssa: &mut SsaFunction<MockTarget>| {
                ssa.create_variable(
                    VariableOrigin::EntryLiveIn,
                    0,
                    DefSite::entry(),
                    MockType::U64,
                )
            };
            let (high, low) = (live_in(&mut ssa), live_in(&mut ssa));
            let wide_high = define(&mut ssa, 0, MockType::U128);
            let amount = define(&mut ssa, 1, MockType::U8);
            let shifted = define(&mut ssa, 2, MockType::U128);
            let wide_low = define(&mut ssa, 3, MockType::U128);
            let pair = define(&mut ssa, 4, MockType::U128);
            let first = define(&mut ssa, 5, MockType::U64);
            let mut block = SsaBlock::new(0);
            for op in [
                SsaOp::int_conv(wide_high, high, MockType::U128, true),
                SsaOp::Const {
                    dest: amount,
                    value: ConstValue::U8(shift),
                },
                SsaOp::Shl {
                    dest: shifted,
                    value: wide_high,
                    amount,
                    flags: None,
                },
                SsaOp::int_conv(wide_low, low, MockType::U128, true),
                SsaOp::Or {
                    dest: pair,
                    left: shifted,
                    right: wide_low,
                    flags: None,
                },
            ] {
                block.add_instruction(SsaInstruction::synthetic(op));
            }
            let conversion = SsaOp::int_conv(first, pair, MockType::U64, true);
            block.add_instruction(SsaInstruction::synthetic(conversion.clone()));
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
            ssa.refresh_def_sites();
            ssa.recompute_uses();
            (ssa, conversion, wide_low)
        };
        let (ssa, conversion, wide_low) = build(64);
        assert_eq!(
            collapsed_operand(drop_discarded_disjunct(&ssa, &conversion).as_ref()),
            Some(wide_low),
            "the half shifted past 64 bits is none of the first register's"
        );
        let (ssa, conversion, _) = build(32);
        assert_eq!(
            drop_discarded_disjunct(&ssa, &conversion),
            None,
            "a half shifted by 32 still sets bits the conversion keeps"
        );
    }
}
