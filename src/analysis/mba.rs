//! Folding mixed boolean-arithmetic identities back to the operation they spell.
//!
//! An obfuscator rewrites `a + b` as `(a ^ b) + 2 * (a & b)`, `a - b` as
//! `a + ~b + 1`, `a ^ b` as `(a | b) - (a & b)`, and so on: a simple operation
//! spelled as a tangle of boolean and arithmetic ones that computes the same
//! value for every input. Recovered code that reproduces the tangle has
//! reproduced the obfuscation; the point of recovery is to hand back the
//! operation.
//!
//! Each rule here is a ring identity modulo `2^n`: it holds for every input
//! *at one width*. A match is sound to rewrite when every value the identity
//! names is read at that width, which is what the preconditions below enforce.
//!
//! # What a fold requires
//!
//! The operands of the two sub-terms must be the *same SSA values*. `a ^ b` and
//! `a & b` fold to `a + b` only when both read the same `a` and the same `b`;
//! two independent reads of a `volatile` are two different values and must not
//! be assumed equal, so an obfuscation built on volatile operands correctly
//! does not fold here. The identity is on values, and SSA equality is how a
//! value is known to be the same one.
//!
//! Every value involved must have one width: the outer operation's destination,
//! the operands the fold hands back, and each sub-term between them. A
//! conversion that changes width changes which bits the outer operation sees -
//! `(u64)(a ^ b) + 2 * (u64)(a & b)` over 32-bit operands is the 33-bit sum,
//! not the 32-bit `a + b` - so `peel` looks through a conversion only when it
//! keeps the width, and a width that is not statically known declines the fold.
//!
//! The replacement sets flags differently from the tangle (`a + ~b + 1` and
//! `a - b` disagree on the carry), so a fold also requires the top operation's
//! flags output, when it has one, to be unread.

use std::collections::BTreeMap;

use crate::{
    analysis::operands::peel,
    ir::{SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    pointer::PointerSize,
    target::Target,
};

/// The rewrite an MBA identity yields: the operation the tangle computes.
pub struct MbaFold<T: Target> {
    /// The operation to replace the matched node with, keeping its `dest`.
    pub op: SsaOp<T>,
    /// The identity that matched, for the pass's event log.
    pub description: &'static str,
    /// Definitions inside the tangle that [`Self::op`] reads, rewritten in
    /// place before it, keyed by the variable each defines.
    ///
    /// Usually empty: an identity whose terms the tangle already holds is one
    /// operation replacing the matched node. A *widened* tangle is not. The
    /// replacement reads its terms at the width the result is written at, and
    /// a tangle that widens after combining holds only the combinations at
    /// that width, so the terms have to be recovered before the replacement
    /// can name them. Where the tangle already contains definitions that
    /// produce values of exactly the right type - the conversions that widened
    /// it - redirecting those recovers the terms and creates nothing, which is
    /// why this is a list of rewrites rather than a list of values to
    /// materialise.
    ///
    /// Every entry is a definition the match proved is read only from inside
    /// the tangle, so changing what it holds is not observable elsewhere.
    pub also: Vec<(SsaVarId, SsaOp<T>)>,
    /// The type the destination takes, when [`Self::op`] computes a different
    /// kind of value than the matched node did.
    ///
    /// `None` for every identity whose tangle was already integer arithmetic.
    /// A lea-lowered tangle is not: its outer sum lifted as an address, so its
    /// destination is pointer-typed, and the identity proves the sum is integer
    /// addition. Kept as a pointer, the destination would claim an address the
    /// program never formed, and every reader of its type would believe it.
    pub dest_type: Option<T::Type>,
}

/// A binary sub-term resolved from an operand's defining instruction.
#[derive(Debug)]
enum Term {
    /// `a ^ b`.
    Xor(SsaVarId, SsaVarId),
    /// `a & b`.
    And(SsaVarId, SsaVarId),
    /// `a | b`.
    Or(SsaVarId, SsaVarId),
    /// `a & b` shifted left by one or multiplied by two - twice the carries.
    TwiceAnd(SsaVarId, SsaVarId),
    /// `a + b`.
    Add(SsaVarId, SsaVarId),
    /// `~a` - the one's complement, the `a + ~b + 1` identity's middle term.
    Not(SsaVarId),
}

/// Whether two unordered operand pairs name the same values.
fn same(a: (SsaVarId, SsaVarId), b: (SsaVarId, SsaVarId)) -> bool {
    (a.0 == b.0 && a.1 == b.1) || (a.0 == b.1 && a.1 == b.0)
}

/// The one width every one of `vars` is read whole at: the width those of a
/// stated type agree on, a value of none being the bits it is read as
/// ([`SsaFunction::is_read_whole_at`]); `None` where none states one.
fn one_width<T: Target>(ssa: &SsaFunction<T>, vars: &[SsaVarId]) -> Option<u32> {
    let known = vars.iter().find_map(|var| ssa.bit_width(*var))?;
    vars.iter()
        .all(|var| ssa.is_read_whole_at(*var, known))
        .then_some(known)
}

/// Resolves `var` to a recognised binary sub-term via its defining op.
fn term<T: Target>(
    ssa: &SsaFunction<T>,
    var: SsaVarId,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<Term> {
    let var = peel(ssa, var);
    match ssa.get_definition(var)? {
        SsaOp::Xor { left, right, .. } => Some(Term::Xor(*left, *right)),
        SsaOp::And { left, right, .. } => Some(Term::And(*left, *right)),
        SsaOp::Or { left, right, .. } => Some(Term::Or(*left, *right)),
        // `(a & b) + (a & b)` is twice the carries, the doubling x86 emits at
        // `-O0` where other targets shift or multiply. Checked before the
        // general `Add` below, which would otherwise swallow it.
        SsaOp::Add { left, right, .. }
            if peel(ssa, *left) == peel(ssa, *right)
                && matches!(
                    ssa.get_definition(peel(ssa, *left)),
                    Some(SsaOp::And { .. })
                ) =>
        {
            match ssa.get_definition(peel(ssa, *left))? {
                SsaOp::And { left, right, .. } => Some(Term::TwiceAnd(*left, *right)),
                _ => None,
            }
        }
        SsaOp::Add { left, right, .. } => Some(Term::Add(*left, *right)),
        SsaOp::Not { operand, .. } => Some(Term::Not(*operand)),
        // `(a & b) << 1` is twice the carries.
        SsaOp::Shl { value, amount, .. }
            if constants.get(amount).is_some_and(ConstValue::is_one) =>
        {
            match ssa.get_definition(peel(ssa, *value))? {
                SsaOp::And { left, right, .. } => Some(Term::TwiceAnd(*left, *right)),
                _ => None,
            }
        }
        // `(a & b) * 2`, either operand order.
        SsaOp::Mul { left, right, .. } => {
            let (anded, two) = if constants.get(right).is_some_and(is_two) {
                (*left, true)
            } else if constants.get(left).is_some_and(is_two) {
                (*right, true)
            } else {
                (*left, false)
            };
            if !two {
                return None;
            }
            match ssa.get_definition(peel(ssa, anded))? {
                SsaOp::And { left, right, .. } => Some(Term::TwiceAnd(*left, *right)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether a constant is exactly two.
fn is_two<T: Target>(c: &ConstValue<T>) -> bool {
    c.as_i64() == Some(2)
}

/// Attempts to fold `op` if it is the top of an MBA identity.
///
/// `op` keeps its destination; only what computes into it changes.
///
/// # Arguments
///
/// * `ssa` - The function, for resolving each operand's defining instruction
///   and reading the widths of the values involved.
/// * `op` - The candidate operation.
/// * `constants` - Known constant values, for the `+ 1`, `<< 1` and `* 2`
///   literals.
/// * `uses` - Use counts for the whole function, as
///   [`SsaFunction::count_uses`] reports them, to tell whether `op`'s flags
///   output is read.
/// * `ptr_size` - Target pointer width, the width of a `PtrAdd`'s sum.
///
/// # Returns
///
/// The lea-lowered carry-sum over terms narrower than the address.
///
/// The [`fold`] arm above folds `base + 2*index` to `a + b` when the terms are
/// already as wide as the address. They usually are not: x86-64 takes the
/// boolean operations at the operands' own width and zero-extends each result
/// into the address, so `a` and `b` are 32-bit and the sum is 64-bit. Folding
/// *that* to a 32-bit `Add` would be wrong, and is refused - the address keeps
/// the carry out of bit 31 that a 32-bit sum drops.
///
/// It is still the same identity, taken one width up. `(a ^ b) + 2*(a & b)`
/// equals `a + b` over the integers with no wrap at all, so over the naturals
/// the zero-extended tangle equals `zext(a) + zext(b)` exactly - carry
/// included. Only zero-extension: sign-extending the boolean results is a
/// different value, and the identity says nothing about it.
///
/// The replacement therefore stays at the address's width and reads the terms
/// widened, which the tangle does not hold - it holds the *boolean results*
/// widened. But the two conversions that widened them already produce values
/// of exactly the right type, and the match below proves each is read only by
/// this address, so redirecting each onto one operand recovers the terms
/// without creating a variable. An `Add` at the address's width then sums
/// them, and the destination is retyped as that width's integer (see
/// [`MbaFold::dest_type`]).
///
/// # Arguments
///
/// * `ssa` - The function, for resolving definitions.
/// * `uses` - Use counts, which decide whether redirecting a conversion is
///   observable anywhere else.
/// * `dest` - The address's destination, which the replacement keeps.
/// * `base` - The widened `a ^ b`.
/// * `index` - The widened `a & b`.
fn widened_carry_sum<T: Target>(
    ssa: &SsaFunction<T>,
    uses: &BTreeMap<SsaVarId, usize>,
    dest: SsaVarId,
    base: SsaVarId,
    index: SsaVarId,
) -> Option<MbaFold<T>> {
    let used_once = |var: SsaVarId| uses.get(&var).copied().unwrap_or(0) == 1;
    if !used_once(base) || !used_once(index) {
        return None;
    }
    let widening = |var: SsaVarId| match ssa.get_definition(var) {
        Some(SsaOp::IntConv {
            operand,
            target,
            overflow_check: false,
            unsigned: true,
            ..
        }) => Some((*operand, target.clone())),
        _ => None,
    };
    let (sum, sum_type) = widening(base)?;
    let (carry, carry_type) = widening(index)?;
    if sum_type != carry_type {
        return None;
    }
    let (
        Some(SsaOp::Xor {
            left: x, right: y, ..
        }),
        Some(SsaOp::And {
            left: p, right: q, ..
        }),
    ) = (
        ssa.get_definition(peel(ssa, sum)),
        ssa.get_definition(peel(ssa, carry)),
    )
    else {
        return None;
    };
    let (x, y) = (peel(ssa, *x), peel(ssa, *y));
    let (p, q) = (peel(ssa, *p), peel(ssa, *q));
    if (x, y) != (p, q) && (x, y) != (q, p) {
        return None;
    }
    // The two terms are combined at one width before the widening, and that is
    // the width the identity is taken at.
    one_width(ssa, &[x, y])?;
    let convert = |dest, operand| SsaOp::IntConv {
        dest,
        operand,
        target: sum_type.clone(),
        overflow_check: false,
        unsigned: true,
    };
    Some(MbaFold {
        op: SsaOp::Add {
            dest,
            left: base,
            right: index,
            flags: None,
        },
        description: "mba (widened lea) → add",
        also: vec![(base, convert(base, x)), (index, convert(index, y))],
        dest_type: Some(sum_type),
    })
}

/// The operation an identity proves a tangle computes.
#[derive(Debug, Clone, Copy)]
enum Combined {
    /// `a + b`.
    Add,
    /// `a ^ b`.
    Xor,
    /// `a & b`.
    And,
    /// `a | b`.
    Or,
}

/// A term computed at a wider width than the tangle combines it at, read
/// through the conversion that narrows it: the term, the conversion's result,
/// and the wide value under it.
///
/// `(a & b) << 1` at the narrow width over a narrowed `a & b` is twice the
/// carries too, since doubling the low bits is the low bits of the doubling.
fn narrowed_term<T: Target>(
    ssa: &SsaFunction<T>,
    var: SsaVarId,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<(Term, SsaVarId, SsaVarId)> {
    let narrowing = |var: SsaVarId| -> Option<(SsaVarId, SsaVarId)> {
        let var = peel(ssa, var);
        match ssa.get_definition(var)? {
            SsaOp::IntConv {
                operand,
                overflow_check: false,
                ..
            } if ssa.bit_width(var)? < ssa.bit_width(*operand)? => Some((var, *operand)),
            _ => None,
        }
    };
    let var = peel(ssa, var);
    if let Some(SsaOp::Shl { value, amount, .. }) = ssa.get_definition(var)
        && constants.get(amount).is_some_and(ConstValue::is_one)
    {
        let (conv, wide) = narrowing(*value)?;
        return match ssa.get_definition(wide)? {
            SsaOp::And { left, right, .. } => Some((Term::TwiceAnd(*left, *right), conv, wide)),
            _ => None,
        };
    }
    let (conv, wide) = narrowing(var)?;
    Some((term(ssa, wide, constants)?, conv, wide))
}

/// The identities of [`fold`] over terms computed wider than the tangle
/// combines them.
///
/// RISC-V has no 32-bit `xor`, `and` or `or`: an `int` tangle computes its
/// boolean terms at the register's 64 bits, narrows each to 32, and combines
/// them there. The identity holds at the wide width, so its result there,
/// narrowed, is the tangle's: `(a ^ b) + 2*(a & b)` over the narrowed terms is
/// the narrowed `a + b`. The replacement creates nothing: the wide term the
/// tangle reads first (the carry sum's `a ^ b`, a difference's minuend) is
/// redefined as the identity's operation at the wide width, and the matched
/// node becomes a copy of the conversion that already narrows it - both read
/// only inside the tangle, which the match checks, so nothing else sees them
/// change.
fn narrowed_tangle<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<MbaFold<T>> {
    let dest = op.dest()?;
    let (left, right) = match op {
        SsaOp::Add { left, right, .. }
        | SsaOp::Sub { left, right, .. }
        | SsaOp::Or { left, right, .. } => (*left, *right),
        _ => return None,
    };
    let (l, l_conv, l_wide) = narrowed_term(ssa, left, constants)?;
    let (r, r_conv, r_wide) = narrowed_term(ssa, right, constants)?;
    // The term (with its conversion) that is redefined to compute the wide
    // operation the identity proves the tangle computes.
    let pick = |carrier_is_left: bool| {
        if carrier_is_left {
            (l_conv, l_wide)
        } else {
            (r_conv, r_wide)
        }
    };
    let (pair, operation, carrier, description) = match (op, &l, &r) {
        (SsaOp::Add { .. }, Term::Xor(a, b), Term::TwiceAnd(c, d)) if same((*a, *b), (*c, *d)) => {
            ((*a, *b), Combined::Add, pick(true), "mba (narrowed) → add")
        }
        (SsaOp::Add { .. }, Term::TwiceAnd(c, d), Term::Xor(a, b)) if same((*a, *b), (*c, *d)) => {
            ((*a, *b), Combined::Add, pick(false), "mba (narrowed) → add")
        }
        (SsaOp::Add { .. }, Term::Or(a, b), Term::And(c, d)) if same((*a, *b), (*c, *d)) => {
            ((*a, *b), Combined::Add, pick(true), "mba (narrowed) → add")
        }
        (SsaOp::Add { .. }, Term::And(c, d), Term::Or(a, b)) if same((*a, *b), (*c, *d)) => {
            ((*a, *b), Combined::Add, pick(false), "mba (narrowed) → add")
        }
        (SsaOp::Sub { .. }, Term::Or(a, b), Term::And(c, d)) if same((*a, *b), (*c, *d)) => {
            ((*a, *b), Combined::Xor, pick(true), "mba (narrowed) → xor")
        }
        (SsaOp::Sub { .. }, Term::Add(a, b), Term::Or(c, d)) if same((*a, *b), (*c, *d)) => {
            ((*a, *b), Combined::And, pick(true), "mba (narrowed) → and")
        }
        (SsaOp::Or { .. }, Term::And(a, b), Term::Xor(c, d))
        | (SsaOp::Or { .. }, Term::Xor(c, d), Term::And(a, b))
            if same((*a, *b), (*c, *d)) =>
        {
            ((*a, *b), Combined::Or, pick(true), "mba (narrowed) → or")
        }
        _ => return None,
    };
    let (carrier_conv, carrier_wide) = carrier;
    let used_once = |var: SsaVarId| uses.get(&var).copied().unwrap_or(0) == 1;
    if !used_once(carrier_conv) || !used_once(carrier_wide) {
        return None;
    }
    // The identity is taken at the width its terms are computed at, and the
    // narrowing reads the tangle's width out of it.
    one_width(ssa, &[pair.0, pair.1, l_wide, r_wide])?;
    one_width(ssa, &[dest, l_conv, r_conv])?;
    let (a, b) = pair;
    let wide = match operation {
        Combined::Add => SsaOp::Add {
            dest: carrier_wide,
            left: a,
            right: b,
            flags: None,
        },
        Combined::Xor => SsaOp::Xor {
            dest: carrier_wide,
            left: a,
            right: b,
            flags: None,
        },
        Combined::And => SsaOp::And {
            dest: carrier_wide,
            left: a,
            right: b,
            flags: None,
        },
        Combined::Or => SsaOp::Or {
            dest: carrier_wide,
            left: a,
            right: b,
            flags: None,
        },
    };
    Some(MbaFold {
        op: SsaOp::Copy {
            dest,
            src: carrier_conv,
        },
        description,
        also: vec![(carrier_wide, wide)],
        dest_type: None,
    })
}

/// The operation the tangle computes, or `None` when `op` spells no identity,
/// the values involved do not share one known width, or `op`'s flags output is
/// read.
#[must_use]
pub fn fold<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    uses: &BTreeMap<SsaVarId, usize>,
    ptr_size: PointerSize,
) -> Option<MbaFold<T>> {
    // The replacement's flags are not the tangle's (`a + ~b + 1` sets the
    // carry where `a - b` clears it), so a reader of them would see a
    // different value. The original output is carried across only as a
    // definition nobody reads.
    if op
        .flags_dest()
        .is_some_and(|flags| uses.get(&flags).copied().unwrap_or(0) != 0)
    {
        return None;
    }
    if let Some(folded) = narrowed_tangle(ssa, op, constants, uses) {
        return Some(folded);
    }
    let dest = op.dest()?;
    // The operands the fold hands back must be read at the width the
    // destination is written at, or the result is a different ring's answer.
    let same_width = |a: SsaVarId, b: SsaVarId| one_width(ssa, &[dest, a, b]).is_some();
    match op {
        // a + b  ==  (a ^ b) + 2*(a & b)  ==  (a | b) + (a & b)
        SsaOp::Add {
            left, right, flags, ..
        } => {
            // a - b  ==  a + ~b + 1, the two's-complement spelling. The lift
            // produces it nested - `add(add(a, not(b)), 1)` - so the outer
            // `Add` carries the `+ 1` and its left operand carries the sum: an
            // x86 lift gives `v12 = add v6, v11` over `v9 = not v8`, then
            // `v15 = add v12, 1`. Checked before the carry identities because
            // its right operand is a bare constant, which `term` cannot
            // resolve.
            if constants.get(right).is_some_and(ConstValue::is_one)
                && let Some(SsaOp::Add {
                    left: inner_left,
                    right: inner_right,
                    ..
                }) = ssa.get_definition(peel(ssa, *left))
            {
                let complement = match (
                    term(ssa, *inner_left, constants),
                    term(ssa, *inner_right, constants),
                ) {
                    (Some(Term::Not(b)), _) => Some((*inner_right, b)),
                    (_, Some(Term::Not(b))) => Some((*inner_left, b)),
                    _ => None,
                };
                if let Some((a, b)) = complement
                    && same_width(a, b)
                {
                    return Some(MbaFold {
                        op: SsaOp::Sub {
                            dest,
                            left: a,
                            right: b,
                            flags: *flags,
                        },
                        description: "mba → sub",
                        also: Vec::new(),
                        dest_type: None,
                    });
                }
            }
            let (l, r) = (term(ssa, *left, constants)?, term(ssa, *right, constants)?);
            let pair = match (&l, &r) {
                (Term::Xor(a, b), Term::TwiceAnd(c, d))
                | (Term::TwiceAnd(c, d), Term::Xor(a, b))
                    if same((*a, *b), (*c, *d)) =>
                {
                    Some((*a, *b))
                }
                (Term::Or(a, b), Term::And(c, d)) | (Term::And(c, d), Term::Or(a, b))
                    if same((*a, *b), (*c, *d)) =>
                {
                    Some((*a, *b))
                }
                _ => None,
            }?;
            same_width(pair.0, pair.1).then_some(MbaFold {
                op: SsaOp::Add {
                    dest,
                    left: pair.0,
                    right: pair.1,
                    flags: *flags,
                },
                description: "mba → add",
                also: Vec::new(),
                dest_type: None,
            })
        }
        // The same `a + b == (a ^ b) + 2*(a & b)` identity, but with the outer
        // sum lowered to a scaled address computation: gcc emits `2*(a & b)` as
        // an `lea [xor + and*2]`, which lifts to a `PtrAdd` (`base +
        // index*stride`) rather than an `Add`, so the `Add` arm never sees it
        // and the tangle survives - the dominant x86 shape. Matched only as
        // `base + 2*index` with no displacement, so it is unambiguously the
        // two-term doubled-carry sum and not a genuine array index.
        //
        // The address sum is computed at the pointer width, so the operands
        // must be pointer-wide too: over 32-bit operands with 64-bit pointers
        // the `lea` keeps the carry out of bit 31 that a 32-bit `Add` drops.
        //
        // The sum is integer addition, which the address form only spelled, so
        // the destination the lift typed as a pointer is retyped as the terms'
        // integer (see [`MbaFold::dest_type`]).
        SsaOp::PtrAdd {
            base,
            index: Some(index),
            stride: 2,
            offset: 0,
            ..
        } => {
            if let (Some(Term::Xor(a, b)), Some(Term::And(c, d))) =
                (term(ssa, *base, constants), term(ssa, *index, constants))
                && one_width(ssa, &[a, b, *base, *index]) == Some(ptr_size.bits())
                && same((a, b), (c, d))
            {
                return Some(MbaFold {
                    op: SsaOp::Add {
                        dest,
                        left: a,
                        right: b,
                        flags: None,
                    },
                    description: "mba (lea) → add",
                    also: Vec::new(),
                    dest_type: Some(ssa.variable(a)?.var_type().clone()),
                });
            }
            widened_carry_sum(ssa, uses, dest, *base, *index)
        }
        // a ^ b  ==  (a | b) - (a & b), and  a & b  ==  (a + b) - (a | b).
        SsaOp::Sub {
            left, right, flags, ..
        } => {
            let (l, r) = (term(ssa, *left, constants)?, term(ssa, *right, constants)?);
            match (&l, &r) {
                (Term::Or(a, b), Term::And(c, d))
                    if same((*a, *b), (*c, *d)) && same_width(*a, *b) =>
                {
                    Some(MbaFold {
                        op: SsaOp::Xor {
                            dest,
                            left: *a,
                            right: *b,
                            flags: *flags,
                        },
                        description: "mba → xor",
                        also: Vec::new(),
                        dest_type: None,
                    })
                }
                (Term::Add(a, b), Term::Or(c, d))
                    if same((*a, *b), (*c, *d)) && same_width(*a, *b) =>
                {
                    Some(MbaFold {
                        op: SsaOp::And {
                            dest,
                            left: *a,
                            right: *b,
                            flags: *flags,
                        },
                        description: "mba → and",
                        also: Vec::new(),
                        dest_type: None,
                    })
                }
                _ => None,
            }
        }
        // a | b  ==  (a & b) | (a ^ b): the bits set in either are exactly the
        // bits set in both plus the bits set in one.
        SsaOp::Or {
            left, right, flags, ..
        } => {
            let (l, r) = (term(ssa, *left, constants)?, term(ssa, *right, constants)?);
            match (&l, &r) {
                (Term::And(a, b), Term::Xor(c, d)) | (Term::Xor(c, d), Term::And(a, b))
                    if same((*a, *b), (*c, *d)) && same_width(*a, *b) =>
                {
                    Some(MbaFold {
                        op: SsaOp::Or {
                            dest,
                            left: *a,
                            right: *b,
                            flags: *flags,
                        },
                        description: "mba → or",
                        also: Vec::new(),
                        dest_type: None,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        ir::SsaFunctionBuilder,
        testing::{MockTarget, MockType, mock_i32, mock_i64, mock_ptr},
    };

    /// Builds `(a ^ b) + ((a & b) + (a & b))` - the doubling written as a
    /// self-addition, which is how x86 emits `2 * (a & b)` at `-O0`.
    fn self_add_carry_fixture() -> (
        SsaFunction<MockTarget>,
        SsaOp<MockTarget>,
        BTreeMap<SsaVarId, ConstValue<MockTarget>>,
    ) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
        let mut add_op = None;
        builder
            .in_block(0, |block| {
                let a = block.const_i32(mock_i32(), 5)?;
                let b = block.const_i32(mock_i32(), 3)?;
                let xor = block.xor(mock_i32(), a, b)?;
                let and = block.and(mock_i32(), a, b)?;
                let twice = block.add(mock_i32(), and, and)?;
                let sum = block.add(mock_i32(), xor, twice)?;
                add_op = Some(SsaOp::Add {
                    dest: sum,
                    left: xor,
                    right: twice,
                    flags: None,
                });
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        (ssa, add_op.expect("add op"), constants)
    }

    /// The doubling written as `x + x` is twice the carries just as `x << 1`
    /// is. x86 at `-O0` emits it this way, so `term` has to read a self-sum of
    /// an `And` as the doubled carries rather than as a plain `Term::Add`.
    #[test]
    fn a_self_added_carry_sum_folds_to_add() {
        let (ssa, add_op, constants) = self_add_carry_fixture();
        let fold = fold(
            &ssa,
            &add_op,
            &constants,
            &ssa.count_uses(),
            PointerSize::Bit64,
        )
        .expect("the identity folds");
        assert_eq!(fold.description, "mba → add");
        match fold.op {
            SsaOp::Add { .. } => {}
            other => panic!("expected an Add, got {other:?}"),
        }
    }

    /// **Terms of no stated type are the bits the identity reads.** An `-O0`
    /// argument a move copies states no type; the identity's operations state
    /// its width. The control types the terms at another width, a different
    /// ring's answer.
    #[test]
    fn untyped_terms_fold_at_their_identitys_width() {
        let folded = |ty: MockType| {
            let (mut ssa, add_op, constants) = self_add_carry_fixture();
            let terms: Vec<SsaVarId> = ssa
                .iter_instructions()
                .filter_map(|(_, _, instr)| match instr.op() {
                    SsaOp::Const { dest, .. } => Some(*dest),
                    _ => None,
                })
                .collect();
            for term in terms {
                if let Some(variable) = ssa.variable_mut(term) {
                    variable.set_type(ty);
                }
            }
            fold(
                &ssa,
                &add_op,
                &constants,
                &ssa.count_uses(),
                PointerSize::Bit64,
            )
            .map(|fold| fold.description)
        };
        assert_eq!(folded(MockType::Unknown), Some("mba → add"));
        assert_eq!(folded(MockType::I64), None, "the control");
    }

    /// The control: `x + y` over two *different* values is not a doubling, so
    /// the sum-of-carries identity must not fire.
    #[test]
    fn a_sum_of_two_different_ands_does_not_fold() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
        let mut add_op = None;
        builder
            .in_block(0, |block| {
                let a = block.const_i32(mock_i32(), 5)?;
                let b = block.const_i32(mock_i32(), 3)?;
                let c = block.const_i32(mock_i32(), 7)?;
                let xor = block.xor(mock_i32(), a, b)?;
                let and1 = block.and(mock_i32(), a, b)?;
                let and2 = block.and(mock_i32(), a, c)?;
                let twice = block.add(mock_i32(), and1, and2)?;
                let sum = block.add(mock_i32(), xor, twice)?;
                add_op = Some(SsaOp::Add {
                    dest: sum,
                    left: xor,
                    right: twice,
                    flags: None,
                });
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        assert!(
            fold(
                &ssa,
                &add_op.expect("add op"),
                &constants,
                &ssa.count_uses(),
                PointerSize::Bit64,
            )
            .is_none(),
            "`(a & b) + (a & c)` is not a doubling and must not fold"
        );
    }

    /// `a + ~b + 1` is `a - b`, the identity the `Term::Not` sub-term exists
    /// for.
    #[test]
    fn a_complement_sum_folds_to_sub() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
        let mut add_op = None;
        builder
            .in_block(0, |block| {
                let a = block.const_i32(mock_i32(), 5)?;
                let b = block.const_i32(mock_i32(), 3)?;
                let one = block.const_i32(mock_i32(), 1)?;
                let complement = block.not(mock_i32(), b)?;
                let inner = block.add(mock_i32(), a, complement)?;
                let sum = block.add(mock_i32(), inner, one)?;
                add_op = Some(SsaOp::Add {
                    dest: sum,
                    left: inner,
                    right: one,
                    flags: None,
                });
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        let fold = fold(
            &ssa,
            &add_op.expect("add op"),
            &constants,
            &ssa.count_uses(),
            PointerSize::Bit64,
        )
        .expect("the identity folds");
        assert_eq!(fold.description, "mba → sub");
        match fold.op {
            SsaOp::Sub { .. } => {}
            other => panic!("expected a Sub, got {other:?}"),
        }
    }

    /// The control: `a + ~b + 2` is not `a - b`, so the displacement must be
    /// exactly one for the complement identity to hold.
    #[test]
    fn a_complement_sum_with_the_wrong_addend_does_not_fold() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
        let mut add_op = None;
        builder
            .in_block(0, |block| {
                let a = block.const_i32(mock_i32(), 5)?;
                let b = block.const_i32(mock_i32(), 3)?;
                let two = block.const_i32(mock_i32(), 2)?;
                let complement = block.not(mock_i32(), b)?;
                let inner = block.add(mock_i32(), a, complement)?;
                let sum = block.add(mock_i32(), inner, two)?;
                add_op = Some(SsaOp::Add {
                    dest: sum,
                    left: inner,
                    right: two,
                    flags: None,
                });
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        assert!(
            fold(
                &ssa,
                &add_op.expect("add op"),
                &constants,
                &ssa.count_uses(),
                PointerSize::Bit64,
            )
            .is_none(),
            "`a + ~b + 2` is not `a - b` and must not fold"
        );
    }

    /// Builds `(a ^ b) + ((a & b) << 1)` in one block, returns the function and
    /// the `Add` op that tops the identity, plus the constant map.
    fn xoradd_add_fixture() -> (
        SsaFunction<MockTarget>,
        SsaOp<MockTarget>,
        BTreeMap<SsaVarId, ConstValue<MockTarget>>,
    ) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
        let mut add_op = None;
        builder
            .in_block(0, |block| {
                let a = block.const_i32(mock_i32(), 5)?;
                let b = block.const_i32(mock_i32(), 3)?;
                let one = block.const_i32(mock_i32(), 1)?;
                let xor = block.xor(mock_i32(), a, b)?;
                let and = block.and(mock_i32(), a, b)?;
                let twice = block.shl(mock_i32(), and, one)?;
                let sum = block.add(mock_i32(), xor, twice)?;
                add_op = Some(SsaOp::Add {
                    dest: sum,
                    left: xor,
                    right: twice,
                    flags: None,
                });
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        // `one` is the only constant `fold` consults (the shift amount).
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        (ssa, add_op.expect("add op"), constants)
    }

    /// Builds `lea [(a ^ b) + (a & b)*2]` over `ty` operands, returning the
    /// function and the `PtrAdd` that tops the identity.
    fn lea_fixture(ty: MockType) -> (SsaFunction<MockTarget>, SsaOp<MockTarget>) {
        let def = if ty == MockType::I64 {
            mock_i64
        } else {
            mock_i32
        };
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut ptradd_op = None;
        builder
            .in_block(0, |block| {
                let a = block.load_arg(def(), 0)?;
                let b = block.load_arg(def(), 1)?;
                let xor = block.xor(def(), a, b)?;
                let and = block.and(def(), a, b)?;
                let op = |dest| SsaOp::PtrAdd {
                    dest,
                    base: xor,
                    index: Some(and),
                    stride: 2,
                    offset: 0,
                    result_type: MockType::Ptr,
                };
                let sum = block.emit_def(mock_ptr(), op)?;
                ptradd_op = Some(op(sum));
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        (ssa, ptradd_op.expect("op"))
    }

    /// The lea-lowered addition identity - `(a ^ b) + 2*(a & b)` where the outer
    /// sum is a scaled `PtrAdd` (`base=xor`, `index=and`, `stride=2`) rather than
    /// an `Add` - folds to `a + b` when the operands are as wide as a pointer.
    ///
    /// The lift typed the address's destination as a pointer. The fold states
    /// the integer type the sum really has, so the destination is not left a
    /// pointer defined by integer arithmetic.
    #[test]
    fn a_lea_lowered_carry_sum_folds_to_add() {
        for (ty, ptr_size) in [
            (MockType::I64, PointerSize::Bit64),
            (MockType::I32, PointerSize::Bit32),
        ] {
            let (ssa, op) = lea_fixture(ty);
            // No constants are consulted: the `*2` is the `PtrAdd` stride, not
            // a shift by a constant one.
            let fold = fold(&ssa, &op, &BTreeMap::new(), &ssa.count_uses(), ptr_size)
                .expect("the lea-lowered identity folds");
            assert_eq!(fold.description, "mba (lea) → add");
            let SsaOp::Add { left, right, .. } = fold.op else {
                panic!("expected an Add, got {:?}", fold.op);
            };
            assert_eq!(
                fold.dest_type,
                Some(ty),
                "the sum is the terms' integer, not the address's pointer"
            );
            let SsaOp::PtrAdd {
                base: xor,
                index: Some(and),
                ..
            } = op
            else {
                panic!("the fixture tops with a PtrAdd");
            };
            let terms = |var| match ssa.get_definition(var) {
                Some(SsaOp::Xor { left, right, .. } | SsaOp::And { left, right, .. }) => {
                    (*left, *right)
                }
                other => panic!("expected a boolean term, got {other:?}"),
            };
            assert_eq!((left, right), terms(xor));
            assert_eq!(terms(xor), terms(and), "both terms read the same a, b");
        }
    }

    /// Builds the widened form gcc emits at `-O2`: the boolean operations at
    /// 32 bits, each result zero-extended to 64, and the scaled address over
    /// the extensions.
    fn widened_lea_fixture() -> (SsaFunction<MockTarget>, SsaOp<MockTarget>) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let mut ptradd_op = None;
        builder
            .in_block(0, |block| {
                let a = block.load_arg(mock_i32(), 0)?;
                let b = block.load_arg(mock_i32(), 1)?;
                let xor = block.xor(mock_i32(), a, b)?;
                let and = block.and(mock_i32(), a, b)?;
                let widen = |operand| {
                    move |dest| SsaOp::IntConv {
                        dest,
                        operand,
                        target: MockType::I64,
                        overflow_check: false,
                        unsigned: true,
                    }
                };
                let wide_xor = block.emit_def(mock_i64(), widen(xor))?;
                let wide_and = block.emit_def(mock_i64(), widen(and))?;
                let op = |dest| SsaOp::PtrAdd {
                    dest,
                    base: wide_xor,
                    index: Some(wide_and),
                    stride: 2,
                    offset: 0,
                    result_type: MockType::Ptr,
                };
                let sum = block.emit_def(mock_ptr(), op)?;
                ptradd_op = Some(op(sum));
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        (ssa, ptradd_op.expect("op"))
    }

    /// The widened lea-lowered identity folds without dropping the carry.
    ///
    /// gcc takes the boolean operations at the operands' width and zero-extends
    /// each result into the address, so the terms are narrower than the sum.
    /// Folding to a 32-bit `Add` would be wrong - the sibling test below says
    /// why - so the fold stays at the address's width and recovers the terms by
    /// redirecting the two conversions, which is what `also` carries. Nothing
    /// is created: the conversions already produce values of the right type.
    #[test]
    fn a_widened_lea_carry_sum_folds_at_the_address_width() {
        let (ssa, op) = widened_lea_fixture();
        let fold = fold(
            &ssa,
            &op,
            &BTreeMap::new(),
            &ssa.count_uses(),
            PointerSize::Bit64,
        )
        .expect("the widened identity folds");
        assert_eq!(fold.description, "mba (widened lea) → add");
        assert!(
            matches!(fold.op, SsaOp::Add { flags: None, .. }),
            "the scale is removed and the terms are added"
        );
        assert_eq!(
            fold.dest_type,
            Some(MockType::I64),
            "the sum stays at the address's width, as an integer"
        );
        assert_eq!(fold.also.len(), 2, "both conversions are redirected");
        for (_, op) in &fold.also {
            assert!(
                matches!(op, SsaOp::IntConv { unsigned: true, .. }),
                "each redirect widens one term, still zero-extending"
            );
        }
    }

    /// A conversion read anywhere else is not redirected.
    ///
    /// The redirect changes what a definition holds, which is only invisible
    /// while this address is its one reader. A second reader would see the
    /// term where it expected the boolean result, so the fold declines.
    #[test]
    fn a_widened_lea_whose_conversion_has_another_reader_does_not_fold() {
        let (ssa, op) = widened_lea_fixture();
        let mut uses = ssa.count_uses();
        let SsaOp::PtrAdd { base, .. } = &op else {
            panic!("the fixture tops with a PtrAdd");
        };
        uses.insert(*base, 2);
        assert!(
            fold(&ssa, &op, &BTreeMap::new(), &uses, PointerSize::Bit64).is_none(),
            "a conversion something else reads must keep what it holds"
        );
    }

    /// A `lea` sums at the pointer width. Over 32-bit operands with 64-bit
    /// pointers it keeps the carry out of bit 31 (`a = b = 0x8000_0000` sums to
    /// `0x1_0000_0000`), which a 32-bit `Add` would drop, so it is not `a + b`.
    #[test]
    fn a_lea_over_operands_narrower_than_a_pointer_does_not_fold() {
        let (ssa, op) = lea_fixture(MockType::I32);
        assert!(
            fold(
                &ssa,
                &op,
                &BTreeMap::new(),
                &ssa.count_uses(),
                PointerSize::Bit64
            )
            .is_none(),
            "a 64-bit address sum of 32-bit operands is not their 32-bit sum"
        );
    }

    /// The classic `(a ^ b) + 2*(a & b)` addition identity folds to `a + b`.
    #[test]
    fn xor_carry_sum_folds_to_add() {
        let (ssa, add_op, constants) = xoradd_add_fixture();
        let fold = fold(
            &ssa,
            &add_op,
            &constants,
            &ssa.count_uses(),
            PointerSize::Bit64,
        )
        .expect("the identity folds");
        assert_eq!(fold.description, "mba → add");
        match fold.op {
            SsaOp::Add { .. } => {}
            other => panic!("expected an Add, got {other:?}"),
        }
    }

    /// The replacement sets flags differently from the tangle, so a flags
    /// output that is read refuses the fold and one that is not read allows it.
    #[test]
    fn a_read_flags_output_refuses_the_fold() {
        let (ssa, add_op, constants) = xoradd_add_fixture();
        let SsaOp::Add {
            dest, left, right, ..
        } = add_op
        else {
            panic!("the fixture tops the identity with an Add");
        };
        let flags = SsaVarId::from_index(ssa.variable_count());
        let with_flags = SsaOp::Add {
            dest,
            left,
            right,
            flags: Some(flags),
        };
        let read = BTreeMap::from([(flags, 1)]);
        assert!(
            fold(&ssa, &with_flags, &constants, &read, PointerSize::Bit64).is_none(),
            "a reader of the flags would see the replacement's flags"
        );
        assert!(
            fold(
                &ssa,
                &with_flags,
                &constants,
                &BTreeMap::new(),
                PointerSize::Bit64
            )
            .is_some(),
            "unread flags do not block the fold"
        );
    }

    /// A fold requires the two sub-terms to name the *same* values: an `Add`
    /// whose `xor` and `and` read different operands is not the identity and
    /// must be left alone.
    #[test]
    fn a_mismatched_pair_does_not_fold() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
        let mut add_op = None;
        builder
            .in_block(0, |block| {
                let a = block.const_i32(mock_i32(), 5)?;
                let b = block.const_i32(mock_i32(), 3)?;
                let c = block.const_i32(mock_i32(), 7)?;
                let one = block.const_i32(mock_i32(), 1)?;
                let xor = block.xor(mock_i32(), a, b)?;
                let and = block.and(mock_i32(), a, c)?; // different second operand
                let twice = block.shl(mock_i32(), and, one)?;
                let sum = block.add(mock_i32(), xor, twice)?;
                add_op = Some(SsaOp::Add {
                    dest: sum,
                    left: xor,
                    right: twice,
                    flags: None,
                });
                block.ret(Some(sum))?;
                Ok(())
            })
            .expect("build");
        let ssa = builder.finish().expect("finish");
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        assert!(
            fold(
                &ssa,
                &add_op.unwrap(),
                &constants,
                &ssa.count_uses(),
                PointerSize::Bit64,
            )
            .is_none(),
            "a mismatched pair is not the identity and must not fold"
        );
    }
}
