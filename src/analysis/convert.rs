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

use crate::{
    ir::{function::SsaFunction, ops::SsaOp},
    target::Target,
};

/// Collapses an integer conversion whose operand is another integer
/// conversion, where the intermediate one cannot be observed.
///
/// A lift emits one conversion per width change, so a value that is narrowed,
/// widened and narrowed again arrives as a chain — and the chain is what a
/// reader sees as `(uint16_t)(uint64_t)(uint32_t)x`. Measured over the
/// committed fixtures, **40.8%** of `x86_64` integer conversions take another
/// conversion as their operand, and 28.1% on `arm_64`.
///
/// # When the intermediate is unobservable
///
/// A conversion to width `w` yields the low `w` bits of its operand, extended
/// above `w` when the operand is narrower — by sign or by zero, according to
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
/// extension either way — the intermediate's before collapsing, the surviving
/// conversion's after — so the two are interchangeable exactly when they agree
/// on sign or zero. That is the second arm, and it is what lets
/// `(uint32_t)(uint64_t)x` collapse where `(uint32_t)(int64_t)x` may not.
///
/// A conversion that checks overflow is never removed: the check is the point
/// of it.
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
/// - [`SsaOp::Copy`] when `op` converts to the type its operand already has,
///   so the conversion converts nothing;
/// - [`SsaOp::IntConv`] reading whichever variable the walk reached, when one
///   or more links fell away.
///
/// `None` also covers every refusal — an overflow check, a width the rule
/// cannot clear, a link whose definition is not recoverable from its def site,
/// and a def-use chain long enough to be cyclic.
///
/// # Correctness
///
/// Each link is resolved with
/// [`SsaFunction::recorded_definition`](SsaFunction::recorded_definition), the
/// guarded lookup, so a def site an edit left stale stops the walk instead of
/// importing an unrelated conversion's widths — a missed optimization rather
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
    // an already-narrow one. So the identity appears here and nowhere earlier —
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
        let Some(source_width) = ssa
            .variable(*source)
            .and_then(|variable| T::bit_width(variable.var_type()))
        else {
            break;
        };

        // The result must not read past what this link kept.
        if outer_width > link_width {
            break;
        }
        // Below the source width, every bit the result keeps came from the
        // source directly. At or above it, the bits between the source and the
        // result are this link's extension, and the collapsed form supplies its
        // own — so the two must agree on which.
        if outer_width > source_width && *link_unsigned != *unsigned {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            ops::SsaOp,
            variable::{DefSite, SsaVarId},
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
    /// change nothing" — so it stops at the first link that does, and leaves
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
    /// conversion is never the one removed — not even when its widths would
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
    /// widths rewrites the outer conversion to an operand that never fed it —
    /// silent wrong code, not a missed optimization — so the guarded lookup
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
    /// Without the bound this test does not fail — it hangs.
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

    // ---------------------------------------------------------------------
    // The width oracle
    // ---------------------------------------------------------------------

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
}
