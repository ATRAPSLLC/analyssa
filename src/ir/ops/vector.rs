//! Vector and SIMD descriptors - lane kinds, shuffle and permute payloads,
//! SVE / SME / AMX-tile operands, and the vector operation kind taxonomies.
//!
//! Split from [`crate::ir::ops::kinds`] because the vector surface is large enough to
//! read on its own and is consulted as a unit when adding lane semantics.

use std::{cmp::Ordering, fmt};

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    ir::{
        float::{FloatBinary, FloatBits, FloatFormat, FloatRounding, RaisingArithmetic},
        integer::BitWidth,
        lanes::VectorConst,
        ops::native::FlagsMask,
        variable::SsaVarId,
    },
    target::VectorShuffleMask,
};

mod emit;

mod crypto;

pub use self::crypto::VectorCryptoKind;

/// Which lanes a [`SsaOp::VectorShuffle`] selects from its inputs, laid end
/// to end, the left lowest.
///
/// [`SsaOp::VectorShuffle`]: crate::ir::ops::def::SsaOp::VectorShuffle
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorShuffleSelector {
    /// A lane selector for each result lane, fixed by the instruction
    /// (`pshufd`, `zip1`, `vslideup.vi`).
    Mask(VectorShuffleMask),
    /// A vector of integer lanes, each the number of the input lane its
    /// result lane takes (`vpermd`, `pshufb`, `tbl`, `vrgather.vv`): the
    /// result has a lane per index lane, as wide as the inputs' lanes. Every
    /// index lies within the inputs; what an instruction makes of one past
    /// them (zero, the destination's lane, a lane reduced modulo the count)
    /// its lift states before and after the shuffle.
    Index(SsaVarId),
}

impl VectorShuffleSelector {
    /// Returns the fixed mask, or `None` for lanes an index vector selects.
    #[must_use]
    pub const fn mask(&self) -> Option<&VectorShuffleMask> {
        match self {
            Self::Mask(mask) => Some(mask),
            Self::Index(_) => None,
        }
    }

    /// Returns the index vector, or `None` for a fixed mask.
    #[must_use]
    pub const fn index(&self) -> Option<SsaVarId> {
        match self {
            Self::Mask(_) => None,
            Self::Index(index) => Some(*index),
        }
    }
}

/// How a float dot product adds its products ([`Contraction::FloatDot`]):
/// each lane group's products, exact in the accumulator's format, added to
/// the accumulator lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FloatDotRule {
    /// x86 `vdpbf16ps`: each product added to the accumulator in turn, the odd
    /// pair's first, each sum rounded to nearest even, with denormal inputs
    /// and outputs flushed to zero whatever MXCSR states.
    Sequential,
    /// Arm `bfdot` (A64 and SVE) and A32 `vdot.bf16`: the two products summed,
    /// and the sum added to the accumulator (the pseudocode's `BFDotAdd`),
    /// each addition rounded as the floating-point control register's `EBF`
    /// bit selects: clear, to odd with denormals flushed and FPCR's own
    /// controls ignored; set (FEAT_EBF16), as FPCR's rounding mode and
    /// denormal controls state.
    Fpcr,
    /// Arm `fdot` of half-precision pairs (SVE2.1, SME2): the products
    /// summed with one rounding and that sum added to the accumulator with
    /// another (the pseudocode's `FPDotAdd`), rounding as FPCR states, and a
    /// NaN chosen among all the inputs in `FPProcessNaNs4`'s order.
    ///
    /// The forms writing ZA compute the same way with default NaNs and no
    /// exceptions (`FPDotAdd_ZA`); that is their floating-point environment
    /// rather than another rule, and an operation states its environment
    /// where every other operation does.
    DotAdd,
    /// Arm's FP8 arithmetic (`fdot`, `fvdot`, `fmlalb`, `fmlall` of bytes):
    /// the first factor decoded as FPMR's `F8S1` names and the second as its
    /// `F8S2` names, the products and their sum exact, scaled by
    /// `2^-LSCALE` (by its low four bits into binary16, the pseudocode's
    /// `FP8DotAddFP`), and added to the accumulator with one rounding to
    /// nearest even, an overflow saturating to the largest finite value
    /// where FPMR's `OSM` is set. Whatever FPCR states, no input or result
    /// is flushed, a NaN result is the default NaN, and no exception is
    /// raised. FPMR is the contraction's last input.
    Fp8,
}

/// What a contraction computes ([`SsaOp::Contract`]): products of lanes
/// summed into lanes. Its inputs are the two factors, then the accumulator
/// where it adds to one, as an instruction that names its addend apart
/// from its destination orders them (`vpmadcswd`); one whose destination is
/// its accumulator names the accumulator first instead. A contraction of
/// FP8 factors, dot, matrix or outer product, reads last the value that
/// names their format at run time (FPMR).
///
/// [`SsaOp::Contract`]: crate::ir::ops::def::SsaOp::Contract
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Contraction {
    /// An integer dot product of groups of lanes ([`DotShape`]): x86
    /// `pmaddwd`, `pmaddubsw` and the `vpdp*` family, Arm `sdot`, `udot` and
    /// `usdot`.
    Dot(DotShape),
    /// A float dot product of lane groups into wider lanes, added to the
    /// accumulator by `rule`: bfloat16 pairs into binary32 (`bfdot`,
    /// `vdpbf16ps`), half-precision pairs into binary32 (Arm's `fdot`) and
    /// FP8 groups into either (`fdot` of bytes), or of fewer products per
    /// lane where the instruction multiplies fewer of the factors its lane
    /// covers (FP8 `fmlalb`, `fmlallbb` and `fvdotb`, over the bytes they
    /// select). One operation states every float dot, so nothing can state
    /// one dot two ways.
    FloatDot {
        /// The lanes multiplied and the ones they are summed into.
        elements: FloatElements,
        /// How the products and the accumulation round.
        rule: FloatDotRule,
    },
    /// The low or high 52 bits of the product of each pair of unsigned
    /// 52-bit lanes added to the 64-bit accumulator lane (x86
    /// `vpmadd52luq`, `vpmadd52huq`).
    MultiplyAdd52 {
        /// Whether the high 52 bits are added.
        high: bool,
    },
    /// A float dot product of the lanes `control`'s high nibble selects,
    /// written to the lanes its low nibble selects and zero elsewhere (x86
    /// `dpps`, `dppd`), of lanes `element_bits` wide.
    MaskedDot {
        /// The lane-participation and result-broadcast control.
        control: u8,
        /// The width of the lanes.
        element_bits: u16,
    },
    /// A matrix product added to the accumulator matrix ([`MatrixShape`]).
    Matrix(MatrixShape),
    /// An outer product added to or subtracted from a tile, under two
    /// governing predicates ([`OuterShape`]). Its inputs are the two
    /// predicates, the two factors, then the tile.
    Outer(OuterShape),
    /// A complex product or rotation ([`ComplexShape`]). A rotation's inputs
    /// are the source added to, then the source turned.
    Complex(ComplexShape),
}

impl Contraction {
    /// Returns the contraction's IR name: `dot`, `dot.bf16`, `dot.bf16.fpcr`,
    /// `dot.f16`, `dot.fp8`, `dot.f`, `muladd52.lo`, `muladd52.hi`,
    /// `dot.masked`, `matrix`,
    /// `matrix.complex.re`, `matrix.complex.im`, `outer.add`, `outer.sub`,
    /// `complex.mul`, `complex.mul.conj`, `complex.muladd`,
    /// `complex.muladd.conj`, `complex.add.rot90` and `complex.add.rot270`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dot(_) => "dot",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Format(FloatFormat::BFloat16),
                        ..
                    },
                rule: FloatDotRule::Sequential,
            } => "dot.bf16",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Format(FloatFormat::BFloat16),
                        ..
                    },
                rule: FloatDotRule::Fpcr,
            } => "dot.bf16.fpcr",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Format(FloatFormat::Half),
                        ..
                    },
                rule: FloatDotRule::DotAdd,
            } => "dot.f16",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Fp8,
                        ..
                    },
                rule: FloatDotRule::Fp8,
            } => "dot.fp8",
            Self::FloatDot { .. } => "dot.f",
            Self::MultiplyAdd52 { high: false } => "muladd52.lo",
            Self::MultiplyAdd52 { high: true } => "muladd52.hi",
            Self::MaskedDot { .. } => "dot.masked",
            Self::Matrix(shape) => match shape.complex {
                None => "matrix",
                Some(ComplexPart::Real) => "matrix.complex.re",
                Some(ComplexPart::Imaginary) => "matrix.complex.im",
            },
            Self::Outer(OuterShape {
                subtract: true,
                product: OuterProduct::Multiply,
                ..
            }) => "outer.sub",
            Self::Outer(OuterShape {
                product: OuterProduct::Multiply,
                ..
            }) => "outer.add",
            Self::Outer(OuterShape {
                subtract: true,
                product: OuterProduct::MatchingBits,
                ..
            }) => "outer.sub.xnor_count",
            Self::Outer(OuterShape {
                product: OuterProduct::MatchingBits,
                ..
            }) => "outer.add.xnor_count",
            Self::Complex(shape) => match shape.product {
                ComplexProduct::Multiply {
                    conjugate: false,
                    accumulates: false,
                } => "complex.mul",
                ComplexProduct::Multiply {
                    conjugate: true,
                    accumulates: false,
                } => "complex.mul.conj",
                ComplexProduct::Multiply {
                    conjugate: false,
                    accumulates: true,
                } => "complex.muladd",
                ComplexProduct::Multiply {
                    conjugate: true,
                    accumulates: true,
                } => "complex.muladd.conj",
                ComplexProduct::Rotate(ComplexRotation::Quarter) => "complex.add.rot90",
                ComplexProduct::Rotate(ComplexRotation::ThreeQuarters) => "complex.add.rot270",
            },
        }
    }

    /// Returns the stable fingerprint key of the contraction.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Dot(_) => "contract.dot",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Format(FloatFormat::BFloat16),
                        ..
                    },
                rule: FloatDotRule::Sequential,
            } => "contract.bfdot",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Format(FloatFormat::BFloat16),
                        ..
                    },
                rule: FloatDotRule::Fpcr,
            } => "contract.bfdot.fpcr",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Format(FloatFormat::Half),
                        ..
                    },
                rule: FloatDotRule::DotAdd,
            } => "contract.fdot.f16",
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Fp8,
                        ..
                    },
                rule: FloatDotRule::Fp8,
            } => "contract.fdot.fp8",
            Self::FloatDot { .. } => "contract.fdot",
            Self::MultiplyAdd52 { high: false } => "contract.madd52lo",
            Self::MultiplyAdd52 { high: true } => "contract.madd52hi",
            Self::MaskedDot { .. } => "contract.dpps",
            Self::Matrix(shape) => match shape.complex {
                None => "contract.matrix",
                Some(ComplexPart::Real) => "contract.matrix.cre",
                Some(ComplexPart::Imaginary) => "contract.matrix.cim",
            },
            Self::Outer(OuterShape {
                subtract: true,
                product: OuterProduct::Multiply,
                ..
            }) => "contract.outer.sub",
            Self::Outer(OuterShape {
                product: OuterProduct::Multiply,
                ..
            }) => "contract.outer",
            Self::Outer(OuterShape {
                subtract: true,
                product: OuterProduct::MatchingBits,
                ..
            }) => "contract.outer.sub.xnor",
            Self::Outer(OuterShape {
                product: OuterProduct::MatchingBits,
                ..
            }) => "contract.outer.xnor",
            Self::Complex(shape) => match shape.product {
                ComplexProduct::Multiply {
                    conjugate: false,
                    accumulates: false,
                } => "contract.cmul",
                ComplexProduct::Multiply {
                    conjugate: true,
                    accumulates: false,
                } => "contract.cmul.conj",
                ComplexProduct::Multiply {
                    conjugate: false,
                    accumulates: true,
                } => "contract.cmadd",
                ComplexProduct::Multiply {
                    conjugate: true,
                    accumulates: true,
                } => "contract.cmadd.conj",
                ComplexProduct::Rotate(ComplexRotation::Quarter) => "contract.cadd.rot90",
                ComplexProduct::Rotate(ComplexRotation::ThreeQuarters) => "contract.cadd.rot270",
            },
        }
    }

    /// Returns the inputs: the two factors (an outer product's predicates
    /// before them), then the accumulator where the contraction adds to one,
    /// then a tile product's shape ([`MatrixLayout::Tiles`]), and last a
    /// sparse outer product's control ([`OuterShape::selection`]).
    #[must_use]
    pub const fn input_count(self) -> usize {
        match self {
            // Its factors, its accumulator, then the value that names their
            // format at run time (FPMR).
            Self::FloatDot {
                elements:
                    FloatElements {
                        source: FloatSource::Fp8,
                        ..
                    },
                ..
            } => 4,
            Self::Dot(shape) if !shape.accumulates => 2,
            Self::Complex(ComplexShape {
                product:
                    ComplexProduct::Multiply {
                        accumulates: false, ..
                    },
                ..
            }) => 2,
            Self::MaskedDot { .. }
            | Self::Complex(ComplexShape {
                product: ComplexProduct::Rotate(_),
                ..
            }) => 2,
            Self::Matrix(MatrixShape {
                layout: MatrixLayout::Tiles,
                ..
            }) => 6,
            // Its factors, its accumulator, then FPMR.
            Self::Matrix(MatrixShape { elements, .. }) if elements.names_format_at_run_time() => 4,
            Self::Dot(_)
            | Self::FloatDot { .. }
            | Self::MultiplyAdd52 { .. }
            | Self::Matrix(_)
            | Self::Complex(_) => 3,
            // Two predicates where it has them, the first factor's values,
            // the second factor, a sparse product's control, the tile, and
            // FPMR where its factors are FP8.
            Self::Outer(OuterShape {
                elements,
                governed,
                selection,
                ..
            }) => {
                let (values, control) = match selection {
                    Some(OuterSelection { values, .. }) => (values as usize, 1),
                    None => (1, 0),
                };
                values
                    .saturating_add(control)
                    .saturating_add(if governed { 4 } else { 2 })
                    .saturating_add(if elements.names_format_at_run_time() {
                        1
                    } else {
                        0
                    })
            }
        }
    }

    /// Whether the contraction computes floats.
    #[must_use]
    pub const fn is_float(self) -> bool {
        match self {
            Self::FloatDot { .. } | Self::MaskedDot { .. } => true,
            Self::Dot(_) | Self::MultiplyAdd52 { .. } => false,
            Self::Matrix(shape) => shape.elements.is_float(),
            Self::Outer(shape) => shape.elements.is_float(),
            Self::Complex(shape) => shape.element.kind.is_float(),
        }
    }

    /// Whether the contraction may raise float exceptions: one that computes
    /// floats, but for x86's bfloat16 dot product and AMX's tile products,
    /// which neither consult nor update MXCSR, and Arm's FP8 arithmetic of
    /// every shape, which raises nothing ([`FloatDotRule::Fp8`]).
    #[must_use]
    pub const fn raises_float_exceptions(self) -> bool {
        match self {
            Self::FloatDot {
                rule: FloatDotRule::Sequential | FloatDotRule::Fp8,
                ..
            }
            | Self::Matrix(MatrixShape {
                layout: MatrixLayout::Tiles,
                ..
            })
            | Self::Matrix(MatrixShape {
                elements:
                    ContractElements::Float(FloatElements {
                        source: FloatSource::Fp8,
                        ..
                    }),
                ..
            })
            | Self::Outer(OuterShape {
                elements:
                    ContractElements::Float(FloatElements {
                        source: FloatSource::Fp8,
                        ..
                    }),
                ..
            }) => false,
            _ => self.is_float(),
        }
    }
}

/// The integer lanes a dot product multiplies and sums, and how it adds the
/// sum: each destination lane is the sum of the products of the source lanes
/// it covers, the first source's lanes read [`Self::signed_a`], the second's
/// [`Self::signed_b`], plus the accumulator's lane where the product
/// [`Self::accumulates`], wrapped to the lane or saturated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DotShape {
    /// The width of the lanes multiplied, in bits.
    pub source_bits: u32,
    /// The width of the lanes summed into, in bits: a whole number of source
    /// lanes, the group one destination lane sums.
    pub dest_bits: u32,
    /// Whether the first source's lanes are signed.
    pub signed_a: bool,
    /// Whether the second source's lanes are signed.
    pub signed_b: bool,
    /// Whether the sum is added to an accumulator's lanes.
    pub accumulates: bool,
    /// How the sum fits the lane: `None` wraps, `Some(true)` saturates it to
    /// the signed range and `Some(false)` to the unsigned one.
    pub saturates: Option<bool>,
}

/// The float factors a contraction multiplies ([`ContractElements::Float`],
/// [`Contraction::FloatDot`]): a stated format, or the 8-bit one a register
/// names at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FloatSource {
    /// Factors of the format the instruction states.
    Format(FloatFormat),
    /// 8-bit factors whose encoding and scale the program selects at run
    /// time (Arm's FP8, as FPMR states): their width is 8 whichever
    /// encoding it names, so a reader counting lanes has its answer, while
    /// one reading the values has none until the register's contents are
    /// known.
    Fp8,
}

impl FloatSource {
    /// Returns the format the factors are in, or `None` where a register
    /// names it at run time.
    #[must_use]
    pub const fn format(self) -> Option<FloatFormat> {
        match self {
            Self::Format(format) => Some(format),
            Self::Fp8 => None,
        }
    }

    /// Returns the width of one factor, in bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Format(format) => format.bits(),
            Self::Fp8 => 8,
        }
    }
}

/// The source as SSA display writes it: the format's own name (`bf16`,
/// `f16`, `f32`), or `fp8` for the factors a register names at run time.
impl fmt::Display for FloatSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(format) => f.write_str(format.printed()),
            Self::Fp8 => f.write_str("fp8"),
        }
    }
}

/// The floats a contraction multiplies and the ones it sums them into
/// ([`ContractElements::Float`], [`Contraction::FloatDot`]): each element of
/// the `dest` format the sum of [`Self::group`] products of factors
/// `source` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FloatElements {
    /// The factors' format.
    pub source: FloatSource,
    /// The accumulator's format.
    pub dest: FloatFormat,
    /// How many products one accumulator element sums: as many factors as
    /// it is wide ([`Self::new`]), or fewer where the instruction multiplies
    /// fewer of the factors its lane covers ([`Self::with_group`]).
    pub group: u8,
}

impl FloatElements {
    /// Factors of `source` summed into elements of `dest`, each element the
    /// sum of as many products as it is wide over one factor, none where it
    /// holds less than one.
    #[must_use]
    pub const fn new(source: FloatSource, dest: FloatFormat) -> Self {
        Self {
            source,
            dest,
            group: Self::widths_group(source, dest),
        }
    }

    /// The same factors and accumulator, each element the sum of `group`
    /// products: one for a widening multiply-add (Arm FP8 `fmlalb`), two
    /// for a pair dot into elements four factors wide (FP8 `fvdotb`).
    #[must_use]
    pub const fn with_group(self, group: u8) -> Self {
        Self { group, ..self }
    }

    /// Returns how many products one accumulator element sums.
    #[must_use]
    pub const fn group(self) -> u32 {
        self.group as u32
    }

    /// Whether the group is the one the widths give ([`Self::new`]), which
    /// a fingerprint leaves unstated.
    #[must_use]
    pub const fn group_is_the_widths(self) -> bool {
        self.group == Self::widths_group(self.source, self.dest)
    }

    /// The products an element of `dest` covers of factors of `source`: its
    /// width over one factor's, 0 where it holds less than one.
    const fn widths_group(source: FloatSource, dest: FloatFormat) -> u8 {
        match dest.bits().checked_div(source.bits()) {
            Some(group) if group <= u8::MAX as u32 => group as u8,
            _ => 0,
        }
    }
}

/// The elements as SSA display writes them: the factors' format, then the
/// accumulator's - `bf16 -> f32`, `fp8 -> f16`.
impl fmt::Display for FloatElements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.source, self.dest.printed())?;
        if !self.group_is_the_widths() {
            write!(f, " x{}", self.group)?;
        }
        Ok(())
    }
}

/// The elements a matrix or outer-product contraction multiplies and the
/// ones it sums into: integers or floats, each accumulator element the sum
/// of [`Self::group`] products per pair of factor positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ContractElements {
    /// Integer factors `source_bits` wide summed into integer elements
    /// `dest_bits` wide, wrapping.
    Integer {
        /// The width of the factors' elements, in bits.
        source_bits: u16,
        /// The width of the accumulator's elements, in bits.
        dest_bits: u16,
        /// Whether the first factor's elements are signed.
        signed_a: bool,
        /// Whether the second factor's elements are signed.
        signed_b: bool,
    },
    /// Float factors summed into float elements ([`FloatElements`]).
    Float(FloatElements),
}

impl ContractElements {
    /// Returns the products one accumulator element sums for each pair of
    /// factor positions: the accumulator's width over the factors'.
    #[must_use]
    pub const fn group(self) -> u32 {
        let (source, dest) = match self {
            Self::Integer {
                source_bits,
                dest_bits,
                ..
            } => (source_bits as u32, dest_bits as u32),
            Self::Float(elements) => return elements.group(),
        };
        match dest.checked_div(source) {
            Some(group) => group,
            None => 0,
        }
    }

    /// Whether the elements are floats.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float(_))
    }

    /// Whether the factors' format is named at run time
    /// ([`FloatSource::Fp8`]), which the contraction then reads as its last
    /// input.
    #[must_use]
    pub const fn names_format_at_run_time(self) -> bool {
        matches!(
            self,
            Self::Float(FloatElements {
                source: FloatSource::Fp8,
                ..
            })
        )
    }
}

/// The elements as SSA display writes them: `i8xu8 -> 32` for integers, the
/// first factor's sign and width, the second's, then the accumulator's
/// width; the factors' format and the accumulator's for floats
/// ([`FloatElements`]).
impl fmt::Display for ContractElements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Integer {
                source_bits,
                dest_bits,
                signed_a,
                signed_b,
            } => {
                let sign = |signed: bool| if signed { 'i' } else { 'u' };
                write!(
                    f,
                    "{}{source_bits}x{}{source_bits} -> {dest_bits}",
                    sign(signed_a),
                    sign(signed_b)
                )
            }
            Self::Float(elements) => write!(f, "{elements}"),
        }
    }
}

/// How a matrix contraction lays its matrices out in its operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MatrixLayout {
    /// Arm's `smmla`, `ummla`, `usmmla`, `fmmla` and `bfmmla` (A64, SVE and
    /// A32): each `segment_bits` of the first factor holds a matrix of two
    /// rows in row order, each segment of the second the transpose of a
    /// matrix of two columns, and each segment of the accumulator their 2 by
    /// 2 product, each element the sum of a row's and a column's products
    /// added to it as the instruction's pseudocode orders and rounds them.
    Segments {
        /// The width of a segment, in bits: 128, or 256 for SVE's
        /// double-precision `fmmla`.
        segment_bits: u16,
    },
    /// AMX's `tdp*` tiles, each 16 rows of 64 bytes, the rows laid one after
    /// another: the product's shape is three inputs after the accumulator,
    /// the rows of the first factor and of the result (M), the first
    /// factor's row bytes (K groups of four) and the result's row bytes (N
    /// elements of four bytes), as the tile configuration states them. Row
    /// `k` of the second factor packs the products one 32-bit accumulator
    /// element sums (Intel's VNNI layout); floats round to nearest even with
    /// denormals flushed, raising nothing. The result's bytes past its row
    /// bytes, and its rows past M, are zero.
    Tiles,
}

/// The part of each complex product a complex matrix contraction adds (AMX
/// `tcmmrlfp16ps`, `tcmmimfp16ps`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComplexPart {
    /// The real part.
    Real,
    /// The imaginary part.
    Imaginary,
}

/// A matrix product added to an accumulator matrix ([`Contraction::Matrix`]):
/// its elements, its layout, and, for a product of complex elements (each a
/// pair of float elements, the real part first), the part each product
/// contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MatrixShape {
    /// The factors' and the accumulator's elements.
    pub elements: ContractElements,
    /// Where the matrices lie in the operands.
    pub layout: MatrixLayout,
    /// The part of each complex product added, for complex factors.
    pub complex: Option<ComplexPart>,
}

/// An outer product added to or subtracted from a tile
/// ([`Contraction::Outer`], Arm SME's `*mopa` and `*mops`): element `(i, j)`
/// of the tile takes the products of the first factor's group `i` and the
/// second's group `j`, element by element and summed, where the first
/// governing predicate's group `i` and the second's group `j` are active, as
/// the instruction's pseudocode orders and rounds them.
///
/// A sparse product ([`Self::selection`], `*tmopa`) sums the same products
/// of fewer elements: a control value it reads before the tile names which
/// of the first factor's candidates each group multiplies. It is also
/// unpredicated ([`Self::governed`]), so it reads no predicate before its
/// factors - two facts of one instruction, each stated on its own, because
/// neither follows from the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OuterShape {
    /// The factors' and the tile's elements.
    pub elements: ContractElements,
    /// Whether the sum is subtracted from the tile (`*mops`).
    pub subtract: bool,
    /// What the product of two elements is.
    pub product: OuterProduct,
    /// Whether two governing predicates select the groups that take part,
    /// which the contraction then reads before the factors: Arm's `*mopa`
    /// and `*mops` have them and its sparse `*tmopa` does not.
    pub governed: bool,
    /// How a sparse product chooses the first factor's elements, `None`
    /// where every element of a group is multiplied.
    pub selection: Option<OuterSelection>,
}

/// How a sparse outer product chooses the elements each group of products
/// multiplies ([`OuterShape::selection`], Arm SME2's `*tmopa`).
///
/// The first factor is `values` of the contraction's inputs, each offering
/// `candidates` elements per group, and the control input holds one bit per
/// candidate: `values * candidates` bits for each group of the second
/// factor, a group's bits together and the groups in order. So the choice is
/// a property of the column and the same for every row of the tile, and a
/// group multiplies the candidates whose bit is set, lowest first, the rest
/// reading zero, which adds nothing to the sum.
///
/// `per_value` says how a group's products - [`ContractElements::group`] of
/// them - are divided among those values: each value fills its own equal
/// share (Arm's 4-way `*tmopa`, four products of eight candidates, two from
/// each of two vectors), or they fill the group in turn and a value whose
/// bits are all set can fill it alone (its 2-way form, two products of four
/// candidates, either or both from one vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OuterSelection {
    /// How many of the contraction's inputs the first factor spans.
    pub values: u8,
    /// The candidates each of those offers a group of products.
    pub candidates: u8,
    /// Whether each value fills its own equal share of a group's products,
    /// rather than the values filling the group in turn.
    pub per_value: bool,
}

/// What an outer product ([`OuterShape`]) takes of a pair of elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OuterProduct {
    /// Their product.
    #[default]
    Multiply,
    /// The count of the bits in which they agree, `popcount(!(a ^ b))`: Arm
    /// SME2's `bmopa` and `bmops`.
    MatchingBits,
}

/// The turn a complex rotation applies to its second source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComplexRotation {
    /// A quarter turn: multiplied by `i` (`#90`).
    Quarter,
    /// Three quarter turns: multiplied by `-i` (`#270`).
    ThreeQuarters,
}

/// What a complex contraction computes of complex elements, each a pair of
/// lanes, the real part in the even lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComplexProduct {
    /// The product of the two factors, the second conjugated where
    /// `conjugate` states (x86 `vfmulcph`, `vfcmulcph`), added to the
    /// accumulator where `accumulates` states (`vfmaddcph`, `vfcmaddcph`).
    Multiply {
        /// Whether the second factor is conjugated.
        conjugate: bool,
        /// Whether the product is added to an accumulator.
        accumulates: bool,
    },
    /// The first source plus the second turned by `rotation` (Arm `fcadd`,
    /// `cadd`, `sqcadd`).
    Rotate(ComplexRotation),
}

/// A complex contraction ([`Contraction::Complex`]): what it computes, and
/// the lanes it computes on, a float element's rounding the one its
/// instruction states in place of the environment's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ComplexShape {
    /// What is computed.
    pub product: ComplexProduct,
    /// The lanes the parts lie in.
    pub element: VectorElement,
    /// Whether each part saturates to the signed range (`sqcadd`).
    pub saturates: bool,
}

/// The SVE float helper step a [`Kernel::FloatHelper`] computes, lane by
/// lane: the transcendental-acceleration primitives.
///
/// [`Kernel::FloatHelper`]: crate::ir::ops::Kernel::FloatHelper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum FpHelperKind {
    /// Reciprocal exponent (`frecpx`).
    ReciprocalExponent,
    /// Extract the biased exponent (`flogb`).
    ExtractExponent,
    /// Exponential acceleration table lookup (`fexpa`).
    ExpAccelerate,
    /// Trigonometric multiply-add (`ftmad`).
    TrigMulAdd,
    /// Trigonometric select-multiply (`ftsmul`).
    TrigSelectMul,
    /// Trigonometric select (`ftssel`).
    TrigSelect,
}

impl FpHelperKind {
    /// Returns the inputs: the source of `frecpx`, `flogb` and `fexpa`, the
    /// two of `ftssel` and `ftsmul`, and `ftmad`'s two and its immediate.
    #[must_use]
    pub const fn input_count(self) -> usize {
        match self {
            Self::ReciprocalExponent | Self::ExtractExponent | Self::ExpAccelerate => 1,
            Self::TrigSelect | Self::TrigSelectMul => 2,
            Self::TrigMulAdd => 3,
        }
    }

    /// Returns the IR name of this helper, such as `recip.exponent`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ReciprocalExponent => "recip.exponent",
            Self::ExtractExponent => "exponent.int",
            Self::ExpAccelerate => "exp.accel",
            Self::TrigMulAdd => "trig.muladd",
            Self::TrigSelectMul => "trig.select_mul",
            Self::TrigSelect => "trig.select",
        }
    }

    /// Returns the stable display / fingerprint key for this helper.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::ReciprocalExponent => "fphelper.frecpx",
            Self::ExtractExponent => "fphelper.flogb",
            Self::ExpAccelerate => "fphelper.fexpa",
            Self::TrigMulAdd => "fphelper.ftmad",
            Self::TrigSelectMul => "fphelper.ftsmul",
            Self::TrigSelect => "fphelper.ftssel",
        }
    }

    /// The lanes this helper computes of `inputs`, element by element, as the
    /// Arm architecture defines it, the lanes binary16, binary32 or binary64
    /// as `element` states.
    ///
    /// - [`Self::ReciprocalExponent`] of `[x]`: `x`'s sign with its exponent
    ///   complemented and a zero fraction, the largest finite exponent for a
    ///   zero or subnormal `x` (`frecpx`).
    /// - [`Self::ExtractExponent`] of `[x]`: the unbiased exponent as a signed
    ///   integer of the lane's width, its maximum for an infinity and its
    ///   minimum for a zero (`flogb`).
    /// - [`Self::ExpAccelerate`] of `[x]`: the fraction of `2^(i / 32)` (half)
    ///   or `2^(i / 64)` from the table of the low bits' index `i`, under the
    ///   exponent the bits above them state (`fexpa`).
    /// - [`Self::TrigMulAdd`] of `[n, m, index]`: `n * |m|` plus the sine
    ///   series coefficient `index` (`m` positive) or the cosine one (`m`
    ///   negative), rounded once (`ftmad`); `index` is the immediate, a scalar.
    /// - [`Self::TrigSelectMul`] of `[n, m]`: `n * n` with the sign of `m`'s
    ///   bit 0 (`ftsmul`).
    /// - [`Self::TrigSelect`] of `[n, m]`: `1.0` where `m`'s bit 0 is set and
    ///   `n` otherwise, negated where its bit 1 is (`ftssel`).
    ///
    /// The products round by `element`'s rounding, else by `environment`, and
    /// fold only where the rounding decides nothing or is known. A NaN folds
    /// nothing where the processor's alternate handling or default-NaN mode
    /// chooses the result, and `flogb` of a subnormal nothing, which
    /// flush-to-zero would read as zero.
    ///
    /// # Returns
    ///
    /// The result's lanes, or `None` for another format, inputs of another
    /// count or shape, and a lane that does not fold.
    #[must_use]
    pub fn evaluate_lanes(
        self,
        inputs: &[VectorConst],
        element: VectorElement,
        environment: Option<FloatRounding>,
    ) -> Option<VectorConst> {
        let format = match (element.kind, element.bits) {
            (VectorElementKind::Float, 16) => FloatFormat::Half,
            (VectorElementKind::Float, 32) => FloatFormat::Single,
            (VectorElementKind::Float, 64) => FloatFormat::Double,
            _ => return None,
        };
        let rounding = element.rounding_in(environment);
        let bits = element.bits;
        let (data, index) = match (self, inputs) {
            (Self::TrigMulAdd, [n, m, index]) if index.lanes() == 1 => (
                vec![n.relaned(bits)?, m.relaned(bits)?],
                Some(index.lane(0)),
            ),
            (Self::TrigMulAdd, _) => return None,
            (_, inputs) => (
                inputs
                    .iter()
                    .map(|input| input.relaned(bits))
                    .collect::<Option<Vec<_>>>()?,
                None,
            ),
        };
        let lanes = data.first()?.lanes();
        if data.iter().any(|input| input.lanes() != lanes) || data.len() != self.data_inputs() {
            return None;
        }
        let values = (0..lanes)
            .map(|lane| {
                let operands = data
                    .iter()
                    .map(|input| input.lane(lane))
                    .collect::<Vec<_>>();
                self.lane(&operands, format, index, rounding)
            })
            .collect::<Option<Vec<_>>>()?;
        VectorConst::from_lanes(bits, values)
    }

    /// How many vector inputs the helper reads, its immediate apart.
    const fn data_inputs(self) -> usize {
        match self {
            Self::ReciprocalExponent | Self::ExtractExponent | Self::ExpAccelerate => 1,
            Self::TrigMulAdd | Self::TrigSelectMul | Self::TrigSelect => 2,
        }
    }

    /// One lane of the helper, of its operands' bits.
    fn lane(
        self,
        operands: &[u128],
        format: FloatFormat,
        index: Option<u128>,
        rounding: Option<FloatRounding>,
    ) -> Option<u128> {
        let bits = format.bits();
        let fraction = format.fraction_bits();
        let exponent_mask = BitWidth::new(format.exponent_bits()).checked_mask_u128()?;
        let sign = 1u128.checked_shl(bits.checked_sub(1)?)?;
        let float = |value: u128| FloatBits::new(format, value);
        match (self, operands) {
            (Self::ReciprocalExponent, [x]) => {
                let value = float(*x)?;
                if value.is_nan() {
                    return None;
                }
                let exponent = (x >> fraction) & exponent_mask;
                let result = if exponent == 0 {
                    exponent_mask.checked_sub(1)?
                } else {
                    !exponent & exponent_mask
                };
                Some(x & sign | result << fraction)
            }
            (Self::ExtractExponent, [x]) => {
                let value = float(*x)?;
                let exponent = (x >> fraction) & exponent_mask;
                let significand = x & BitWidth::new(fraction).checked_mask_u128()?;
                let bias = exponent_mask >> 1;
                let lane = BitWidth::new(bits).checked_mask_u128()?;
                let result = match (exponent, significand) {
                    // A zero, and a NaN, is the least integer.
                    (0, 0) => sign,
                    _ if value.is_nan() => sign,
                    // Flush-to-zero decides a subnormal.
                    (0, _) => return None,
                    (exponent, 0) if exponent == exponent_mask => sign.checked_sub(1)?,
                    (exponent, _) => {
                        i128::try_from(exponent)
                            .ok()?
                            .checked_sub(i128::try_from(bias).ok()?)?
                            .cast_unsigned()
                            & lane
                    }
                };
                Some(result)
            }
            (Self::ExpAccelerate, [x]) => {
                let (index_bits, table): (u32, &[u64]) = match format {
                    FloatFormat::Half => (5, &EXP_ACCELERATE_HALF),
                    FloatFormat::Single => (6, &EXP_ACCELERATE_SINGLE),
                    FloatFormat::Double => (6, &EXP_ACCELERATE_DOUBLE),
                    _ => return None,
                };
                let index =
                    usize::try_from(x & BitWidth::new(index_bits).checked_mask_u128()?).ok()?;
                let exponent = (x >> index_bits) & exponent_mask;
                Some(u128::from(*table.get(index)?) | exponent << fraction)
            }
            (Self::TrigMulAdd, [n, m]) => {
                let (n, m) = (float(*n)?, float(*m)?);
                if n.is_nan() || m.is_nan() {
                    return None;
                }
                // The immediate is three bits, the series' term; a negative
                // `m` reads the cosine series, eight entries on.
                let index = usize::try_from(index?).ok().filter(|index| *index < 8)?;
                let index = if m.bits() & sign != 0 {
                    index.checked_add(8)?
                } else {
                    index
                };
                let table: &[u64] = match format {
                    FloatFormat::Half => &TRIG_COEFFICIENTS_HALF,
                    FloatFormat::Single => &TRIG_COEFFICIENTS_SINGLE,
                    FloatFormat::Double => &TRIG_COEFFICIENTS_DOUBLE,
                    _ => return None,
                };
                let coefficient = float(u128::from(*table.get(index)?))?;
                Some(n.checked_mul_add(m.abs(), coefficient, rounding)?.bits())
            }
            (Self::TrigSelectMul, [n, m]) => {
                let n = float(*n)?;
                let square = n.checked_mul(n, rounding)?;
                if square.is_nan() {
                    return None;
                }
                let square = square.abs();
                Some(if m & 1 == 1 { square.negate() } else { square }.bits())
            }
            (Self::TrigSelect, [n, m]) => {
                let one = format.from_integer(1);
                let chosen = if m & 1 == 1 { one } else { *n };
                if m & 2 == 0 {
                    return Some(chosen);
                }
                let chosen = float(chosen)?;
                // The alternate handling decides a NaN's negation.
                if chosen.is_nan() {
                    return None;
                }
                Some(chosen.negate().bits())
            }
            _ => None,
        }
    }
}

/// `fexpa`'s binary16 fractions, `2^(i / 32)` for each index `i`, from the
/// Arm architecture's `FPExpA`.
const EXP_ACCELERATE_HALF: [u64; 32] = [
    0x0000, 0x0016, 0x002d, 0x0045, 0x005d, 0x0075, 0x008e, 0x00a8, 0x00c2, 0x00dc, 0x00f8, 0x0114,
    0x0130, 0x014d, 0x016b, 0x0189, 0x01a8, 0x01c8, 0x01e8, 0x0209, 0x022b, 0x024e, 0x0271, 0x0295,
    0x02ba, 0x02e0, 0x0306, 0x032e, 0x0356, 0x037f, 0x03a9, 0x03d4,
];

/// `fexpa`'s binary32 fractions, `2^(i / 64)` for each index `i`, from the
/// Arm architecture's `FPExpA`.
const EXP_ACCELERATE_SINGLE: [u64; 64] = [
    0x000000, 0x0164d2, 0x02cd87, 0x043a29, 0x05aac3, 0x071f62, 0x08980f, 0x0a14d5, 0x0b95c2,
    0x0d1adf, 0x0ea43a, 0x1031dc, 0x11c3d3, 0x135a2b, 0x14f4f0, 0x16942d, 0x1837f0, 0x19e046,
    0x1b8d3a, 0x1d3eda, 0x1ef532, 0x20b051, 0x227043, 0x243516, 0x25fed7, 0x27cd94, 0x29a15b,
    0x2b7a3a, 0x2d583f, 0x2f3b79, 0x3123f6, 0x3311c4, 0x3504f3, 0x36fd92, 0x38fbaf, 0x3aff5b,
    0x3d08a4, 0x3f179a, 0x412c4d, 0x4346cd, 0x45672a, 0x478d75, 0x49b9be, 0x4bec15, 0x4e248c,
    0x506334, 0x52a81e, 0x54f35b, 0x5744fd, 0x599d16, 0x5bfbb8, 0x5e60f5, 0x60ccdf, 0x633f89,
    0x65b907, 0x68396a, 0x6ac0c7, 0x6d4f30, 0x6fe4ba, 0x728177, 0x75257d, 0x77d0df, 0x7a83b3,
    0x7d3e0c,
];

/// `fexpa`'s binary64 fractions, `2^(i / 64)` for each index `i`, from the
/// Arm architecture's `FPExpA`.
const EXP_ACCELERATE_DOUBLE: [u64; 64] = [
    0x0000000000000,
    0x02C9A3E778061,
    0x059B0D3158574,
    0x0874518759BC8,
    0x0B5586CF9890F,
    0x0E3EC32D3D1A2,
    0x11301D0125B51,
    0x1429AAEA92DE0,
    0x172B83C7D517B,
    0x1A35BEB6FCB75,
    0x1D4873168B9AA,
    0x2063B88628CD6,
    0x2387A6E756238,
    0x26B4565E27CDD,
    0x29E9DF51FDEE1,
    0x2D285A6E4030B,
    0x306FE0A31B715,
    0x33C08B26416FF,
    0x371A7373AA9CB,
    0x3A7DB34E59FF7,
    0x3DEA64C123422,
    0x4160A21F72E2A,
    0x44E086061892D,
    0x486A2B5C13CD0,
    0x4BFDAD5362A27,
    0x4F9B2769D2CA7,
    0x5342B569D4F82,
    0x56F4736B527DA,
    0x5AB07DD485429,
    0x5E76F15AD2148,
    0x6247EB03A5585,
    0x6623882552225,
    0x6A09E667F3BCD,
    0x6DFB23C651A2F,
    0x71F75E8EC5F74,
    0x75FEB564267C9,
    0x7A11473EB0187,
    0x7E2F336CF4E62,
    0x82589994CCE13,
    0x868D99B4492ED,
    0x8ACE5422AA0DB,
    0x8F1AE99157736,
    0x93737B0CDC5E5,
    0x97D829FDE4E50,
    0x9C49182A3F090,
    0xA0C667B5DE565,
    0xA5503B23E255D,
    0xA9E6B5579FDBF,
    0xAE89F995AD3AD,
    0xB33A2B84F15FB,
    0xB7F76F2FB5E47,
    0xBCC1E904BC1D2,
    0xC199BDD85529C,
    0xC67F12E57D14B,
    0xCB720DCEF9069,
    0xD072D4A07897C,
    0xD5818DCFBA487,
    0xDA9E603DB3285,
    0xDFC97337B9B5F,
    0xE502EE78B3FF6,
    0xEA4AFA2A490DA,
    0xEFA1BEE615A27,
    0xF50765B6E4540,
    0xFA7C1819E90D8,
];

/// `ftmad`'s binary16 coefficients, the sine series' (indices 0 to 7) then
/// the cosine series' (8 to 15), from the Arm architecture's `FPTrigMAdd`.
const TRIG_COEFFICIENTS_HALF: [u64; 16] = [
    0x3c00, 0xb155, 0x2030, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x3c00, 0xb800, 0x293a, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000,
];

/// `ftmad`'s binary32 coefficients, the sine series' (indices 0 to 7) then
/// the cosine series' (8 to 15), from the Arm architecture's `FPTrigMAdd`.
const TRIG_COEFFICIENTS_SINGLE: [u64; 16] = [
    0x3f800000, 0xbe2aaaab, 0x3c088886, 0xb95008b9, 0x36369d6d, 0x00000000, 0x00000000, 0x00000000,
    0x3f800000, 0xbf000000, 0x3d2aaaa6, 0xbab60705, 0x37cd37cc, 0x00000000, 0x00000000, 0x00000000,
];

/// `ftmad`'s binary64 coefficients, the sine series' (indices 0 to 7) then
/// the cosine series' (8 to 15), from the Arm architecture's `FPTrigMAdd`.
const TRIG_COEFFICIENTS_DOUBLE: [u64; 16] = [
    0x3ff0000000000000,
    0xbfc5555555555543,
    0x3f8111111110f30c,
    0xbf2a01a019b92fc6,
    0x3ec71de351f3d22b,
    0xbe5ae5e2b60f7b91,
    0x3de5d8408868552f,
    0x0000000000000000,
    0x3ff0000000000000,
    0xbfe0000000000000,
    0x3fa5555555555536,
    0xbf56c16c16c13a0b,
    0x3efa01a019b1e8d8,
    0xbe927e4f7282f468,
    0x3e21ee96d2641b13,
    0xbda8f76380fbb401,
];

/// The elements an SVE predicate-count pattern selects of the elements a
/// vector holds (Arm `DecodePredCount`): the largest power of two, a fixed
/// number where the vector holds that many, the largest multiple of four or
/// of three, or all of them. A pattern the encoding reserves selects none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ElementPattern {
    /// The largest power of two not above the count (`pow2`).
    Pow2,
    /// The number named, where the count is at least it, and none otherwise
    /// (`vl1` to `vl8`, `vl16` to `vl256`).
    Fixed(u16),
    /// The largest multiple of four not above the count (`mul4`).
    Mul4,
    /// The largest multiple of three not above the count (`mul3`).
    Mul3,
    /// Every element (`all`).
    All,
    /// None: an encoding the architecture reserves (`#uimm5`).
    Reserved(u8),
}

/// The pattern as the assembly writes it: `pow2`, `vl16`, `mul4`, `all`, or
/// the reserved field's number.
impl fmt::Display for ElementPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pow2 => f.write_str("pow2"),
            Self::Fixed(count) => write!(f, "vl{count}"),
            Self::Mul4 => f.write_str("mul4"),
            Self::Mul3 => f.write_str("mul3"),
            Self::All => f.write_str("all"),
            Self::Reserved(bits) => write!(f, "#{bits}"),
        }
    }
}

impl ElementPattern {
    /// The pattern a 5-bit `pattern` field encodes.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Pow2,
            1..=8 => Self::Fixed(bits as u16),
            9 => Self::Fixed(16),
            10 => Self::Fixed(32),
            11 => Self::Fixed(64),
            12 => Self::Fixed(128),
            13 => Self::Fixed(256),
            29 => Self::Mul4,
            30 => Self::Mul3,
            31 => Self::All,
            other => Self::Reserved(other),
        }
    }

    /// The 5-bit field encoding the pattern, the inverse of
    /// [`Self::from_bits`]; `None` for a fixed count no encoding names.
    #[must_use]
    pub const fn bits(self) -> Option<u8> {
        Some(match self {
            Self::Pow2 => 0,
            Self::Fixed(count @ 1..=8) => count as u8,
            Self::Fixed(16) => 9,
            Self::Fixed(32) => 10,
            Self::Fixed(64) => 11,
            Self::Fixed(128) => 12,
            Self::Fixed(256) => 13,
            Self::Fixed(_) => return None,
            Self::Mul4 => 29,
            Self::Mul3 => 30,
            Self::All => 31,
            Self::Reserved(bits) => bits,
        })
    }

    /// The number of elements the pattern selects of `count`.
    #[must_use]
    pub const fn select(self, count: u64) -> u64 {
        match self {
            Self::Pow2 if count == 0 => 0,
            Self::Pow2 => 1_u64.wrapping_shl(count.ilog2()),
            Self::Fixed(fixed) if count >= fixed as u64 => fixed as u64,
            Self::Fixed(_) | Self::Reserved(_) => 0,
            Self::Mul4 => count & !3,
            Self::Mul3 => count.wrapping_sub(count % 3),
            Self::All => count,
        }
    }
}

/// Boxed payload for [`SsaOp::VectorExtendInLane`] - sign- or zero-extends the
/// low `source_bits` of each `element_bits`-wide lane to the full lane width
/// (AArch64 SVE predicated `sxtb`/`uxtb`/`sxth`/`uxth`/`sxtw`/`uxtw`, and the
/// `sxtw`/`uxtw` index extension performed by SVE `adr`). The `inputs` are
/// `[src]`: a governing predicate merges the result as a predicated identity
/// ([`SsaOp::VectorPredicatedUnary`]) does. Pure.
///
/// [`SsaOp::VectorPredicatedUnary`]: crate::ir::ops::def::SsaOp::VectorPredicatedUnary
///
/// [`SsaOp::VectorExtendInLane`]: crate::ir::ops::def::SsaOp::VectorExtendInLane
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorExtendInLaneData {
    /// `true` to sign-extend (`sxt*`), `false` to zero-extend (`uxt*`).
    pub signed: bool,
    /// Width in bits of the low field that is extended (8/16/32).
    pub source_bits: u16,
    /// Full lane width in bits (16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the extended vector).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the source vector, then any governing predicate.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorElementCount`] - computes the element count
/// `VL / element_bits` of a vector length, scaled by `multiplier`, into a
/// scalar register (AArch64 SVE `cntb`/`cnth`/`cntw`/`cntd`, SME `rdsvl`, RISC-V
/// `VLMAX`). The count is symbolic (the vector length is runtime-defined); the
/// op names the granularity so no concrete materialization is needed. A
/// machine with two vector lengths states which one the count is of as an
/// operand, so a count of a mode the function does not know stays a function
/// of the mode. Pure.
///
/// [`SsaOp::VectorElementCount`]: crate::ir::ops::def::SsaOp::VectorElementCount
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorElementCountData {
    /// Element width selecting the count granularity (`8`/`16`/`32`/`64`).
    pub element_bits: u16,
    /// Constant multiplier from the `mul #imm` form (`1` when absent).
    pub multiplier: u32,
    /// The elements of the count the pattern selects (`all` where no
    /// pattern is written).
    pub pattern: ElementPattern,
    /// On a machine with two vector lengths, the boolean that holds where
    /// the count is of the streaming one (SME's streaming vector length)
    /// rather than the other: true for `rdsvl` and `addsvl`, and the mode the
    /// processor is in (`PSTATE.SM`) for `cntb`. `None` on a machine with one.
    pub streaming: Option<SsaVarId>,
    /// Explicit SSA outputs (the scalar count).
    pub outputs: Vec<SsaVarId>,
}

/// The SVE2/NEON compute operation named by [`VectorSveComputeData`]. One
/// variant per hardware operation keeps the lift lossless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SveComputeKind {
    /// `adclb`/`adclt`: add with carry into the bottom/top long lanes.
    AddCarryBottom,
    /// Add with carry into the top long lanes.
    AddCarryTop,
    /// `sbclb`/`sbclt`: subtract with carry into the bottom/top long lanes.
    SubCarryBottom,
    /// Subtract with carry into the top long lanes.
    SubCarryTop,
    /// `histcnt`: per-element histogram count.
    Histogram,
    /// `histseg`: segmented histogram match count.
    HistogramSegment,
    /// `match`: per-element membership predicate.
    MatchElements,
    /// `nmatch`: per-element non-membership predicate.
    NoMatchElements,
    /// `sclamp`: clamp each lane to a signed `[min,max]` range.
    ClampSigned,
    /// `uclamp`: clamp each lane to an unsigned `[min,max]` range.
    ClampUnsigned,
    /// `sdivr`: reversed signed divide (`b / a`).
    DivideReversedSigned,
    /// `udivr`: reversed unsigned divide (`b / a`).
    DivideReversedUnsigned,
    /// `eorbt`: interleaving exclusive-or, bottom from top.
    InterleaveXorBottomTop,
    /// `eortb`: interleaving exclusive-or, top from bottom.
    InterleaveXorTopBottom,
    /// `sqabs`: saturating absolute value.
    SaturatingAbs,
    /// `sqneg`: saturating negate.
    SaturatingNeg,
    /// `cnot`: logical NOT per lane (`x == 0 ? 1 : 0`).
    LogicalNot,
    /// `cdot`: complex integer dot product with rotation.
    ComplexDotProduct,
    /// `sqrdcmlah`: saturating rounding doubling complex multiply-accumulate.
    ComplexMulAddRounding,
}

impl SveComputeKind {
    /// Returns whether the operation reads its destination as well as its
    /// sources: the carrying adds and subtracts (`adclb`), the clamps
    /// (`sclamp Zd, Zn, Zm` clamps `Zd`), the interleaving exclusive-ors (the
    /// lanes they do not write keep `Zd`'s) and the complex accumulates
    /// (`cdot`, `sqrdcmlah`).
    #[must_use]
    pub const fn accumulates(self) -> bool {
        matches!(
            self,
            Self::AddCarryBottom
                | Self::AddCarryTop
                | Self::SubCarryBottom
                | Self::SubCarryTop
                | Self::ClampSigned
                | Self::ClampUnsigned
                | Self::InterleaveXorBottomTop
                | Self::InterleaveXorTopBottom
                | Self::ComplexDotProduct
                | Self::ComplexMulAddRounding
        )
    }
}

/// Boxed payload for [`SsaOp::VectorSveCompute`] - an SVE2/NEON compute op named
/// precisely by its [`SveComputeKind`]. `rotation` carries the complex-op rotation
/// in degrees (0/90/180/270; 0 otherwise); `element_bits` is the lane width. Pure.
///
/// The `inputs` are the destination's value for an operation that reads it
/// ([`SveComputeKind::accumulates`]), then for `histcnt` the predicate it
/// counts among, one bit per element (bit `i` element `i`'s), then the
/// sources in order. `match` and `nmatch` give one bit per element; a
/// governing predicate the operation does not count by merges or zeroes the
/// result as [`SsaOp::VectorPredicatedUnary`] does.
///
/// [`SsaOp::VectorPredicatedUnary`]: crate::ir::ops::def::SsaOp::VectorPredicatedUnary
///
/// [`SsaOp::VectorSveCompute`]: crate::ir::ops::def::SsaOp::VectorSveCompute
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorSveComputeData {
    /// The exact hardware operation.
    pub op: SveComputeKind,
    /// Lane width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Complex-op rotation in degrees (0/90/180/270); 0 when not applicable.
    pub rotation: u16,
    /// Explicit SSA outputs.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload for [`SsaOp::VectorStructLoadReplicate`] - AArch64 `ld2r`/`ld3r`/
/// `ld4r`: load a single 2/3/4-element structure from memory and replicate each
/// element across all lanes of its destination register.
///
/// Reads memory through its address base and may fault, so it classifies as
/// [`SsaEffectKind::Read`](crate::ir::ops::SsaEffectKind::Read) with [`TrapClass::MemoryFault`] - never pure.
///
/// [`SsaOp::VectorStructLoadReplicate`]: crate::ir::ops::def::SsaOp::VectorStructLoadReplicate
/// [`TrapClass::MemoryFault`]: crate::ir::ops::effects::TrapClass::MemoryFault
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorStructLoadReplicateData {
    /// Number of structure elements / destination registers (2, 3, or 4).
    pub count: u8,
    /// Element width in bits (8/16/32/64).
    pub element_bits: u16,
    /// Explicit SSA outputs (the replicated destination registers).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (the address base register).
    pub inputs: Vec<SsaVarId>,
}

/// A native instruction that writes the condition flags directly, or moves
/// them to or from a register: the IR's name for it.
///
/// No SSA operation has this identity: a lift states each as the flag
/// operations it is (`ReadFlags`, `InsertFlag`, `MergeFlags`), and an
/// interrupt or alignment-check write also as the machine operation it is (a
/// [`MachineStateOp`]).
///
/// **Partition rule against [`MachineStateOp`].** This enum names an
/// instruction whose effect on the flags is a write of bits [`FlagsMask`]
/// models. IF is EFLAGS bit 9 exactly as DF is bit 10 and AC is bit 18, so
/// `cli` and `sti` belong here beside `cld`/`std` and `clac`/`stac`.
/// Everything else is a [`MachineStateOp`]: `clui`/`stui` toggle UIF, which is
/// separate user-interrupt state [`FlagsMask`] cannot express, and MIPS
/// `di`/`ei` optionally return the prior status register.
///
/// [`MachineStateOp`]: crate::ir::ops::MachineStateOp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum FlagAdjustKind {
    /// `cfinv` - invert the carry flag.
    InvertCarry,
    /// `rmif` - rotate a register, mask, and insert into NZCV.
    RotateMaskInsert,
    /// `setf8` - set N and Z from the low 8 bits of a register.
    SetNzFrom8,
    /// `setf16` - set N and Z from the low 16 bits of a register.
    SetNzFrom16,
    /// `axflag` - convert PSTATE flags to the alternative (FP) format.
    ConvertToFpFlags,
    /// `xaflag` - convert alternative (FP) flags back to PSTATE format.
    ConvertFromFpFlags,
    /// Clear the carry flag (`clc`).
    ClearCarry,
    /// Set the carry flag (`stc`).
    SetCarry,
    /// Clear the string-direction flag (`cld`) - string operations advance.
    ClearDirection,
    /// Set the string-direction flag (`std`) - string operations retreat.
    SetDirection,
    /// Clear the alignment-check flag (`clac`).
    ClearAlignCheck,
    /// Set the alignment-check flag (`stac`).
    SetAlignCheck,
    /// Clear the interrupt-enable flag (`cli`) - maskable interrupts are held
    /// off until IF is set again.
    ClearInterrupt,
    /// Set the interrupt-enable flag (`sti`).
    SetInterrupt,
    /// Load the low status-flag byte from a register (`sahf`) - the input
    /// register supplies `SF`, `ZF`, `AF`, `PF` and `CF`; the output is the
    /// updated flags value.
    LoadStatusFromReg,
    /// Store the low status-flag byte into a register (`lahf`) - the input is
    /// the flags value and the output is the written register, so this is the
    /// one adjustment whose result is not a flags value.
    StoreStatusToReg,
    /// Set every bit of a register from the carry flag (`salc`) - the input is
    /// the flags value and the output is the written register.
    SetRegFromCarry,
}

impl FlagAdjustKind {
    /// Returns the IR name of this flag adjustment, such as `flags.cf.flip`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvertCarry => "flags.cf.flip",
            Self::RotateMaskInsert => "flags.insert.rotated",
            Self::SetNzFrom8 => "flags.set_nz.8",
            Self::SetNzFrom16 => "flags.set_nz.16",
            Self::ConvertToFpFlags => "flags.to_fp",
            Self::ConvertFromFpFlags => "flags.from_fp",
            Self::ClearCarry => "flags.cf.clear",
            Self::SetCarry => "flags.cf.set",
            Self::ClearDirection => "flags.dir.clear",
            Self::SetDirection => "flags.dir.set",
            Self::ClearAlignCheck => "flags.align.clear",
            Self::SetAlignCheck => "flags.align.set",
            Self::ClearInterrupt => "flags.intr.clear",
            Self::SetInterrupt => "flags.intr.set",
            Self::LoadStatusFromReg => "flags.from_reg",
            Self::StoreStatusToReg => "flags.to_reg",
            Self::SetRegFromCarry => "flags.cf.spread",
        }
    }

    /// Returns the architectural flag bits this adjustment writes.
    ///
    /// Callers use it to decide whether a pending comparison survives the
    /// adjustment: an op that writes only control flags (`cld`, `cli`, `stac`)
    /// leaves a `cmp`/`jcc` pair foldable, while one that touches the status
    /// flags does not.
    #[must_use]
    pub const fn flags_written(self) -> FlagsMask {
        match self {
            Self::InvertCarry | Self::ClearCarry | Self::SetCarry => FlagsMask::CARRY,
            Self::ClearDirection | Self::SetDirection => FlagsMask::DIRECTION,
            Self::ClearAlignCheck | Self::SetAlignCheck => FlagsMask::ALIGN_CHECK,
            Self::ClearInterrupt | Self::SetInterrupt => FlagsMask::INTERRUPT,
            // `sahf` restores SF/ZF/AF/PF/CF; OF is not in the byte it loads.
            Self::LoadStatusFromReg => FlagsMask::SIGN
                .union(FlagsMask::ZERO)
                .union(FlagsMask::ADJUST)
                .union(FlagsMask::PARITY)
                .union(FlagsMask::CARRY),
            // AArch64 `setf8`/`setf16` (FEAT_FlagM) write N, Z **and V**, and
            // leave C alone. V is the point of the instruction: it is set from
            // `op<msb+1> EOR op<msb>`, which is how a narrow add's overflow is
            // recovered. Omitting it here reports a stale V as still live, so a
            // caller folds a `b.vs` against a flag `setf8` had overwritten.
            Self::SetNzFrom8 | Self::SetNzFrom16 => FlagsMask::SIGN
                .union(FlagsMask::ZERO)
                .union(FlagsMask::OVERFLOW),
            Self::RotateMaskInsert | Self::ConvertToFpFlags | Self::ConvertFromFpFlags => {
                FlagsMask::x86_status()
            }
            // Both read the flags and write a register instead.
            Self::StoreStatusToReg | Self::SetRegFromCarry => FlagsMask::from_bits(0),
        }
    }

    /// The native mnemonic this adjustment lifts from.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::InvertCarry => "cfinv",
            Self::RotateMaskInsert => "rmif",
            Self::SetNzFrom8 => "setf8",
            Self::SetNzFrom16 => "setf16",
            Self::ConvertToFpFlags => "axflag",
            Self::ConvertFromFpFlags => "xaflag",
            Self::ClearCarry => "clc",
            Self::SetCarry => "stc",
            Self::ClearDirection => "cld",
            Self::SetDirection => "std",
            Self::ClearAlignCheck => "clac",
            Self::SetAlignCheck => "stac",
            Self::ClearInterrupt => "cli",
            Self::SetInterrupt => "sti",
            Self::LoadStatusFromReg => "sahf",
            Self::StoreStatusToReg => "lahf",
            Self::SetRegFromCarry => "salc",
        }
    }
}

/// Boxed payload for [`SsaOp::VectorPredicateWhile`] - generates a predicate by
/// comparing a counter from a scalar base against a scalar limit (AArch64 SVE
/// `whilelt`/`whilele`/`whilege`/`whilegt` signed and `whilelo`/`whilels`/
/// `whilehs`/`whilehi` unsigned). The `inputs` are `[first, last]`. Pure.
///
/// The result holds one bit per element, bit `e` element `e`'s, for
/// `elements` elements. A `Lt` or `Le` comparison counts up from element `0`,
/// the counter `first + e`; a `Ge` or `Gt` one counts down from the last
/// element, the counter `first - (elements - 1 - e)`. Each element is set while
/// every counter so far satisfies the comparison with `last`, and clear from
/// the first that does not; the counters are integers of no width, so none
/// wraps.
///
/// [`SsaOp::VectorPredicateWhile`]: crate::ir::ops::def::SsaOp::VectorPredicateWhile
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorPredicateWhileData {
    /// The comparison relating each counter to the limit.
    pub kind: VectorCompareKind,
    /// `true` for the unsigned forms (`whilelo`/`whilels`/`whilehs`/`whilehi`).
    pub unsigned: bool,
    /// The elements of the vector the predicate governs.
    pub elements: u16,
    /// Explicit SSA outputs (the generated predicate).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs: the scalar base and limit.
    pub inputs: Vec<SsaVarId>,
}

impl VectorPredicateWhileData {
    /// Returns the predicate the comparison of `first`, the counter's base,
    /// with `last` generates: bit `e` set where element `e` is (see
    /// [`VectorPredicateWhileData`]). `None` for more elements than 128 bits
    /// hold, or a comparison no `while` makes.
    #[must_use]
    pub fn evaluate(&self, first: i128, last: i128) -> Option<u128> {
        let elements = u32::from(self.elements);
        if elements > 128 {
            return None;
        }
        let holds = |counter: i128| -> Option<bool> {
            Some(match self.kind {
                VectorCompareKind::Lt => counter < last,
                VectorCompareKind::Le => counter <= last,
                VectorCompareKind::Ge => counter >= last,
                VectorCompareKind::Gt => counter > last,
                _ => return None,
            })
        };
        let counting_up = matches!(self.kind, VectorCompareKind::Lt | VectorCompareKind::Le);
        let mut bits = 0u128;
        let mut counter = first;
        for step in 0..elements {
            if !holds(counter)? {
                break;
            }
            let element = if counting_up {
                step
            } else {
                elements.checked_sub(1)?.checked_sub(step)?
            };
            bits |= 1u128.checked_shl(element)?;
            counter = if counting_up {
                counter.checked_add(1)?
            } else {
                counter.checked_sub(1)?
            };
        }
        Some(bits)
    }
}

/// Lane element class of a vector operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorElementKind {
    /// Integer lanes.
    Integer,
    /// Floating-point lanes of the IEEE format of their width.
    Float,
    /// Element class not recovered by the decoder.
    #[default]
    Unknown,
    /// bfloat16 lanes, 16 bits wide like binary16's.
    BFloat16,
    /// Floating-point lanes of the IEEE format of their width whose rounding
    /// and special values the implementation decides: 3DNow!'s singles, which
    /// round in one mode the processor chooses (to nearest or toward zero)
    /// whatever MXCSR says, and whose NaN and infinity results are undefined.
    /// Only a result exact in every mode, of numbers, is theirs to know
    /// ([`Self::opens_specials`]).
    FloatOpen,
}

impl VectorElementKind {
    /// The float format of lanes `bits` wide of this kind: bfloat16 where
    /// the kind says, the IEEE format of the width otherwise.
    #[must_use]
    pub const fn float_format(self, bits: u32) -> Option<FloatFormat> {
        match self {
            Self::BFloat16 if bits == 16 => Some(FloatFormat::BFloat16),
            Self::BFloat16 => None,
            _ => FloatFormat::of_width(bits),
        }
    }

    /// Whether the lanes hold floats, of any format.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float | Self::BFloat16 | Self::FloatOpen)
    }

    /// Whether the implementation decides what the lanes' NaNs and
    /// infinities compute ([`Self::FloatOpen`]): a fold of them declines an
    /// operand or a result that is not a number.
    #[must_use]
    pub const fn opens_specials(self) -> bool {
        matches!(self, Self::FloatOpen)
    }
}

/// Lane element descriptor of a vector operation.
///
/// Carried on the SIMD ops so a host projecting the IR back out (e.g. the Visus
/// cleaned-IR lower leg) can render `paddd` vs `paddb` vs `addps` distinctly. The
/// descriptor travels with the op through normalization, so it is not lost the
/// way a side-table keyed by value id would be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorElement {
    /// Lane element class (integer / floating-point / unknown).
    pub kind: VectorElementKind,
    /// Width of one lane in bits (`0` when not recovered).
    pub bits: u32,
    /// `true` when the op operates on a single (scalar) lane rather than a
    /// packed vector - e.g. `addss`/`addsd` vs `addps`/`addpd`.
    pub scalar: bool,
    /// Whether integer lanes are read as signed (`Some(true)`) or unsigned
    /// (`Some(false)`), for an operation whose result depends on it: the
    /// saturating additions and subtractions, minimum and maximum, the
    /// average, the absolute difference, division and remainder, the high
    /// multiply, and the shift right, arithmetic where signed (NEON
    /// `vqadd.s8` and `vqadd.u8`, x86 `pminsb` and `pminub`). `None` where the
    /// operation does not depend on it, its lanes are no integers, or the
    /// producer does not state it.
    pub signed: Option<bool>,
    /// The rounding float lanes take where the instruction states one in
    /// place of the environment's mode (RISC-V `fadd.s fa0, fa1, fa2, rtz`,
    /// AVX-512 `vaddps zmm0, zmm1, zmm2, {rz-sae}`), for an operation whose
    /// result depends on it: addition, subtraction, multiplication, division,
    /// the fused multiply-add, the square root, the round to an integral
    /// value and a reduction's sum and product. `None` where the operation
    /// rounds by the environment's mode, does not round, or its lanes are no
    /// floats.
    pub rounding: Option<FloatRounding>,
}

impl VectorElement {
    /// Returns packed integer lanes `bits` wide (`0`: the operands' own),
    /// read signed or unsigned as `signed` states.
    #[must_use]
    pub const fn integer(bits: u32, signed: Option<bool>) -> Self {
        Self {
            kind: VectorElementKind::Integer,
            bits,
            scalar: false,
            signed,
            rounding: None,
        }
    }

    /// Returns packed float lanes `bits` wide.
    #[must_use]
    pub const fn float(bits: u32) -> Self {
        Self {
            kind: VectorElementKind::Float,
            bits,
            scalar: false,
            signed: None,
            rounding: None,
        }
    }

    /// Returns the element as a scalar of its width: the value of a scalar
    /// computation, or the lowest lane of a scalar form.
    #[must_use]
    pub const fn as_scalar(self) -> Self {
        Self {
            scalar: true,
            ..self
        }
    }

    /// Returns the element rounding its float lanes by `rounding`.
    #[must_use]
    pub const fn rounded(self, rounding: FloatRounding) -> Self {
        Self {
            rounding: Some(rounding),
            ..self
        }
    }

    /// Returns the element of an exact operation on the same lanes (a
    /// negation beside the rounded product it feeds), which states no
    /// rounding.
    #[must_use]
    pub const fn exact(self) -> Self {
        Self {
            rounding: None,
            ..self
        }
    }

    /// The rounding the lanes take: the one the element states, else the
    /// environment's ([`crate::target::Target::environment_rounding`]), `None`
    /// for a mode the program sets at run time. Lanes whose rounding the
    /// implementation decides ([`VectorElementKind::FloatOpen`]) take none of
    /// the environment's, so only a result exact in every mode folds.
    #[must_use]
    pub fn rounding_in(self, environment: Option<FloatRounding>) -> Option<FloatRounding> {
        if self.kind.opens_specials() {
            return self.rounding;
        }
        self.rounding.or(environment)
    }

    /// Returns the element as an operation that does or does not depend on
    /// the lanes' signedness reads it: unchanged where it does, stating no
    /// signedness where it does not or the lanes are floats.
    #[must_use]
    pub const fn read_by(self, reads_signedness: bool) -> Self {
        if reads_signedness && !self.kind.is_float() {
            self
        } else {
            Self {
                signed: None,
                ..self
            }
        }
    }

    /// Returns the display suffix of the lanes' stated signedness: `.s`, `.u`,
    /// or nothing where none is stated.
    pub fn signedness_suffix(&self) -> &'static str {
        match self.signed {
            Some(true) => ".s",
            Some(false) => ".u",
            None => "",
        }
    }

    /// Returns the display suffix of the lanes' stated rounding (`.tozero`),
    /// or nothing where none is stated.
    pub fn rounding_suffix(&self) -> String {
        self.rounding
            .map(|rounding| format!(".{rounding}"))
            .unwrap_or_default()
    }
}

/// Vector unary operation kind.
///
/// Each variant's discriminant is its stable codec number, the `u16` it
/// converts to and from: append only, never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum VectorUnaryKind {
    /// Lane-wise integer or floating negation.
    Neg = 3,
    /// Lane-wise bitwise not.
    Not = 4,
    /// Lane-wise population count.
    Popcount = 0,
    /// Lane-wise absolute value.
    Abs = 2,
    /// Horizontal lane sum.
    HorizontalAdd = 6,
    /// Lane-wise floating-point square root.
    Sqrt = 1,
    /// Lane-wise floating-point round to integral value in the current
    /// rounding mode, or the direction the element states (result stays
    /// floating-point; C `nearbyint`), without signalling inexact (IEEE 754
    /// roundToIntegral): ARM `vrintr`, AArch64 `frinti`, RISC-V `fround`.
    Round = 5,
    /// Lane-wise floating-point reciprocal approximation (`1/x`): x86
    /// `vrcp14*`/`vrcp28*`/`vrcpps`, AArch64 `frecpe`.
    Reciprocal = 7,
    /// Lane-wise floating-point reciprocal-square-root approximation
    /// (`1/sqrt(x)`): x86 `vrsqrt14*`/`vrsqrt28*`/`vrsqrtps`, AArch64 `frsqrte`.
    ReciprocalSqrt = 8,
    /// Lane-wise floating-point exponent extraction (`vgetexp*`): returns the
    /// unbiased exponent as a floating-point value.
    GetExponent = 9,
    /// Lane-wise identity (pass-through). Used as the operation of a predicated
    /// merge / blend (`vpblendm*`/`vblendm*`), where each active lane copies the
    /// source and inactive lanes take the passthrough.
    Identity = 13,
    /// Lane-wise count of leading zero bits (`vplzcntd`/`vplzcntq`).
    LeadingZeros = 14,
    /// Lane-wise floating-point fractional part `x - trunc(x)` (AMD XOP
    /// `vfrczps`/`vfrczpd`/`vfrczss`/`vfrczsd`).
    Fraction = 15,
    /// Lane-wise conflict detection (`vpconflictd`/`vpconflictq`): each lane
    /// receives a bitmask of the preceding lanes equal to it.
    Conflict = 16,
    /// Lane-wise base-2 exponential `2^x` (`vexp2*`).
    Exp2 = 17,
    /// Lane-wise base-2 logarithm `log2(x)` (`vlog2*`).
    Log2 = 18,
    /// Lane-wise count of leading sign bits (`vcls`).
    LeadingSignBits = 19,
    /// Lane-wise count of leading one bits (MSA `nloc`).
    LeadingOnes = 20,
    /// Round toward zero to an integral value (truncate; VB6 `Fix`).
    Truncate = 21,
    /// Round down to an integral value (floor, toward −∞; VB6 `Int`).
    Floor = 22,
    /// Round up to an integral value (ceiling, toward +∞; C `ceil`): AArch64
    /// `frintp`, ARM `vrintp`.
    Ceil = 25,
    /// Round to the nearest integral value, ties to even, whatever the current
    /// mode (C `roundeven`): AArch64 `frintn`, ARM `vrintn`.
    RoundNearestEven = 26,
    /// Round to the nearest integral value, ties away from zero (C `round`):
    /// AArch64 `frinta`, ARM `vrinta`.
    RoundNearestAway = 27,
    /// Sign of the value (`-1` / `0` / `+1`; VB6 `Sgn`).
    Sign = 23,
    /// Lane-wise bit reversal within each element (AArch64 NEON/SVE `rbit`).
    BitReverse = 24,
    /// Lane-wise count of trailing zero bits; a zero lane returns its bit width.
    TrailingZeros = 28,
    /// Reverses byte order within each integer element.
    ByteSwap = 29,
    /// Reverses bits independently within each byte, preserving byte order.
    BitReverseIn8 = 30,
    /// Lane-wise floating-point round to integral value as [`Self::Round`]
    /// rounds, signalling inexact where the result differs from the source
    /// (IEEE 754 roundToIntegralExact; C `rint`): x87 `frndint`, ARM
    /// `vrintx`, AArch64 `frintx`, RISC-V `froundnx`.
    RoundExact = 31,
    /// Lane-wise floating-point round to an integral value a signed 32-bit
    /// integer holds, as [`Self::RoundExact`] rounds: a result out of that
    /// range, a NaN or an infinity is `-2^31`, signalling invalid (AArch64
    /// `frint32x`, `frint32z`).
    RoundInt32 = 32,
    /// [`Self::RoundInt32`] for a signed 64-bit integer, out of range `-2^63`
    /// (AArch64 `frint64x`, `frint64z`).
    RoundInt64 = 33,
    /// Lane-wise floating negation computed as an arithmetic operation: a
    /// number's sign flipped as [`Self::Neg`] flips it, and a NaN propagated
    /// as arithmetic propagates one, a signalling NaN raising invalid and
    /// giving the default NaN, a quiet one unchanged (legacy MIPS `neg.fmt`,
    /// where `FCSR.ABS2008` is clear).
    NegArithmetic = 34,
    /// Lane-wise floating absolute value computed as an arithmetic
    /// operation: a number's sign cleared as [`Self::Abs`] clears it, and a
    /// NaN propagated as [`Self::NegArithmetic`] propagates one (legacy MIPS
    /// `abs.fmt`).
    AbsArithmetic = 35,
}

impl VectorUnaryKind {
    /// The arithmetic whose exceptions a float lane of this operation
    /// raises ([`crate::ir::float::FloatRules::raises`]): the square root and
    /// the exact round; `None` for one whose flags are not stated here.
    #[must_use]
    pub const fn raising(self) -> Option<RaisingArithmetic> {
        match self {
            Self::Sqrt => Some(RaisingArithmetic::Sqrt),
            Self::RoundExact => Some(RaisingArithmetic::RoundExact),
            _ => None,
        }
    }

    /// Whether each bit of the result is computed of the operand's same bit
    /// alone, whatever lanes it is read in: the bitwise not and the identity.
    #[must_use]
    pub const fn is_bitwise(self) -> bool {
        matches!(self, Self::Not | Self::Identity)
    }

    /// Whether the operation's result on integer lanes depends on reading them
    /// signed or unsigned ([`VectorElement::signed`]): the sign, which an
    /// unsigned lane never has negative (VB6's `Sgn` of a `Byte`).
    #[must_use]
    pub const fn reads_signedness(self) -> bool {
        matches!(self, Self::Sign)
    }

    /// Whether the operation's result on float lanes depends on the rounding
    /// ([`VectorElement::rounding`]): the square root and the rounds to an
    /// integral value in the environment's mode.
    #[must_use]
    pub const fn reads_rounding(self) -> bool {
        matches!(
            self,
            Self::Sqrt | Self::Round | Self::RoundExact | Self::RoundInt32 | Self::RoundInt64
        )
    }

    /// The width of the signed integer a round to an integral value bounds
    /// its result by ([`Self::RoundInt32`], [`Self::RoundInt64`]).
    #[must_use]
    pub const fn integer_bound(self) -> Option<u32> {
        match self {
            Self::RoundInt32 => Some(32),
            Self::RoundInt64 => Some(64),
            _ => None,
        }
    }
}

/// Vector binary operation kind.
///
/// Each variant's discriminant is its stable codec number, the `u16` it
/// converts to and from: append only, never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum VectorBinaryKind {
    /// Lane-wise addition.
    Add = 0,
    /// Lane-wise subtraction.
    Sub = 1,
    /// Lane-wise multiplication.
    Mul = 2,
    /// Lane-wise division, of signed or unsigned integer lanes as the element
    /// [states](VectorElement::signed).
    Div = 3,
    /// Lane-wise minimum. For floats a NaN operand gives NaN (IEEE 754-2019
    /// `minimum`: AArch64 `fmin`, RISC-V `fminm`); [`Self::MinNum`] is the
    /// minimum that favours the number, [`Self::MinLess`] the select x86
    /// computes. Integer lanes are compared signed or unsigned as the element
    /// [states](VectorElement::signed).
    Min = 4,
    /// Lane-wise maximum, NaN as [`Self::Min`] treats it.
    Max = 5,
    /// Lane-wise bitwise and.
    And = 6,
    /// Lane-wise bitwise or.
    Or = 7,
    /// Lane-wise bitwise xor.
    Xor = 8,
    /// Lane-wise shift left by a count. A count at or past the lane width
    /// shifts every bit out.
    Shl = 9,
    /// Lane-wise shift right by a count: arithmetic where the element is
    /// [signed](VectorElement::signed), logical otherwise. A count at or past
    /// the lane width shifts every bit out, leaving zero, or the sign in every
    /// bit of an arithmetic shift (NEON `sshr` and `ushr` by the lane width,
    /// x86 `psraw` and `psrlw` by 16 or more).
    Shr = 10,
    /// Lane-wise bitwise and-not (`a & ~b`).
    AndNot = 11,
    /// Lane-wise multiply keeping the high half of the product of signed or
    /// unsigned lanes as the element [states](VectorElement::signed) (x86
    /// `pmulhw` and `pmulhuw`).
    MulHigh = 12,
    /// Lane-wise signed multiply keeping the rounded high half of the product:
    /// `(a * b + 2^(n - 1)) >> n` for `n`-bit lanes (3DNow! `pmulhrw`, A32
    /// `smmulr`). The doubling form is [`Self::MulHighDoubleRound`].
    MulHighRound = 21,
    /// Lane-wise rounding average `(a + b + 1) >> 1`, computed without
    /// overflow on signed or unsigned lanes as the element
    /// [states](VectorElement::signed).
    Avg = 13,
    /// Lane-wise absolute difference `|a - b|` of signed or unsigned lanes as
    /// the element [states](VectorElement::signed).
    AbsDiff = 14,
    /// Lane-wise saturating addition, clamped to the lanes' signed or
    /// unsigned range as the element [states](VectorElement::signed).
    SatAdd = 15,
    /// Lane-wise saturating subtraction, clamped as [`Self::SatAdd`] is.
    SatSub = 16,
    /// Lane-wise floating-point scale by a float exponent: `a * 2^floor(b)`
    /// (x86 `vscalef*`). A scale by an integer exponent is
    /// [`Self::ScaleInteger`].
    Scale = 17,
    /// Lane-wise bitwise rotate left (x86 `vprol*`, XOP `vprot*`).
    Rol = 19,
    /// Lane-wise bitwise rotate right (x86 `vpror*`).
    Ror = 20,
    /// Lane-wise variable logical shift by a signed count, the sign selecting
    /// direction (AMD XOP `vpshl*`: positive shifts left, negative right).
    VariableShiftLogical = 22,
    /// Lane-wise variable arithmetic shift by a signed count (AMD XOP `vpsha*`).
    VariableShiftArithmetic = 23,
    /// Lane-wise apply-sign: negate / zero / keep the first operand according to
    /// the sign of the second (x86 `psign*`).
    ApplySign = 24,
    /// 3DNow! reciprocal Newton-Raphson refinement step 1 (`pfrcpit1`).
    ReciprocalIter1 = 25,
    /// 3DNow! reciprocal Newton-Raphson refinement step 2 (`pfrcpit2`).
    ReciprocalIter2 = 26,
    /// 3DNow! reciprocal-square-root Newton-Raphson refinement step 1
    /// (`pfrsqit1`).
    ReciprocalSqrtIter1 = 27,
    /// Lane-wise maximum by magnitude: per lane, keep the operand with the
    /// greater absolute value (Cyrix EMMI `pmagw`, MSA `max_a`, `fmax_a`).
    /// Integer lanes compare their absolute values unsigned, so the minimum
    /// integer has the greatest. A tie and a NaN follow the architecture: MSA
    /// keeps the second source on an integer tie, the greater operand on a
    /// float tie, and the number when one operand is a quiet NaN (IEEE
    /// 754-2008 `maxNumMag`).
    MaxMagnitude = 28,
    /// Lane-wise negated multiply: `-(a * b)` (AArch64 scalar-FP `fnmul`).
    MulNegate = 29,
    /// Bitwise OR with complemented second operand: `a | ~b` (NEON `vorn`).
    OrNot = 30,
    /// Lane-wise add of absolute values: `|a| + |b|` (MSA `add_a`).
    AddAbs = 31,
    /// Lane-wise integer remainder: `a % b` (MSA `mod_s`/`mod_u`), of signed
    /// or unsigned lanes as the element [states](VectorElement::signed), as
    /// [`Self::Div`] divides.
    Rem = 32,
    /// Lane-wise bitwise NOR: `~(a | b)` (MSA `nor.v`).
    Nor = 33,
    /// Lane-wise signed multiply keeping the rounded high half of the doubled
    /// product, without saturation: `(2 * a * b + 2^(n - 1)) >> n` for `n`-bit
    /// lanes (`pmulhrsw`). The minimum integer times itself wraps to the
    /// minimum.
    MulHighDoubleRound = 34,
    /// Lane-wise signed multiply keeping the high half of the doubled product,
    /// saturated: `sat((2 * a * b) >> n)` (AArch64 `sqdmulh`, A32 `vqdmulh`,
    /// MSA `mul_q`). Only the minimum integer times itself saturates, to the
    /// maximum.
    MulHighDoubleSat = 35,
    /// Lane-wise signed multiply keeping the rounded high half of the doubled
    /// product, saturated: `sat((2 * a * b + 2^(n - 1)) >> n)` (AArch64
    /// `sqrdmulh`, A32 `vqrdmulh`, MSA `mulr_q`).
    MulHighDoubleSatRound = 36,
    /// Lane-wise floating-point minimum that returns the number when one
    /// operand is a quiet NaN (IEEE 754-2008 `minNum`): AArch64 `fminnm`, ARM
    /// `vminnm`, RISC-V `fmin`, MSA `fmin`.
    MinNum = 37,
    /// Lane-wise floating-point maximum that returns the number when one
    /// operand is a quiet NaN (`maxNum`): AArch64 `fmaxnm`, ARM `vmaxnm`,
    /// RISC-V `fmax`, MSA `fmax`.
    MaxNum = 38,
    /// Lane-wise minimum by magnitude: per lane, keep the operand with the
    /// smaller absolute value (MSA `min_a`, `fmin_a`), ties and NaN as
    /// [`Self::MaxMagnitude`] treats them (the float tie keeps the smaller
    /// operand; IEEE 754-2008 `minNumMag`).
    MinMagnitude = 39,
    /// Lane-wise float minimum as a select: the first operand where it is less
    /// than the second, and the second otherwise (`a < b ? a : b`), so a NaN
    /// in either operand or two equal zeros give the second operand, whole
    /// (x86 `minss`, `minpd`, `vminph`). The compare signals: a quiet NaN raises
    /// invalid.
    MinLess = 40,
    /// Lane-wise float maximum as a select, `a > b ? a : b`, NaN and zeros as
    /// [`Self::MinLess`] treats them (x86 `maxss`, `maxpd`, `vmaxph`).
    MaxGreater = 41,
    /// Lane-wise floating-point scale by an integer exponent, C's `ldexp`:
    /// each float lane of `a` times two to the power of the signed integer
    /// lane of `b` beside it (SVE `fscale`, `bfscale`, Xeon Phi `vscaleps`).
    ScaleInteger = 42,
    /// Lane-wise floating-point minimum that returns the number beside any
    /// NaN, quiet or signalling, and orders negative zero below positive zero
    /// (IEEE 754-2019 `minimumNumber`): AArch64 FEAT_FPv2 `fmininm`.
    MinimumNumber = 43,
    /// Lane-wise floating-point maximum, NaN and zeros as
    /// [`Self::MinimumNumber`] treats them (IEEE 754-2019 `maximumNumber`):
    /// AArch64 `fmaxinm`.
    MaximumNumber = 44,
    /// Lane-wise floating-point minimum by magnitude: the operand of the
    /// smaller absolute value, of two of equal magnitude the lesser (negative
    /// zero below positive), and NaN for a NaN operand (IEEE 754-2019
    /// `minimumMagnitude`): AArch64 `fminmg`.
    MinimumMagnitude = 45,
    /// Lane-wise floating-point maximum by magnitude, ties the greater and
    /// NaN as [`Self::MinimumMagnitude`] (IEEE 754-2019 `maximumMagnitude`):
    /// AArch64 `fmaxmg`.
    MaximumMagnitude = 46,
    /// Lane-wise floating-point minimum by magnitude that returns the number
    /// beside any NaN, ties as [`Self::MinimumMagnitude`] (IEEE 754-2019
    /// `minimumMagnitudeNumber`): AArch64 `fminmginm`.
    MinimumMagnitudeNumber = 47,
    /// Lane-wise floating-point maximum by magnitude that returns the number
    /// beside any NaN, ties as [`Self::MaximumMagnitude`] (IEEE 754-2019
    /// `maximumMagnitudeNumber`): AArch64 `fmaxmginm`.
    MaximumMagnitudeNumber = 48,
    /// Lane-wise floating-point absolute minimum: the lesser of the operands'
    /// absolute values, the second's on a tie, and beside a NaN the NaN the
    /// architecture's processing of the operands as they are gives, as
    /// [`Self::Min`] does (AArch64 `famin`, `FPAbsMin`).
    MinAbs = 49,
    /// Lane-wise floating-point absolute maximum: the greater of the
    /// operands' absolute values, ties and NaNs as [`Self::MinAbs`] treats
    /// them (AArch64 `famax`, `FPAbsMax`).
    MaxAbs = 50,
}

impl VectorBinaryKind {
    /// The arithmetic whose exceptions a float lane of this operation
    /// raises ([`crate::ir::float::FloatRules::raises`]); `None` for one whose
    /// flags are not stated here.
    #[must_use]
    pub const fn raising(self) -> Option<RaisingArithmetic> {
        Some(RaisingArithmetic::Binary(match self {
            Self::Add => FloatBinary::Add,
            Self::Sub => FloatBinary::Subtract,
            Self::Mul => FloatBinary::Multiply,
            Self::Div => FloatBinary::Divide,
            // Every minimum and maximum raises invalid for a signalling NaN
            // operand and nothing else, as `minimumNumber` does.
            Self::MinNum
            | Self::MinimumNumber
            | Self::MinimumMagnitude
            | Self::MinimumMagnitudeNumber
            | Self::MinAbs => FloatBinary::MinimumNumber,
            Self::MaxNum
            | Self::MaximumNumber
            | Self::MaximumMagnitude
            | Self::MaximumMagnitudeNumber
            | Self::MaxAbs => FloatBinary::MaximumNumber,
            // The select compares, and its compare signals.
            Self::MinLess | Self::MaxGreater => {
                return Some(RaisingArithmetic::Compare { signaling: true });
            }
            _ => return None,
        }))
    }

    /// Whether each bit of the result is computed of the operands' same bit
    /// alone, whatever lanes they are read in (x86 `andps` and `pand` are the
    /// same operation): `and`, `or`, `xor` and their complemented forms.
    #[must_use]
    pub const fn is_bitwise(self) -> bool {
        matches!(
            self,
            Self::And | Self::Or | Self::Xor | Self::AndNot | Self::OrNot | Self::Nor
        )
    }

    /// Whether the operation's result on integer lanes depends on reading them
    /// signed or unsigned ([`VectorElement::signed`]).
    #[must_use]
    pub const fn reads_signedness(self) -> bool {
        matches!(
            self,
            Self::Div
                | Self::Rem
                | Self::Min
                | Self::Max
                | Self::Shr
                | Self::MulHigh
                | Self::MulHighRound
                | Self::Avg
                | Self::AbsDiff
                | Self::SatAdd
                | Self::SatSub
        )
    }

    /// Whether the operation's result on float lanes depends on the rounding
    /// ([`VectorElement::rounding`]): addition, subtraction, multiplication,
    /// division and the scales by a power of two.
    #[must_use]
    pub const fn reads_rounding(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Scale | Self::ScaleInteger
        )
    }

    /// The operand this minimum or maximum gives of the floats `a` and `b`,
    /// where every architecture computing it gives the same: `None` for an
    /// operation that is no minimum or maximum, and where the result is the
    /// architecture's choice.
    ///
    /// [`Self::MinLess`] and [`Self::MaxGreater`] are selects, fixed for every
    /// input. [`Self::Min`] and [`Self::Max`] give a NaN for a NaN operand,
    /// whose payload is the architecture's; [`Self::MinNum`] and
    /// [`Self::MaxNum`] give the number beside a quiet NaN and leave a
    /// signalling NaN, two NaNs and the sign of two equal zeros to the
    /// architecture; the magnitude forms give the operand of the smaller or
    /// larger absolute value, the number beside a quiet NaN, and leave a tie
    /// to the architecture. The IEEE 754-2019 forms ([`Self::MinimumNumber`]
    /// and its siblings) state ties and zeros: of two equal operands the
    /// lesser or greater, negative zero below positive; the number forms give
    /// the number beside any NaN, and every form leaves a NaN result to the
    /// architecture. The absolute forms ([`Self::MinAbs`], [`Self::MaxAbs`])
    /// give the lesser or greater absolute value itself, the second's on a
    /// tie, and leave a NaN to the architecture.
    #[must_use]
    pub fn float_extreme(self, a: FloatBits, b: FloatBits) -> Option<FloatBits> {
        let order = |x: FloatBits, y: FloatBits| x.compare(y);
        let pick = |ordering: Option<Ordering>, wanted: Ordering| match ordering? {
            ordering if ordering == wanted => Some(a),
            Ordering::Equal => (a == b).then_some(a),
            _ => Some(b),
        };
        // Beside a quiet NaN the number forms give the number.
        let number = || match (a.is_nan(), b.is_nan()) {
            (true, false) if !a.is_signaling() => Some(Some(b)),
            (false, true) if !b.is_signaling() => Some(Some(a)),
            (false, false) => None,
            _ => Some(None),
        };
        // Of two operands that compare equal (one value, or a value and its
        // negation) the lesser or the greater, negative below positive.
        let signed = |least: bool| {
            if a == b || a.is_negative() == least {
                a
            } else {
                b
            }
        };
        let by = |ordering: Ordering, least: bool| match ordering {
            Ordering::Equal => signed(least),
            Ordering::Less => {
                if least {
                    a
                } else {
                    b
                }
            }
            Ordering::Greater => {
                if least {
                    b
                } else {
                    a
                }
            }
        };
        // Beside any NaN the IEEE 754-2019 number forms give the number.
        let any_number = || match (a.is_nan(), b.is_nan()) {
            (true, false) => Some(Some(b)),
            (false, true) => Some(Some(a)),
            (false, false) => None,
            (true, true) => Some(None),
        };
        let magnitude = |least: bool| {
            (!a.is_nan() && !b.is_nan())
                .then(|| order(a.abs(), b.abs()).map(|ordering| by(ordering, least)))
                .flatten()
        };
        match self {
            Self::MinimumNumber | Self::MaximumNumber => any_number().unwrap_or_else(|| {
                order(a, b).map(|ordering| by(ordering, self == Self::MinimumNumber))
            }),
            Self::MinimumMagnitude | Self::MaximumMagnitude => {
                magnitude(self == Self::MinimumMagnitude)
            }
            Self::MinimumMagnitudeNumber | Self::MaximumMagnitudeNumber => {
                any_number().unwrap_or_else(|| magnitude(self == Self::MinimumMagnitudeNumber))
            }
            // Of the absolute values the lesser or the greater, the second
            // on a tie; a NaN's processing is the architecture's.
            Self::MinAbs | Self::MaxAbs => (!a.is_nan() && !b.is_nan())
                .then(|| {
                    let (first, second) = (a.abs(), b.abs());
                    let wanted = if self == Self::MinAbs {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                    Some(if order(first, second)? == wanted {
                        first
                    } else {
                        second
                    })
                })
                .flatten(),
            Self::MinLess => Some(if order(a, b) == Some(Ordering::Less) {
                a
            } else {
                b
            }),
            Self::MaxGreater => Some(if order(a, b) == Some(Ordering::Greater) {
                a
            } else {
                b
            }),
            Self::Min => pick(order(a, b), Ordering::Less),
            Self::Max => pick(order(a, b), Ordering::Greater),
            Self::MinNum => number().unwrap_or_else(|| pick(order(a, b), Ordering::Less)),
            Self::MaxNum => number().unwrap_or_else(|| pick(order(a, b), Ordering::Greater)),
            Self::MinMagnitude | Self::MaxMagnitude => number().unwrap_or_else(|| {
                let wanted = if self == Self::MinMagnitude {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
                match order(a.abs(), b.abs())? {
                    Ordering::Equal => None,
                    ordering => Some(if ordering == wanted { a } else { b }),
                }
            }),
            _ => None,
        }
    }
}

/// What a float lane operation an 8-bit control immediate selects computes
/// ([`Kernel::FloatControlled`]).
///
/// Each variant's discriminant is its stable codec number, the `u16` it
/// converts to and from: append only, never renumber.
///
/// [`Kernel::FloatControlled`]: crate::ir::ops::Kernel::FloatControlled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum ControlledKind {
    /// Rounds each lane to the number of fraction bits the control's upper
    /// four bits name, by the rounding its low bits name, signalling inexact
    /// unless bit 3 suppresses it (x86 `vrndscale*`).
    RoundScale = 0,
    /// Each lane less its round as [`Self::RoundScale`] rounds it (x86
    /// `vreduce*`).
    Reduce = 1,
    /// Each lane's significand, normalized to the interval and given the sign
    /// the control names (x86 `vgetmant*`).
    GetMantissa = 2,
    /// The minimum, maximum, or the one of greater or smaller magnitude, of
    /// each lane pair, with the sign the control names (x86 `vrange*`).
    Range = 3,
    /// Each lane of the first source replaced by the token the table of the
    /// third selects for the class of the second, the control naming which
    /// classes raise invalid and divide by zero (x86 `vfixupimm*`).
    Fixup = 4,
    /// A mask, a bit per lane set where the lane is of a class the control
    /// names (x86 `vfpclass*`).
    Classify = 5,
    /// Rounds each lane to the fixed-point fraction width the control's bits
    /// 6..4 index (0, 4, 5, 8, 16, 24, 31 or 32 bits), in the direction its
    /// low two bits name (to nearest even, down, up, toward zero), signalling
    /// inexact: `round(x * 2^n) / 2^n` (Xeon Phi `vrndfxpnt*`).
    RoundFixedPoint = 6,
}

impl ControlledKind {
    /// Returns the inputs of the operation over packed lanes (`scalar`
    /// false) or of its scalar form, which also reads the source whose upper
    /// lanes it keeps: one for a class test and a fixed-point round, two for
    /// a range, three for a fix-up (the destination, the source and the
    /// table), and one packed or two scalar for a round, a remainder and a
    /// mantissa.
    #[must_use]
    pub const fn input_count(self, scalar: bool) -> usize {
        match self {
            Self::Classify | Self::RoundFixedPoint => 1,
            Self::Range => 2,
            Self::Fixup => 3,
            Self::RoundScale | Self::Reduce | Self::GetMantissa => {
                1usize.saturating_add(scalar as usize)
            }
        }
    }

    /// The operation's IR name: `round.scale`, `round.remainder`,
    /// `mantissa`, `range`, `fixup`, `classify.test` and `round.fixedpoint`;
    /// a round or a remainder whose control does not suppress the inexact
    /// signal is `.exact` ([`Kernel::name`](crate::ir::ops::Kernel::name)).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RoundScale => "round.scale",
            Self::Reduce => "round.remainder",
            Self::GetMantissa => "mantissa",
            Self::Range => "range",
            Self::Fixup => "fixup",
            Self::Classify => "classify.test",
            Self::RoundFixedPoint => "round.fixedpoint",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::RoundScale => "controlled.roundscale",
            Self::Reduce => "controlled.reduce",
            Self::GetMantissa => "controlled.getmant",
            Self::Range => "controlled.range",
            Self::Fixup => "controlled.fixup",
            Self::Classify => "controlled.fpclass",
            Self::RoundFixedPoint => "controlled.rndfxpnt",
        }
    }

    /// The lanes this operation computes of `inputs` under `control`, lanes
    /// binary16, binary32 or binary64 as `element` states, as the Intel SDM
    /// defines each (`RoundToInteger`, `ReduceArgument`, `GetMant`, `Range`,
    /// `FIXUPIMM`, `CheckFPClass`; Xeon Phi's `vrndfxpnt` from 327364-001):
    ///
    /// - [`Self::RoundScale`], [`Self::Reduce`], [`Self::GetMantissa`],
    ///   [`Self::RoundFixedPoint`] and [`Self::Classify`] of `[source]`;
    /// - [`Self::Range`] of `[first, second]`;
    /// - [`Self::Fixup`] of `[destination, source, table]`.
    ///
    /// A scalar form (`element.scalar`) computes lane zero and keeps the
    /// other lanes of the input before its data (`vrndscalesd xmm1, xmm2,
    /// xmm3` reads `[xmm2, xmm3]` and keeps `xmm2`'s), and of its source for a
    /// fix-up (`vfixupimmsd xmm1, xmm2, xmm3` keeps `xmm2`'s). A class test
    /// gives each lane all ones where its source is of a class the control
    /// names.
    ///
    /// A rounding the control leaves to `MXCSR` (its bit 2) rounds by
    /// `environment` and folds, where that is unknown, only where every mode
    /// agrees. A subnormal operand folds nothing, as `MXCSR.DAZ` decides it,
    /// and neither does a reduction to a subnormal, which `MXCSR.FZ` decides.
    ///
    /// # Returns
    ///
    /// The result's lanes, or `None` for another format, inputs of another
    /// count or shape, and a lane that does not fold.
    #[must_use]
    pub fn evaluate_lanes(
        self,
        control: u8,
        inputs: &[VectorConst],
        element: VectorElement,
        environment: Option<FloatRounding>,
    ) -> Option<VectorConst> {
        let format = match (element.kind, element.bits) {
            (VectorElementKind::Float, 16) => FloatFormat::Half,
            (VectorElementKind::Float, 32) => FloatFormat::Single,
            (VectorElementKind::Float, 64) => FloatFormat::Double,
            _ => return None,
        };
        let bits = element.bits;
        let inputs = inputs
            .iter()
            .map(|input| input.relaned(bits))
            .collect::<Option<Vec<_>>>()?;
        let data = match self {
            Self::Range => 2,
            Self::Fixup => 3,
            _ => 1,
        };
        // The data inputs, and for a scalar form the input whose other lanes
        // the result keeps.
        let (kept, data_inputs) = match (element.scalar, self) {
            (false, _) if inputs.len() == data => (None, inputs.as_slice()),
            (true, Self::Fixup) if inputs.len() == data => (inputs.get(1), inputs.as_slice()),
            (true, Self::Classify) if inputs.len() == 1 => (None, inputs.as_slice()),
            (true, Self::Range) if inputs.len() == data => (inputs.first(), inputs.as_slice()),
            (true, _) if inputs.len() == data.checked_add(1)? => (inputs.first(), inputs.get(1..)?),
            _ => return None,
        };
        let lanes = data_inputs.first()?.lanes();
        if data_inputs.iter().any(|input| input.lanes() != lanes) {
            return None;
        }
        let computed = if element.scalar { 1 } else { lanes };
        let values = (0..computed)
            .map(|lane| {
                let operands: Vec<u128> =
                    data_inputs.iter().map(|input| input.lane(lane)).collect();
                self.lane(&operands, control, format, environment)
            })
            .collect::<Option<Vec<_>>>()?;
        match kept {
            Some(kept) => kept.with_lane(0, *values.first()?),
            // A scalar class test sets lane zero's bit and no other.
            None if element.scalar => VectorConst::from_lanes(
                bits,
                values.iter().copied().chain(std::iter::repeat_n(
                    0,
                    usize::try_from(lanes.checked_sub(1)?).ok()?,
                )),
            ),
            None => VectorConst::from_lanes(bits, values),
        }
    }

    /// One lane of the operation, of its operands' bits.
    fn lane(
        self,
        operands: &[u128],
        control: u8,
        format: FloatFormat,
        environment: Option<FloatRounding>,
    ) -> Option<u128> {
        let float = |value: u128| FloatBits::new(format, value);
        let width = format.bits();
        let sign_bit = 1u128.checked_shl(width.checked_sub(1)?)?;
        let negative = |value: u128| value & sign_bit != 0;
        // The rounding the control names, or the environment's.
        let rounding = match (control & 4 != 0, control & 3) {
            (true, _) => environment,
            (false, 0) => Some(FloatRounding::NearestEven),
            (false, 1) => Some(FloatRounding::Down),
            (false, 2) => Some(FloatRounding::Up),
            (false, _) => Some(FloatRounding::TowardZero),
        };
        match (self, operands) {
            (Self::RoundScale, [x]) => {
                let scale = i32::from(control >> 4);
                FloatRounding::decide(rounding, |mode| {
                    Some(float(*x)?.round_scaled(scale, mode)?.bits())
                })
            }
            (Self::RoundFixedPoint, [x]) => {
                let index = usize::from((control >> 4) & 7);
                let scale = *[0, 4, 5, 8, 16, 24, 31, 32].get(index)?;
                FloatRounding::decide(rounding, |mode| {
                    Some(float(*x)?.round_scaled(scale, mode)?.bits())
                })
            }
            (Self::Reduce, [x]) => {
                let value = float(*x)?;
                if value.is_nan() {
                    return Some(value.quieted().bits());
                }
                if !value.is_finite() {
                    return Some(0);
                }
                let scale = i32::from(control >> 4);
                FloatRounding::decide(rounding, |mode| {
                    let rounded = value.round_scaled(scale, mode)?;
                    let reduced = value.checked_sub(rounded, Some(mode))?;
                    (!reduced.is_subnormal()).then_some(reduced.bits())
                })
            }
            (Self::GetMantissa, [x]) => {
                Some(float(*x)?.mantissa(control & 3, (control >> 2) & 3)?.bits())
            }
            (Self::Range, [a, b]) => {
                let (first, second) = (float(*a)?, float(*b)?);
                if first.is_subnormal() || second.is_subnormal() {
                    return None;
                }
                if first.is_signaling() && first.is_nan() {
                    return Some(first.quieted().bits());
                }
                if second.is_signaling() && second.is_nan() {
                    return Some(second.quieted().bits());
                }
                let magnitude = |value: u128| value & !sign_bit;
                let picked = if second.is_nan() {
                    *a
                } else if first.is_nan() {
                    *b
                } else if magnitude(*a) == magnitude(*b) && negative(*a) != negative(*b) {
                    // Zeros and equal magnitudes of opposite signs: the
                    // minimum is the negative one, the maximum the positive.
                    let maximum = control & 1 != 0;
                    if maximum == negative(*a) { *b } else { *a }
                } else {
                    let (left, right) = match control & 2 {
                        0 => (first, second),
                        _ => (first.abs(), second.abs()),
                    };
                    let first_le = left.compare(right)? != std::cmp::Ordering::Greater;
                    match (control & 1 != 0, first_le) {
                        (false, true) | (true, false) => *a,
                        (false, false) | (true, true) => *b,
                    }
                };
                Some(match (control >> 2) & 3 {
                    0 => magnitude(picked) | (*a & sign_bit),
                    1 => picked,
                    2 => magnitude(picked),
                    _ => magnitude(picked) | sign_bit,
                })
            }
            (Self::Fixup, [destination, source, table]) => {
                let value = float(*source)?;
                if value.is_subnormal() {
                    return None;
                }
                let one = format.from_integer(1);
                let token = if value.is_nan() {
                    u32::from(value.is_signaling())
                } else if !value.is_finite() {
                    if negative(*source) { 4 } else { 5 }
                } else if *source & !sign_bit == 0 {
                    2
                } else if *source == one {
                    3
                } else if negative(*source) {
                    6
                } else {
                    7
                };
                let response = (*table >> token.checked_mul(4)?) & 0xf;
                let constant = |value: f64| {
                    FloatBits::new(
                        format,
                        FloatFormat::Double.convert(u128::from(value.to_bits()), format)?,
                    )
                };
                let infinity = constant(f64::INFINITY)?;
                let maximum = FloatBits::largest(format)?;
                Some(match response {
                    0 => *destination,
                    1 => *source,
                    2 => value.quieted().bits(),
                    3 => FloatBits::indefinite(format)?.bits(),
                    4 => infinity.negate().bits(),
                    5 => infinity.bits(),
                    6 if negative(*source) => infinity.negate().bits(),
                    6 => infinity.bits(),
                    7 => sign_bit,
                    8 => 0,
                    9 => one | sign_bit,
                    10 => one,
                    11 => constant(0.5)?.bits(),
                    12 => constant(90.0)?.bits(),
                    13 => constant(std::f64::consts::FRAC_PI_2)?.bits(),
                    14 => maximum.bits(),
                    _ => maximum.negate().bits(),
                })
            }
            (Self::Classify, [x]) => {
                let value = float(*x)?;
                if value.is_subnormal() {
                    return None;
                }
                let zero = *x & !sign_bit == 0;
                let infinite = !value.is_finite() && !value.is_nan();
                let classes = [
                    value.is_nan() && !value.is_signaling(),
                    zero && !negative(*x),
                    zero && negative(*x),
                    infinite && !negative(*x),
                    infinite && negative(*x),
                    false,
                    value.is_finite() && !zero && negative(*x),
                    value.is_nan() && value.is_signaling(),
                ];
                let hit = classes
                    .iter()
                    .enumerate()
                    .any(|(bit, class)| *class && control & (1 << bit) != 0);
                Some(if hit {
                    BitWidth::new(width).checked_mask_u128()?
                } else {
                    0
                })
            }
            _ => None,
        }
    }
}

/// Vector ternary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorTernaryKind {
    /// Lane-wise fused multiply-add.
    Fma,
    /// Lane-wise select controlled by a vector mask.
    Select,
    /// Lane-wise funnel shift left: concatenate the first two operands and shift
    /// left by the third (count), taking the high half (x86 `vpshldv*`).
    FunnelLeft,
    /// Lane-wise funnel shift right: concatenate the first two operands and
    /// shift right by the third (count), taking the low half (x86 `vpshrdv*`).
    FunnelRight,
    /// Lane-wise signed accumulate of the rounded high half of the doubled
    /// product, saturated once at the end: `sat((c * 2^n + 2 * a * b + 2^(n -
    /// 1)) >> n)` for the first two operands `a` and `b` and the accumulator
    /// `c` (AArch64 `sqrdmlah`). Saturating the product first and then the sum
    /// gives another value where `a * b` is the minimum squared.
    MulAddHighDoubleSatRound,
    /// [`Self::MulAddHighDoubleSatRound`] subtracting the doubled product:
    /// `sat((c * 2^n - 2 * a * b + 2^(n - 1)) >> n)` (AArch64 `sqrdmlsh`).
    MulSubHighDoubleSatRound,
    /// Lane-wise fix-up with NaN passthrough of the float second operand by
    /// the 21-bit table of the third, the first operand where the table
    /// leaves the lane unchanged (Xeon Phi `vfixupnanps`, `vfixupnanpd`;
    /// [`FloatBits::fixed_up_passing_nans`](crate::ir::FloatBits::fixed_up_passing_nans)).
    /// A signalling NaN raises invalid.
    FixupNan,
    /// Lane-wise fused multiply-subtract, `a * b - c`, rounded once, the sign
    /// inside the operation: a NaN operand is the result as it was read
    /// (x86's `vfmsub*`, Intel SDM vol. 1, table 14-17), where
    /// [`Self::Fma`] of a negated addend gives the negated NaN (Arm's
    /// `fnmsub`, which negates its operand first).
    Fms,
    /// Lane-wise fused negated multiply-add, `-(a * b) + c`, the sign inside
    /// the operation as [`Self::Fms`]'s (x86's `vfnmadd*`).
    Fnma,
    /// Lane-wise fused negated multiply-subtract, `-(a * b) - c`, the signs
    /// inside the operation as [`Self::Fms`]'s (x86's `vfnmsub*`).
    Fnms,
}

impl VectorTernaryKind {
    /// Whether the operation's result on float lanes depends on the rounding
    /// ([`VectorElement::rounding`]): the fused multiply-adds.
    #[must_use]
    pub const fn reads_rounding(self) -> bool {
        self.fused_signs().is_some()
    }

    /// The signs a fused multiply-add family member applies inside the
    /// operation, `(negate the product, negate the addend)`: `a * b + c`
    /// with `(false, false)` for [`Self::Fma`] through `-(a * b) - c` with
    /// `(true, true)` for [`Self::Fnms`]. `None` for any other kind.
    #[must_use]
    pub const fn fused_signs(self) -> Option<(bool, bool)> {
        match self {
            Self::Fma => Some((false, false)),
            Self::Fms => Some((false, true)),
            Self::Fnma => Some((true, false)),
            Self::Fnms => Some((true, true)),
            Self::Select
            | Self::FunnelLeft
            | Self::FunnelRight
            | Self::MulAddHighDoubleSatRound
            | Self::MulSubHighDoubleSatRound
            | Self::FixupNan => None,
        }
    }
}

/// Vector comparison operation kind.
///
/// Each variant's discriminant is its stable codec number, the `u16` it
/// converts to and from: append only, never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum VectorCompareKind {
    /// Lane-wise equality comparison.
    Eq = 0,
    /// Lane-wise inequality comparison.
    Ne = 2,
    /// Lane-wise less-than comparison.
    Lt = 3,
    /// Lane-wise less-than-or-equal comparison.
    Le = 4,
    /// Lane-wise greater-than comparison.
    Gt = 1,
    /// Lane-wise greater-than-or-equal comparison.
    Ge = 5,
    /// Lane-wise unordered test (either operand is NaN) - x86 `cmpps` predicate
    /// `UNORD`.
    Unordered = 6,
    /// Lane-wise ordered test (neither operand is NaN) - x86 `cmpps` predicate
    /// `ORD`.
    Ordered = 7,
    /// Lane-wise not-less-than (true when `!(a < b)`, including unordered) -
    /// x86 `cmpps` predicate `NLT`.
    NotLt = 8,
    /// Lane-wise not-less-or-equal (true when `!(a <= b)`, including unordered)
    /// - x86 `cmpps` predicate `NLE`.
    NotLe = 9,
    /// Lane-wise not-greater-or-equal (true when `!(a >= b)`, including
    /// unordered) - x86 `cmpps` predicate `NGE`.
    NotGe = 10,
    /// Lane-wise not-greater-than (true when `!(a > b)`, including unordered) -
    /// x86 `cmpps` predicate `NGT`.
    NotGt = 11,
    /// Lane-wise constant-true predicate - x86 `cmpps` `TRUE`.
    AlwaysTrue = 12,
    /// Lane-wise constant-false predicate - x86 `cmpps` `FALSE`.
    AlwaysFalse = 13,
    /// Lane-wise unordered-or-equal (true when either operand is NaN or the
    /// two are equal) - x86 `cmpps` predicate `EQ_UQ`, MIPS MSA `fcueq`.
    UnorderedEq = 14,
    /// Lane-wise ordered-and-not-equal (false when either operand is NaN) -
    /// x86 `cmpps` predicate `NEQ_OQ`, MIPS MSA `fcne`.
    OrderedNe = 15,
}

/// How one side of a vector cast reads or writes its lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorCastKind {
    /// Signed integer lanes.
    Signed,
    /// Unsigned integer lanes.
    Unsigned,
    /// Float lanes of the IEEE format of their width.
    Float,
    /// bfloat16 lanes.
    BFloat16,
    /// Float lanes whose rounding and special values the implementation
    /// decides ([`VectorElementKind::FloatOpen`]).
    FloatOpen,
}

impl VectorCastKind {
    /// The integer lanes of the given signedness: [`Self::Signed`] or
    /// [`Self::Unsigned`].
    #[must_use]
    pub const fn integer(signed: bool) -> Self {
        if signed { Self::Signed } else { Self::Unsigned }
    }

    /// Whether the lanes hold floats, of any format.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float | Self::BFloat16 | Self::FloatOpen)
    }

    /// Whether the implementation decides the lanes' rounding and special
    /// values ([`Self::FloatOpen`]).
    #[must_use]
    pub const fn opens_specials(self) -> bool {
        matches!(self, Self::FloatOpen)
    }
}

/// Vector mask binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorMaskBinaryKind {
    /// Lane-wise predicate conjunction.
    And,
    /// Lane-wise predicate disjunction.
    Or,
    /// Lane-wise predicate exclusive-or.
    Xor,
    /// Lane-wise `left & !right`.
    AndNot,
    /// Lane-wise `left | !right`.
    OrNot,
    /// Lane-wise negated conjunction.
    Nand,
    /// Lane-wise negated disjunction.
    Nor,
    /// Lane-wise equivalence.
    Xnor,
}

impl VectorMaskBinaryKind {
    /// Computes the binary predicate relation at one physical mask bit.
    #[must_use]
    pub const fn evaluate(self, left: bool, right: bool) -> bool {
        match self {
            Self::And => left && right,
            Self::Or => left || right,
            Self::Xor => left != right,
            Self::AndNot => left && !right,
            Self::OrNot => left || !right,
            Self::Nand => !(left && right),
            Self::Nor => !(left || right),
            Self::Xnor => left == right,
        }
    }
}

/// Scalar summary operation over vector lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorReduceKind {
    /// Adds all lanes. Float lanes add in an order the operation does not
    /// state, as SVE's pairwise `faddv` and RISC-V's `vfredusum` do; a sum
    /// in lane order is a chain of adds.
    Add,
    /// Multiplies all lanes, float lanes in an order the operation does not
    /// state.
    Mul,
    /// Bitwise-ands all lanes.
    And,
    /// Bitwise-ors all lanes.
    Or,
    /// Bitwise-xors all lanes.
    Xor,
    /// Computes the minimum lane, NaN as [`VectorBinaryKind::Min`] treats it.
    Min,
    /// Computes the maximum lane, NaN as [`VectorBinaryKind::Max`] treats it.
    Max,
    /// Tests whether any predicate lane is true.
    Any,
    /// Tests whether all predicate lanes are true.
    All,
    /// Computes the minimum lane, a quiet NaN lane losing to a number
    /// ([`VectorBinaryKind::MinNum`]): AArch64 `fminnmv`.
    MinNum,
    /// Computes the maximum lane, a quiet NaN lane losing to a number
    /// ([`VectorBinaryKind::MaxNum`]): AArch64 `fmaxnmv`.
    MaxNum,
    /// Takes the other lanes from the first: of a pair, the even lane less
    /// the odd (AMD XOP `vphsub*`). Only a grouped reduction
    /// ([`ReduceGroups`]) takes it.
    Difference,
    /// Computes the minimum lane as [`VectorBinaryKind::MinimumNumber`]
    /// does: AArch64 `fmininmv`.
    MinimumNumber,
    /// Computes the maximum lane as [`VectorBinaryKind::MaximumNumber`]
    /// does: AArch64 `fmaxinmv`.
    MaximumNumber,
    /// Computes the lane of least magnitude as
    /// [`VectorBinaryKind::MinimumMagnitude`] does: AArch64 `fminmgv`.
    MinimumMagnitude,
    /// Computes the lane of greatest magnitude as
    /// [`VectorBinaryKind::MaximumMagnitude`] does: AArch64 `fmaxmgv`.
    MaximumMagnitude,
    /// Computes the lane of least magnitude as
    /// [`VectorBinaryKind::MinimumMagnitudeNumber`] does: AArch64
    /// `fminmginmv`.
    MinimumMagnitudeNumber,
    /// Computes the lane of greatest magnitude as
    /// [`VectorBinaryKind::MaximumMagnitudeNumber`] does: AArch64
    /// `fmaxmginmv`.
    MaximumMagnitudeNumber,
}

/// How a [`SsaOp::VectorReduce`] groups its lanes: each run of `lanes`
/// adjacent lanes reduces into one lane of the result, `bits` wide, each lane
/// extended to it as the reduction's element reads it (signed or not). A sum
/// of byte pairs into halfwords (`vpaddl`, `uaddlp`), of eight absolute byte
/// differences into a quadword (`psadbw`'s second step), of a pair's
/// difference (`vphsubbw`).
///
/// [`SsaOp::VectorReduce`]: crate::ir::ops::def::SsaOp::VectorReduce
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReduceGroups {
    /// The lanes each result lane reduces.
    pub lanes: u32,
    /// The width of each result lane, in bits.
    pub bits: u32,
}

impl VectorReduceKind {
    /// Whether the reduction's result on float lanes depends on the rounding
    /// ([`VectorElement::rounding`]): the sum and the product.
    #[must_use]
    pub const fn reads_rounding(self) -> bool {
        matches!(self, Self::Add | Self::Mul)
    }

    /// Whether the reduction's result on integer lanes depends on reading them
    /// signed or unsigned ([`VectorElement::signed`]): the minimum and the
    /// maximum.
    #[must_use]
    pub const fn reads_signedness(self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }

    /// The operation the reduction applies between two lanes: the sum's
    /// add, the product's multiply, a minimum's or a maximum's own; `None`
    /// for the bitwise and predicate reductions, which compute no float.
    #[must_use]
    pub const fn step(self) -> Option<VectorBinaryKind> {
        Some(match self {
            Self::Add => VectorBinaryKind::Add,
            Self::Mul => VectorBinaryKind::Mul,
            Self::Min => VectorBinaryKind::Min,
            Self::Max => VectorBinaryKind::Max,
            Self::MinNum => VectorBinaryKind::MinNum,
            Self::MaxNum => VectorBinaryKind::MaxNum,
            _ => return self.extreme(),
        })
    }

    /// The minimum or maximum the reduction applies between lanes, for the
    /// IEEE 754-2019 forms each computes as ([`Self::MinimumNumber`] and its
    /// siblings), and `None` for any other reduction.
    #[must_use]
    pub const fn extreme(self) -> Option<VectorBinaryKind> {
        Some(match self {
            Self::MinimumNumber => VectorBinaryKind::MinimumNumber,
            Self::MaximumNumber => VectorBinaryKind::MaximumNumber,
            Self::MinimumMagnitude => VectorBinaryKind::MinimumMagnitude,
            Self::MaximumMagnitude => VectorBinaryKind::MaximumMagnitude,
            Self::MinimumMagnitudeNumber => VectorBinaryKind::MinimumMagnitudeNumber,
            Self::MaximumMagnitudeNumber => VectorBinaryKind::MaximumMagnitudeNumber,
            _ => return None,
        })
    }
}

/// Operation used when extracting a scalar bitmask from a vector or mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorBitmaskKind {
    /// Extracts each lane's most significant bit, as in x86 `pmovmskb`.
    LaneMostSignificantBits,
    /// Extracts predicate lane bits from a vector mask.
    PredicateBits,
}

/// Controls predicated vector memory behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorMaskMode {
    /// Inactive lanes preserve a passthrough vector value.
    Merge,
    /// Inactive lanes are zeroed or ignored.
    Zero,
}

/// Direction of an AVX-512-style vector lane packing operation.
///
/// Each variant's discriminant is its stable codec number, the `u16` it
/// converts to and from: append only, never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum VectorPackKind {
    /// Packs active source lanes contiguously into the low destination lanes.
    Compress = 0,
    /// Expands contiguous low source lanes into active destination lanes.
    Expand = 1,
}

/// Fault behavior for vector loads that may complete partially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorFaultMode {
    /// Ordinary load where any lane fault traps the instruction.
    Normal,
    /// The first active lane faults normally; later faults suppress following lanes.
    FirstFault,
    /// Element zero faults normally; a later fault shortens the active length.
    ///
    /// Masking element zero off does not transfer its trapping behavior to the
    /// first active element. Inactive elements never access memory or fault.
    FaultOnlyFirst,
    /// No element faults: one that would suppresses itself and every later
    /// element (SVE's `ldnf1`).
    NoFault,
}

/// Memory layout for segmented vector memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorSegmentLayout {
    /// Interleaved structure layout, such as RVV `vlseg` or AArch64 `ld2`.
    Interleaved,
    /// Consecutive structure layout, with each segment stored contiguously.
    Consecutive,
}

impl_kinded_payload! {
    VectorExtendInLaneData { signed, source_bits, element_bits, outputs, inputs };
    optional_input VectorElementCountData (streaming) { element_bits, multiplier, pattern, streaming, outputs };
    VectorSveComputeData { op, element_bits, rotation, outputs, inputs };
    VectorStructLoadReplicateData { count, element_bits, outputs, inputs };
    VectorPredicateWhileData { kind, unsigned, elements, outputs, inputs };
}
