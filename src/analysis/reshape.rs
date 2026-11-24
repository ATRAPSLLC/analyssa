//! Collapsing chains of bit-preserving vector reshapes.
//!
//! A register that is wider than the values it holds is written and read
//! through reshapes: a `float` placed in the low lane of an `xmm` register is
//! reinterpreted as a one-lane vector, widened by a shuffle that zero-fills the
//! lanes above it, merged with the register's old lanes by a second shuffle,
//! and reinterpreted as the register's storage type. Reading it back reverses
//! the trip. Each step is exact and each moves no bit, but a value that crosses
//! a few registers arrives wrapped in a dozen of them, and every reader - a
//! printer, a pattern matcher, a type inference - has to see through all of
//! them to find the value.
//!
//! This module says where every bit of a reshape's result comes from, and from
//! that, the shortest reshape that produces the same bits:
//!
//! - a result that is one value's bits unchanged is that value, reinterpreted
//!   or copied;
//! - a shuffle's result, lane by lane, is either zero or a lane of some value
//!   further up the chain, so it reads that value directly. A lane from an
//!   input that is a known zero is a zero lane, which is what turns a merge
//!   into a register an earlier `pxor` cleared into a plain zero-extension.
//!
//! Only ops that move bits are followed: `Copy`, `Bitcast`,
//! `VectorReinterpret`, `VectorShuffle`, `VectorExtract` and `VectorInsert`,
//! the bit fields an integer conversion, a constant shift, a mask and an or
//! of disjoint fields move, and constants, whose zero bits are zeros like any
//! other. Anything else is where the bits originate.
//!
//! A phi stops the trace, so a register joined at a loop's head hides the
//! value it holds from every reader. [`ReshapedPhis`] finds the webs of phis
//! whose every joined value is one reshape of a narrower value, which are that
//! reshape of the phis of the narrower values; rewritten so, the reshapes
//! reading the phis collapse to those phis.
//!
//! # Examples
//!
//! The shape it collapses, in the IR's own spelling:
//!
//! ```text
//! v2 = vreinterpret v1
//! v3 = vreinterpret v2
//! v4 = vshuffle v3 [L0, Z, Z, Z]   ; reads v1's lanes directly
//! ```

use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
};

use crate::{
    analysis::convert::Readers,
    ir::{
        function::SsaFunction,
        integer::BitWidth,
        lanes::VectorConst,
        ops::{SsaOp, VectorShuffleSelector},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{SsaVarId, VariableOrigin},
    },
    target::{
        Endianness, Target, VectorLaneKind, VectorShape, VectorShuffleLane, VectorShuffleMask,
    },
};

/// How many definitions one collapse may follow before it treats the value it
/// reached as an origin.
///
/// A shuffle has two inputs, so a walk can fan out; the budget bounds the whole
/// walk, not its depth, which keeps a collapse constant-time however the chain
/// is shaped. The chains a register round trip builds are a handful of links
/// deep.
const TRACE_BUDGET: usize = 48;

/// A run of bits in a value, read from its low end upward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bits {
    /// `width` bits that are zero.
    Zero(u32),
    /// `width` bits of `var`, starting at bit `offset`.
    From {
        /// The value the bits are read from.
        var: SsaVarId,
        /// The first bit read.
        offset: u32,
        /// How many bits are read.
        width: u32,
    },
}

impl Bits {
    /// The number of bits in the run.
    const fn width(self) -> u32 {
        match self {
            Self::Zero(width) | Self::From { width, .. } => width,
        }
    }
}

/// Where each bit of a value comes from: runs, from the low end upward.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Layout {
    /// The runs, adjacent runs of one source merged.
    runs: Vec<Bits>,
}

impl Layout {
    /// A layout of one run.
    fn of(run: Bits) -> Self {
        Self { runs: vec![run] }
    }

    /// The total number of bits the layout describes.
    fn width(&self) -> u32 {
        self.runs.iter().map(|run| run.width()).sum()
    }

    /// Whether every bit is zero.
    fn is_zero(&self) -> bool {
        self.runs.iter().all(|run| matches!(run, Bits::Zero(_)))
    }

    /// Appends `run`, merging it into the last run when they continue each
    /// other.
    fn push(&mut self, run: Bits) {
        if run.width() == 0 {
            return;
        }
        let merged = match (self.runs.last(), run) {
            (Some(Bits::Zero(before)), Bits::Zero(after)) => {
                Some(Bits::Zero(before.saturating_add(after)))
            }
            (
                Some(Bits::From { var, offset, width }),
                Bits::From {
                    var: next,
                    offset: next_offset,
                    width: next_width,
                },
            ) if *var == next && offset.checked_add(*width) == Some(next_offset) => {
                Some(Bits::From {
                    var: *var,
                    offset: *offset,
                    width: width.saturating_add(next_width),
                })
            }
            _ => None,
        };
        match merged {
            Some(merged) => {
                if let Some(last) = self.runs.last_mut() {
                    *last = merged;
                }
            }
            None => self.runs.push(run),
        }
    }

    /// Appends every run of `other`.
    fn extend(&mut self, other: &Self) {
        for run in &other.runs {
            self.push(*run);
        }
    }

    /// The `width` bits starting at bit `start`, or `None` when they run past
    /// the end.
    fn slice(&self, start: u32, width: u32) -> Option<Self> {
        let end = start.checked_add(width)?;
        let mut sliced = Self::default();
        let mut at = 0u32;
        for run in &self.runs {
            let run_end = at.saturating_add(run.width());
            let from = start.max(at);
            let to = end.min(run_end);
            if from < to {
                let skip = from.saturating_sub(at);
                sliced.push(match *run {
                    Bits::Zero(_) => Bits::Zero(to.saturating_sub(from)),
                    Bits::From { var, offset, .. } => Bits::From {
                        var,
                        offset: offset.saturating_add(skip),
                        width: to.saturating_sub(from),
                    },
                });
            }
            at = run_end;
        }
        (sliced.width() == width).then_some(sliced)
    }

    /// The bit positions at which a run of this layout begins, and its end.
    fn cuts(&self) -> Vec<u32> {
        let mut at = 0u32;
        let mut cuts = vec![0];
        for run in &self.runs {
            at = at.saturating_add(run.width());
            cuts.push(at);
        }
        cuts
    }

    /// The layout of `self | other`, when every bit is zero in at least one of
    /// the two - the shape a merge of disjoint fields has. `None` when both
    /// supply the same bit.
    fn or(&self, other: &Self) -> Option<Self> {
        let width = self.width();
        if other.width() != width {
            return None;
        }
        let mut cuts = self.cuts();
        cuts.extend(other.cuts());
        cuts.sort_unstable();
        cuts.dedup();
        let mut merged = Self::default();
        for pair in cuts.windows(2) {
            let [start, end] = pair else {
                continue;
            };
            let (left, right) = (
                self.slice(*start, end.saturating_sub(*start))?,
                other.slice(*start, end.saturating_sub(*start))?,
            );
            let chosen = if left.is_zero() {
                right
            } else if right.is_zero() {
                left
            } else {
                return None;
            };
            merged.extend(&chosen);
        }
        Some(merged)
    }

    /// The layout of `self & mask`: the bits the mask keeps, zero elsewhere.
    fn masked(&self, mask: u64) -> Option<Self> {
        let width = self.width();
        if width > 64 {
            return None;
        }
        let mut masked = Self::default();
        let mut start = 0u32;
        while start < width {
            let keep = (mask >> start) & 1 == 1;
            let mut end = start.saturating_add(1);
            while end < width && ((mask >> end) & 1 == 1) == keep {
                end = end.saturating_add(1);
            }
            if keep {
                masked.extend(&self.slice(start, end.saturating_sub(start))?);
            } else {
                masked.push(Bits::Zero(end.saturating_sub(start)));
            }
            start = end;
        }
        Some(masked)
    }

    /// The single source this layout is all of, when it is one run covering
    /// `var` from bit 0.
    fn whole(&self) -> Option<(SsaVarId, u32)> {
        match self.runs.as_slice() {
            [
                Bits::From {
                    var,
                    offset: 0,
                    width,
                },
            ] => Some((*var, *width)),
            _ => None,
        }
    }

    /// The values the layout's runs read bits of.
    fn sources(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.runs.iter().filter_map(|run| match run {
            Bits::From { var, .. } => Some(*var),
            Bits::Zero(_) => None,
        })
    }

    /// The layout with every run of `from`'s bits reading `to`'s instead.
    fn renamed(&self, from: SsaVarId, to: SsaVarId) -> Self {
        Self {
            runs: self
                .runs
                .iter()
                .map(|run| match *run {
                    Bits::From { var, offset, width } if var == from => Bits::From {
                        var: to,
                        offset,
                        width,
                    },
                    other => other,
                })
                .collect(),
        }
    }

    /// The constant of `source_type` that this layout, a shape whose every
    /// run is zero or bits of one source, rebuilds as `constant`: each bit a
    /// run reads from the source is the constant's bit under it, and each
    /// zero run has zeros under it. Source bits no run reads are zero.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function, for the constants `constant` reads.
    /// * `constant` - The layout of the joined constant, as wide as this one,
    ///   its every run zero or a constant's bits.
    /// * `source_type` - The type of the source, at most 128 bits wide.
    ///
    /// # Returns
    ///
    /// The constant, or `None` when no source value gives `constant`: a bit
    /// under a zero run is set, or two runs reading one source bit see
    /// different bits under them.
    fn source_constant<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        constant: &Self,
        source_type: &T::Type,
    ) -> Option<ConstValue<T>> {
        let source_width = T::bit_width(source_type)?;
        if source_width > 128 || !source_width.is_multiple_of(8) {
            return None;
        }
        let mut bits = 0u128;
        let mut known = 0u128;
        let mut at = 0u32;
        for run in &self.runs {
            let width = run.width();
            match *run {
                Bits::Zero(_) => {
                    let mut start = 0u32;
                    while start < width {
                        let chunk = width.saturating_sub(start).min(128);
                        if constant_bits(ssa, &constant.slice(at.checked_add(start)?, chunk)?)? != 0
                        {
                            return None;
                        }
                        start = start.saturating_add(chunk);
                    }
                }
                Bits::From { offset, .. } => {
                    let value = constant_bits(ssa, &constant.slice(at, width)?)?;
                    let mask = (width <= 128)
                        .then(|| BitWidth::new(width).mask_u128())?
                        .checked_shl(offset)?;
                    let placed = value.checked_shl(offset)?;
                    if (bits ^ placed) & known & mask != 0 {
                        return None;
                    }
                    bits |= placed;
                    known |= mask;
                }
            }
            at = at.checked_add(width)?;
        }
        let bytes = bits.to_le_bytes();
        ConstValue::from_typed_bytes(
            bytes.get(..usize::try_from(source_width.checked_div(8)?).ok()?)?,
            source_type,
            Endianness::Little,
        )
    }

    /// The chain rebuilding a value of `joined_type` laid out as this layout,
    /// a shape whose every run is zero or bits of `source`, from `source`.
    ///
    /// All of the source at its own width is one view of it. Anything else is
    /// a shuffle of the source's lanes, at the widest lane width every run
    /// starts and ends on in both values, between views of the source and of
    /// the result at that lane width where their types have other lanes.
    ///
    /// # Returns
    ///
    /// The steps, or `None` where the target has no vector type of the lanes
    /// the shuffle needs.
    fn steps<T: Target>(
        &self,
        source: SsaVarId,
        source_type: &T::Type,
        joined_type: &T::Type,
    ) -> Option<Vec<ReshapeStep<T>>> {
        let source_width = T::bit_width(source_type)?;
        let width = self.width();
        if self.whole() == Some((source, source_width)) && source_width == width {
            return Some(vec![ReshapeStep::View(joined_type.clone())]);
        }
        let gcd = |mut left: u32, mut right: u32| {
            while right != 0 {
                (left, right) = (right, left.checked_rem(right).unwrap_or(0));
            }
            left
        };
        let mut lane = gcd(source_width, width);
        for run in &self.runs {
            lane = gcd(lane, run.width());
            if let Bits::From { offset, .. } = run {
                lane = gcd(lane, *offset);
            }
        }
        let lanes_of = |ty: &T::Type, total: u32| {
            if T::vector_shape(ty).is_some_and(|shape| shape.lane_bits == lane) {
                return Some(ty.clone());
            }
            T::vector_type(VectorShape::new(
                total.checked_div(lane)?,
                VectorLaneKind::Integer,
                lane,
                total,
            )?)
        };
        let viewed = lanes_of(source_type, source_width)?;
        let shuffled = lanes_of(joined_type, width)?;
        let mask = (0..width.checked_div(lane)?)
            .map(
                |index| match self.slice(index.checked_mul(lane)?, lane)?.runs.as_slice() {
                    [Bits::Zero(_)] => Some(VectorShuffleLane::Zero),
                    [Bits::From { offset, .. }] => {
                        Some(VectorShuffleLane::Left(offset.checked_div(lane)?))
                    }
                    _ => None,
                },
            )
            .collect::<Option<Vec<_>>>()?;
        let mut steps = Vec::with_capacity(3);
        if viewed != *source_type {
            steps.push(ReshapeStep::View(viewed));
        }
        steps.push(ReshapeStep::Shuffle(
            VectorShuffleMask::new(mask),
            shuffled.clone(),
        ));
        if shuffled != *joined_type {
            steps.push(ReshapeStep::View(joined_type.clone()));
        }
        Some(steps)
    }

    /// The bits of the layout, each run of a value's bits resolved by
    /// `resolve` of the value, its first bit and the run's width, and each
    /// zero run zero; `None` where a run resolves to nothing.
    fn resolved(
        &self,
        mut resolve: impl FnMut(SsaVarId, u32, u32) -> Option<u128>,
    ) -> Option<u128> {
        let mut bits = 0u128;
        let mut at = 0u32;
        for run in &self.runs {
            if let Bits::From { var, offset, width } = *run {
                let mask = (width <= 128).then(|| BitWidth::new(width).mask_u128())?;
                bits |= (resolve(var, offset, width)? & mask).checked_shl(at)?;
            }
            at = at.checked_add(run.width())?;
        }
        Some(bits)
    }
}

/// Follows reshapes back to where their bits originate.
struct Tracer<'a, T: Target> {
    /// The function whose definitions are followed.
    ssa: &'a SsaFunction<T>,
    /// Definitions still allowed to be followed.
    budget: usize,
    /// For each origin, the values found to hold exactly its bits, nearest the
    /// origin first.
    aliases: BTreeMap<SsaVarId, Vec<SsaVarId>>,
}

impl<'a, T: Target> Tracer<'a, T> {
    /// A tracer over `ssa` with a fresh budget.
    fn new(ssa: &'a SsaFunction<T>) -> Self {
        Self {
            ssa,
            budget: TRACE_BUDGET,
            aliases: BTreeMap::new(),
        }
    }

    /// The bit width of `var`'s type.
    fn width(&self, var: SsaVarId) -> Option<u32> {
        self.ssa.bit_width(var)
    }

    /// The lane width of `var`'s type, when it is a fixed vector.
    fn lane_bits(&self, var: SsaVarId) -> Option<u32> {
        T::vector_shape(self.ssa.variable(var)?.var_type()).map(|shape| shape.lane_bits)
    }

    /// The bits of the integer constant `var` is, as wide as `width`.
    fn constant(&self, var: SsaVarId, width: u32) -> Option<u64> {
        let SsaOp::Const { value, .. } = self.ssa.recorded_definition(var)?.op() else {
            return None;
        };
        #[allow(clippy::cast_sign_loss)] // two's complement is the bit pattern wanted
        let bits = value
            .as_u64()
            .or_else(|| value.as_i64().map(|v| v as u64))?;
        Some(BitWidth::new(width).low_bits(bits))
    }

    /// Where the bits of `var` come from.
    fn trace(&mut self, var: SsaVarId) -> Option<Layout> {
        let width = self.width(var)?;
        let leaf = Layout::of(Bits::From {
            var,
            offset: 0,
            width,
        });
        if self.budget == 0 {
            return Some(leaf);
        }
        self.budget = self.budget.saturating_sub(1);
        let Some(definition) = self.ssa.recorded_definition(var) else {
            return Some(leaf);
        };
        let Some(layout) = self
            .bits_of(definition.op())
            .filter(|layout| layout.width() == width)
        else {
            return Some(leaf);
        };
        // `var` holds exactly another value's bits: it can stand in for that
        // value wherever its type fits better.
        if let Some((origin, origin_width)) = layout.whole()
            && origin_width == width
            && origin != var
        {
            self.aliases.entry(origin).or_default().push(var);
        }
        Some(layout)
    }

    /// Where the bits of `op`'s result come from, or `None` when `op` computes
    /// them rather than moving them.
    fn bits_of(&mut self, op: &SsaOp<T>) -> Option<Layout> {
        let dest = op.dest()?;
        let width = self.width(dest)?;
        match op {
            SsaOp::Copy { src, .. } => self.trace(*src),
            SsaOp::Bitcast { operand, .. } | SsaOp::VectorReinterpret { value: operand, .. } => {
                self.trace(*operand)
            }
            // A constant is its own bits up to its highest set bit and zeros
            // above it: the zero high half of a wide constant merges with a
            // field above it as any zero does.
            SsaOp::Const { value, .. } => {
                if value.is_zero() {
                    return Some(Layout::of(Bits::Zero(width)));
                }
                let significant = 128u32
                    .checked_sub(constant_run(value, 0, width)?.leading_zeros())?
                    .min(width);
                let mut layout = Layout::of(Bits::From {
                    var: dest,
                    offset: 0,
                    width: significant,
                });
                layout.push(Bits::Zero(width.saturating_sub(significant)));
                Some(layout)
            }
            // A splat of zero is zero at any width; of anything else it
            // computes new bits.
            SsaOp::VectorSplat { value: operand, .. } => self
                .trace(*operand)
                .filter(Layout::is_zero)
                .map(|_| Layout::of(Bits::Zero(width))),
            // An integer conversion keeps the low bits: all of them, then
            // zeros when it widens unsigned. A signed widening fills with copies
            // of a bit, which no run here can say - unless the value is zero.
            SsaOp::IntConv {
                operand,
                overflow_check: false,
                unsigned,
                ..
            } => {
                let source = self.trace(*operand)?;
                let source_width = source.width();
                if width <= source_width {
                    source.slice(0, width)
                } else if *unsigned || source.is_zero() {
                    let mut widened = source;
                    widened.push(Bits::Zero(width.saturating_sub(source_width)));
                    Some(widened)
                } else {
                    None
                }
            }
            // A mask keeps the bits it names.
            SsaOp::And { left, right, .. } => {
                let (mask, other) =
                    match (self.constant(*right, width), self.constant(*left, width)) {
                        (Some(mask), _) => (mask, *left),
                        (None, Some(mask)) => (mask, *right),
                        (None, None) => return None,
                    };
                self.trace(other)?.masked(mask)
            }
            // An or of disjoint fields is both of them.
            SsaOp::Or { left, right, .. } => {
                let left = self.trace(*left)?;
                let right = self.trace(*right)?;
                left.or(&right)
            }
            // A shift by a constant moves the bits and fills with zeros; an
            // arithmetic right shift fills with the sign, which is not a run.
            SsaOp::Shl { value, amount, .. } => {
                let shift = u32::try_from(self.constant(*amount, 64)?).ok()?;
                if shift >= width {
                    return Some(Layout::of(Bits::Zero(width)));
                }
                let source = self.trace(*value)?;
                let mut shifted = Layout::default();
                shifted.push(Bits::Zero(shift));
                shifted.extend(&source.slice(0, width.saturating_sub(shift))?);
                Some(shifted)
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned: true,
                ..
            } => {
                let shift = u32::try_from(self.constant(*amount, 64)?).ok()?;
                if shift >= width {
                    return Some(Layout::of(Bits::Zero(width)));
                }
                let mut shifted = self
                    .trace(*value)?
                    .slice(shift, width.saturating_sub(shift))?;
                shifted.push(Bits::Zero(shift));
                Some(shifted)
            }
            SsaOp::VectorExtract { vector, lane, .. } => {
                let source = self.trace(*vector)?;
                source.slice(lane.checked_mul(width)?, width)
            }
            // An insert replaces one lane, as wide as the value inserted, and
            // keeps the others.
            SsaOp::VectorInsert {
                vector: base,
                lane,
                value,
                ..
            } => {
                let base = self.trace(*base)?;
                let inserted = self.trace(*value)?;
                let lane_bits = inserted.width();
                let at = lane.checked_mul(lane_bits)?;
                let end = at.checked_add(lane_bits)?;
                let mut layout = base.slice(0, at)?;
                layout.extend(&inserted);
                layout.extend(&base.slice(end, width.checked_sub(end)?)?);
                Some(layout)
            }
            SsaOp::VectorShuffle {
                left,
                right,
                selector: VectorShuffleSelector::Mask(mask),
                ..
            } => {
                let lane_bits = self.lane_bits(dest)?;
                let left = self.trace(*left)?;
                let right = match right {
                    Some(right) => Some(self.trace(*right)?),
                    None => None,
                };
                let mut layout = Layout::default();
                for lane in mask.lanes() {
                    let bits = match lane {
                        VectorShuffleLane::Zero => Layout::of(Bits::Zero(lane_bits)),
                        VectorShuffleLane::Left(index) => {
                            left.slice(index.checked_mul(lane_bits)?, lane_bits)?
                        }
                        VectorShuffleLane::Right(index) => right
                            .as_ref()?
                            .slice(index.checked_mul(lane_bits)?, lane_bits)?,
                        VectorShuffleLane::Undef => return None,
                    };
                    layout.extend(&bits);
                }
                Some(layout)
            }
            _ => None,
        }
    }

    /// A value holding exactly `origin`'s bits whose type is a vector of
    /// `lane_bits` lanes, preferring the one nearest `origin`.
    fn lane_view(&self, origin: SsaVarId, lane_bits: u32) -> Option<SsaVarId> {
        let origin_width = self.width(origin)?;
        iter::once(origin)
            .chain(self.aliases.get(&origin).into_iter().flatten().copied())
            .find(|candidate| {
                self.lane_bits(*candidate) == Some(lane_bits)
                    && self.width(*candidate) == Some(origin_width)
            })
    }
}

/// The `width` bits from bit `offset` of the integer, float or vector constant
/// `value`, `width` up to 128: an integer's two's-complement bits, a float's
/// encoding, or a vector's lanes laid out from lane zero, a vector of any
/// width (a 512-bit register's).
fn constant_run<T: Target>(value: &ConstValue<T>, offset: u32, width: u32) -> Option<u128> {
    let mask = (width <= 128).then(|| BitWidth::new(width).mask_u128())?;
    let bits = match value {
        ConstValue::Vector(vector) => return Some(vector.bits_at(offset, width)),
        ConstValue::Float(float) => float.bits(),
        _ => u128::from(
            value
                .as_u64()
                .or_else(|| value.as_i64().map(i64::cast_unsigned))?,
        ),
    };
    Some(bits.checked_shr(offset).unwrap_or(0) & mask)
}

/// The bits `layout` holds when its every run is zero or a constant's bits
/// (all zero included), `layout` at most 128 bits wide.
fn constant_bits<T: Target>(ssa: &SsaFunction<T>, layout: &Layout) -> Option<u128> {
    layout.resolved(|var, offset, width| {
        let SsaOp::Const { value, .. } = ssa.recorded_definition(var)?.op() else {
            return None;
        };
        constant_run(value, offset, width)
    })
}

/// The constant `layout` is when its every run is zero or a constant's bits
/// (all zero included): an integer of `dest_type`, or a vector of its lanes,
/// of any width, read 128 bits at a time.
fn constant_result<T: Target>(
    ssa: &SsaFunction<T>,
    layout: &Layout,
    dest_type: &T::Type,
) -> Option<ConstValue<T>> {
    let width = layout.width();
    if let Some(shape) = T::vector_shape(dest_type) {
        let chunk = width.clamp(1, 128);
        let chunks = (0..width.div_ceil(chunk))
            .map(|index| {
                let start = index.checked_mul(chunk)?;
                constant_bits(
                    ssa,
                    &layout.slice(start, chunk.min(width.checked_sub(start)?))?,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        return VectorConst::from_lanes(chunk, chunks)?
            .resized(shape.lane_bits, shape.lane_count)
            .map(ConstValue::Vector);
    }
    if !T::is_integer(dest_type) {
        return None;
    }
    let bits = constant_bits(ssa, layout)?;
    if width == 128 {
        ConstValue::of_lane(128, bits)
    } else {
        ConstValue::from_bits_of_type(u64::try_from(bits).ok()?, dest_type)
    }
}

/// The `width` bits from bit `start` of `op`'s result, traced through the
/// reshapes and bit fields that build it, each run of another value's bits
/// resolved by `resolve` of that value, its first bit and the run's width.
///
/// # Returns
///
/// The bits, or `None` when `op` is no reshape of other values or a run
/// resolves to nothing.
#[must_use]
pub fn known_slice<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    start: u32,
    width: u32,
    resolve: impl FnMut(SsaVarId, u32, u32) -> Option<u128>,
) -> Option<u128> {
    let dest = op.dest()?;
    let mut tracer = Tracer::new(ssa);
    let layout = tracer
        .bits_of(op)
        .filter(|layout| tracer.width(dest) == Some(layout.width()))?
        .slice(start, width)?;
    layout.resolved(resolve)
}

/// Returns the shortest reshape producing the same bits as `op`, when it is
/// shorter than `op`.
///
/// `op` must be a reshape - `Bitcast`, `VectorReinterpret`, `VectorShuffle` or
/// `VectorExtract`. Its result is traced back, bit by bit, through every
/// reshape feeding it (see the [module documentation](self)), and rebuilt:
///
/// - all of one value's bits, at that value's type, is a `Copy` of it;
/// - all of one value's bits at another type is one `VectorReinterpret` (a
///   vector result) or `Bitcast` (a scalar one) of it;
/// - a shuffle's lanes, each zero or a lane of at most two values that have a
///   view at the shuffle's lane width, is one shuffle of those views;
/// - an extract of a lane of such a value is an extract from its view.
///
/// # Arguments
///
/// * `ssa` - The function, for definitions and types.
/// * `op` - The reshape to collapse.
///
/// # Returns
///
/// The replacement op defining `op`'s destination, or `None` when nothing
/// shorter exists.
#[must_use]
pub fn collapse_reshape<T: Target>(ssa: &SsaFunction<T>, op: &SsaOp<T>) -> Option<SsaOp<T>> {
    if !matches!(
        op,
        SsaOp::Bitcast { .. }
            | SsaOp::VectorReinterpret { .. }
            | SsaOp::VectorShuffle {
                selector: VectorShuffleSelector::Mask(_),
                ..
            }
            | SsaOp::VectorExtract { .. }
            | SsaOp::IntConv {
                overflow_check: false,
                ..
            }
            | SsaOp::And { .. }
            | SsaOp::Or { .. }
            | SsaOp::Shl { .. }
            | SsaOp::Shr { .. }
    ) {
        return None;
    }
    let dest = op.dest()?;
    let dest_type = ssa.variable(dest)?.var_type().clone();
    let mut tracer = Tracer::new(ssa);
    let width = tracer.width(dest)?;
    let layout = tracer.bits_of(op)?;
    if layout.width() != width {
        return None;
    }

    // Bits that are all a constant's or zero are a constant.
    if let Some(value) = constant_result(ssa, &layout, &dest_type)
        && !matches!(op, SsaOp::Const { .. })
    {
        return Some(SsaOp::Const { dest, value });
    }

    if let Some((origin, origin_width)) = layout.whole()
        && origin_width == width
        && tracer.width(origin) == Some(width)
    {
        let replacement = view(dest, origin, ssa.variable(origin)?.var_type(), &dest_type);
        // Already the one conversion, whatever signedness it states.
        if matches!(replacement, SsaOp::IntConv { .. })
            && matches!(op, SsaOp::IntConv { operand, .. } if *operand == origin)
        {
            return None;
        }
        return (replacement != *op).then_some(replacement);
    }

    // An integer result that is the low bits of an integer, or an integer
    // with zeros above it, is one conversion of it: a field inserted into a
    // register and read back out is the field.
    if T::is_integer(&dest_type) {
        let origin_run = match layout.runs.as_slice() {
            [Bits::From { var, offset: 0, .. }] => Some(*var),
            [
                Bits::From {
                    var,
                    offset: 0,
                    width: kept,
                },
                Bits::Zero(_),
            ] if tracer.width(*var) == Some(*kept) => Some(*var),
            _ => None,
        };
        if let Some(origin) = origin_run
            && ssa
                .variable(origin)
                .is_some_and(|variable| T::is_integer(variable.var_type()))
        {
            // Already the one conversion, whatever signedness it states.
            if matches!(op, SsaOp::IntConv { operand, .. } if *operand == origin) {
                return None;
            }
            let replacement = SsaOp::IntConv {
                dest,
                operand: origin,
                target: dest_type,
                overflow_check: false,
                unsigned: true,
            };
            return (replacement != *op).then_some(replacement);
        }
    }

    let replacement = match op {
        SsaOp::VectorShuffle {
            selector: VectorShuffleSelector::Mask(_),
            ..
        } => {
            let lane_bits = tracer.lane_bits(dest)?;
            let lanes = width.checked_div(lane_bits)?;
            let mut inputs: Vec<SsaVarId> = Vec::with_capacity(2);
            let mut mask = Vec::with_capacity(usize::try_from(lanes).ok()?);
            for lane in 0..lanes {
                let bits = layout.slice(lane.saturating_mul(lane_bits), lane_bits)?;
                let selected = match bits.runs.as_slice() {
                    [Bits::Zero(_)] => VectorShuffleLane::Zero,
                    [Bits::From { var, offset, .. }] if offset.is_multiple_of(lane_bits) => {
                        let view = tracer.lane_view(*var, lane_bits)?;
                        let index = offset.checked_div(lane_bits)?;
                        match inputs.iter().position(|input| *input == view) {
                            Some(0) => VectorShuffleLane::Left(index),
                            Some(_) => VectorShuffleLane::Right(index),
                            None if inputs.len() < 2 => {
                                inputs.push(view);
                                if inputs.len() == 1 {
                                    VectorShuffleLane::Left(index)
                                } else {
                                    VectorShuffleLane::Right(index)
                                }
                            }
                            None => return None,
                        }
                    }
                    _ => return None,
                };
                mask.push(selected);
            }
            let left = *inputs.first()?;
            let identity = inputs.len() == 1
                && mask.iter().enumerate().all(|(at, lane)| {
                    u32::try_from(at).ok().map(VectorShuffleLane::Left) == Some(*lane)
                });
            if identity && ssa.variable(left)?.var_type() == &dest_type {
                return Some(SsaOp::Copy { dest, src: left });
            }
            SsaOp::VectorShuffle {
                dest,
                left,
                right: inputs.get(1).copied(),
                selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(mask)),
            }
        }
        SsaOp::VectorExtract { .. } => {
            let [Bits::From { var, offset, .. }] = layout.runs.as_slice() else {
                return None;
            };
            if !offset.is_multiple_of(width) {
                return None;
            }
            let vector = tracer.lane_view(*var, width)?;
            SsaOp::VectorExtract {
                dest,
                vector,
                lane: offset.checked_div(width)?,
            }
        }
        _ => return None,
    };
    (replacement != *op).then_some(replacement)
}

/// The op giving `dest`, of `dest_type`, the bits of `origin`, of
/// `origin_type` and the same width: a copy at the same type, a
/// reinterpretation for a vector, a sign change for two integers, and a
/// bitcast for anything else.
fn view<T: Target>(
    dest: SsaVarId,
    origin: SsaVarId,
    origin_type: &T::Type,
    dest_type: &T::Type,
) -> SsaOp<T> {
    if origin_type == dest_type {
        SsaOp::Copy { dest, src: origin }
    } else if T::vector_shape(dest_type).is_some() {
        SsaOp::VectorReinterpret {
            dest,
            value: origin,
            target_type: dest_type.clone(),
        }
    } else if T::is_integer(dest_type) && T::is_integer(origin_type) {
        SsaOp::IntConv {
            dest,
            operand: origin,
            target: dest_type.clone(),
            overflow_check: false,
            unsigned: true,
        }
    } else {
        SsaOp::Bitcast {
            dest,
            operand: origin,
            target: dest_type.clone(),
        }
    }
}

/// One op of a chain that rebuilds a value from the value it is a reshape of,
/// reading the result of the step before it.
#[derive(Debug, Clone)]
pub enum ReshapeStep<T: Target> {
    /// The same bits at another type of their width.
    View(T::Type),
    /// The lanes the mask selects from the one input, zero lanes among them,
    /// at the type.
    Shuffle(VectorShuffleMask, T::Type),
}

impl<T: Target> ReshapeStep<T> {
    /// The type of the step's result.
    #[must_use]
    pub fn result_type(&self) -> &T::Type {
        match self {
            Self::View(ty) | Self::Shuffle(_, ty) => ty,
        }
    }

    /// The op computing the step into `dest` from `operand`.
    ///
    /// # Arguments
    ///
    /// * `dest` - The value the op defines, of the step's result type.
    /// * `operand` - The value the step reads: the source, or the result of
    ///   the step before.
    /// * `operand_type` - The type of `operand`.
    #[must_use]
    pub fn op(&self, dest: SsaVarId, operand: SsaVarId, operand_type: &T::Type) -> SsaOp<T> {
        match self {
            Self::View(ty) => view(dest, operand, operand_type, ty),
            Self::Shuffle(mask, _) => SsaOp::VectorShuffle {
                dest,
                left: operand,
                right: None,
                selector: VectorShuffleSelector::Mask(mask.clone()),
            },
        }
    }
}

/// What a phi of a [`ReshapedPhis`] web joins from one predecessor, as the
/// phi of the sources joins it.
#[derive(Debug, Clone)]
pub enum SourceOperand<T: Target> {
    /// The value the joined value is the reshape of.
    Value(SsaVarId),
    /// The result of a phi of the web, whose source phi is joined.
    Phi(SsaVarId),
    /// A constant of the source type, whose reshape the joined constant is,
    /// which the caller defines in the predecessor.
    Constant(ConstValue<T>),
}

/// One phi of a [`ReshapedPhis`] web.
#[derive(Debug, Clone)]
pub struct ReshapedPhi<T: Target> {
    /// The block the phi is in.
    pub block: usize,
    /// The phi's result.
    pub result: SsaVarId,
    /// The phi's origin, which the phi of the sources takes over.
    pub origin: VariableOrigin,
    /// Per operand: the predecessor, the value the phi joins from it, and
    /// what the phi of the sources joins in its place.
    pub operands: Vec<(usize, SsaVarId, SourceOperand<T>)>,
}

/// A web of phis whose every joined value is one reshape of a value of one
/// type, and the phis of those values they are that reshape of.
///
/// A value a machine keeps in a register wider than it is joined at the
/// register's width: a `double` an AArch64 loop carries in `d0` is the low
/// lane of `q0`, written into the register's storage by a reinterpretation, a
/// shuffle that zero-fills the upper lane and a reinterpretation back, and read
/// out by the reverse. The phi joining the register at a loop's head or exit
/// then hides the `double` from every reader, and two exits that return the
/// same `double` - one before the write, one through the phi and back out -
/// read as different values. When every value a phi joins is the same reshape
/// of a value of one type, or a constant that reshape gives, the phi is that
/// reshape of the phi joining those values. Phis that join each other are one
/// web, rewritten together, so a loop's register joined at its head and again
/// at its exits keeps one source phi per join.
///
/// A web is left as it is where a joined value is anything else, where an
/// operand comes over an exceptional edge, and where an operation reads one of
/// its phis other than to move its bits - a `Bitcast`, `VectorReinterpret`,
/// `VectorShuffle` or `VectorExtract`, which collapse to the source once the
/// phi is the reshape of it ([`collapse_reshape`]) - since a phi read whole
/// would keep its reshape alive beside the source phi.
#[derive(Debug, Clone)]
pub struct ReshapedPhis<T: Target> {
    /// The type of the joined sources.
    pub source_type: T::Type,
    /// The chain rebuilding a phi's value from its source phi's, in order:
    /// at least one step, the last of the joined type.
    pub steps: Vec<ReshapeStep<T>>,
    /// The web's phis.
    pub phis: Vec<ReshapedPhi<T>>,
}

impl<T: Target> ReshapedPhis<T> {
    /// Every web of `ssa`'s phis that is the reshape of a web of its sources'
    /// phis.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function.
    /// * `readers` - Who reads each value of `ssa` ([`Readers::of`]).
    #[must_use]
    pub fn of(ssa: &SsaFunction<T>, readers: &Readers<'_, T>) -> Vec<Self> {
        let mut phis: BTreeMap<SsaVarId, (usize, &PhiNode)> = BTreeMap::new();
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            for phi in block.phi_nodes() {
                phis.insert(phi.result(), (block_idx, phi));
            }
        }
        let mut seen = BTreeSet::new();
        let mut webs = Vec::new();
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
            if let Some(found) = Self::web(ssa, readers, &phis, &web) {
                webs.push(found);
            }
        }
        webs
    }

    /// The reshapes `webs` join that nothing reads once their phis are
    /// rewritten, each with the source standing for it from then on: read by
    /// no operation, and by phis of the webs alone - of any web, as two webs
    /// may join one reshape (a loop's head and an exit it leaves by early).
    ///
    /// # Arguments
    ///
    /// * `webs` - The webs being rewritten.
    /// * `readers` - Who reads each value of the function ([`Readers::of`]).
    #[must_use]
    pub fn released(webs: &[Self], readers: &Readers<'_, T>) -> Vec<(SsaVarId, SsaVarId)> {
        let joining: BTreeSet<SsaVarId> = webs
            .iter()
            .flat_map(|web| web.phis.iter().map(|phi| phi.result))
            .collect();
        let mut released: Vec<(SsaVarId, SsaVarId)> = Vec::new();
        for (_, joined, source) in webs
            .iter()
            .flat_map(|web| &web.phis)
            .flat_map(|phi| &phi.operands)
        {
            if let SourceOperand::Value(source) = source
                && readers.operations(*joined).is_empty()
                && readers
                    .phis(*joined)
                    .iter()
                    .all(|reader| joining.contains(reader))
                && !released.contains(&(*joined, *source))
            {
                released.push((*joined, *source));
            }
        }
        released
    }

    /// The web of the phis `web` names, when they are a reshape of their
    /// sources' phis.
    fn web(
        ssa: &SsaFunction<T>,
        readers: &Readers<'_, T>,
        phis: &BTreeMap<SsaVarId, (usize, &PhiNode)>,
        web: &[SsaVarId],
    ) -> Option<Self> {
        let joined_type = ssa.variable(*web.first()?)?.var_type().clone();
        let width = T::bit_width(&joined_type)?;
        // The first reshaped value's source and layout, which every other
        // reshaped value repeats with its own source.
        let mut shape: Option<(SsaVarId, Layout)> = None;
        // The constants joined, by phi and operand, as layouts.
        let mut constants: Vec<(usize, usize, Layout)> = Vec::new();
        let mut found = Vec::with_capacity(web.len());
        for result in web {
            let (block, phi) = phis.get(result)?;
            if *ssa.variable(*result)?.var_type() != joined_type
                || !readers.operations(*result).iter().all(|op| {
                    matches!(
                        op,
                        SsaOp::Bitcast { .. }
                            | SsaOp::VectorReinterpret { .. }
                            | SsaOp::VectorShuffle {
                                selector: VectorShuffleSelector::Mask(_),
                                ..
                            }
                            | SsaOp::VectorExtract { .. }
                    )
                })
            {
                return None;
            }
            let mut operands = Vec::with_capacity(phi.operand_count());
            for operand in phi.operands() {
                if operand.is_exceptional() {
                    return None;
                }
                let value = operand.value();
                let source = if phis.contains_key(&value) {
                    SourceOperand::Phi(value)
                } else {
                    let layout = Tracer::new(ssa)
                        .trace(value)
                        .filter(|layout| layout.width() == width)?;
                    let sources: BTreeSet<SsaVarId> = layout.sources().collect();
                    if sources.iter().all(|var| {
                        matches!(
                            ssa.recorded_definition(*var)
                                .map(|definition| definition.op()),
                            Some(SsaOp::Const { .. })
                        )
                    }) {
                        constants.push((found.len(), operands.len(), layout));
                        // Filled once the shape is known.
                        SourceOperand::Phi(*result)
                    } else {
                        let [source] = sources.into_iter().collect::<Vec<_>>()[..] else {
                            return None;
                        };
                        if source == value {
                            return None;
                        }
                        match &shape {
                            Some((first, first_layout)) => {
                                if ssa.variable(source)?.var_type()
                                    != ssa.variable(*first)?.var_type()
                                    || layout.renamed(source, *first) != *first_layout
                                {
                                    return None;
                                }
                            }
                            None => shape = Some((source, layout)),
                        }
                        SourceOperand::Value(source)
                    }
                };
                operands.push((operand.predecessor(), value, source));
            }
            found.push(ReshapedPhi {
                block: *block,
                result: *result,
                origin: phi.origin(),
                operands,
            });
        }
        let (source, layout) = shape?;
        let source_type = ssa.variable(source)?.var_type().clone();
        for (phi, operand, constant) in constants {
            let value = layout.source_constant(ssa, &constant, &source_type)?;
            found.get_mut(phi)?.operands.get_mut(operand)?.2 = SourceOperand::Constant(value);
        }
        Some(Self {
            steps: layout.steps(source, &source_type, &joined_type)?,
            source_type,
            phis: found,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{SsaBlockBuilder, SsaDefSpec, SsaFunctionBuilder},
        testing::{MockTarget, MockType, function_of},
    };

    /// The op defining `var` in `ssa`.
    fn def_of(ssa: &SsaFunction<MockTarget>, var: SsaVarId) -> SsaOp<MockTarget> {
        ssa.recorded_definition(var)
            .map(|definition| definition.op().clone())
            .unwrap_or(SsaOp::Nop)
    }

    /// Lane selectors from a compact spelling: `Lk`, `Rk`, `Z` or `U`.
    fn mask(lanes: &[&str]) -> VectorShuffleMask {
        VectorShuffleMask::new(
            lanes
                .iter()
                .map(|lane| {
                    let index = || lane[1..].parse().unwrap_or(0);
                    match lane.as_bytes().first() {
                        Some(b'L') => VectorShuffleLane::Left(index()),
                        Some(b'R') => VectorShuffleLane::Right(index()),
                        Some(b'Z') => VectorShuffleLane::Zero,
                        _ => VectorShuffleLane::Undef,
                    }
                })
                .collect(),
        )
    }

    /// **A reinterpret of a reinterpret reads the original.** Two steps that
    /// move no bit are one step, and back to the original type is none.
    #[test]
    fn a_reinterpret_round_trip_is_the_original() {
        let (ssa, vars) = function_of(&[MockType::V4I32, MockType::V2F64, MockType::V4I32], |v| {
            vec![
                SsaOp::VectorReinterpret {
                    dest: v[1],
                    value: v[0],
                    target_type: MockType::V2F64,
                },
                SsaOp::VectorReinterpret {
                    dest: v[2],
                    value: v[1],
                    target_type: MockType::V4I32,
                },
            ]
        });
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[2])),
            Some(SsaOp::Copy {
                dest: vars[2],
                src: vars[0]
            }),
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[1])),
            None,
            "one reinterpret is already the shortest form",
        );
    }

    /// **A lane merged from a cleared register is a zero lane.** A shuffle
    /// taking lane 0 from one value and the rest from a splat of zero is a
    /// zero-extension of that value's lane 0; the zero input disappears.
    #[test]
    fn a_merge_with_a_zero_register_is_a_zero_extension() {
        let (ssa, vars) = function_of(
            &[
                MockType::V4I32,
                MockType::I32,
                MockType::V4I32,
                MockType::V4I32,
            ],
            |v| {
                vec![
                    SsaOp::Const {
                        dest: v[1],
                        value: ConstValue::I32(0),
                    },
                    SsaOp::VectorSplat {
                        dest: v[2],
                        value: v[1],
                        vector_type: MockType::V4I32,
                    },
                    SsaOp::VectorShuffle {
                        dest: v[3],
                        left: v[0],
                        right: Some(v[2]),
                        selector: VectorShuffleSelector::Mask(mask(&["L0", "R1", "R2", "R3"])),
                    },
                ]
            },
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[3])),
            Some(SsaOp::VectorShuffle {
                dest: vars[3],
                left: vars[0],
                right: None,
                selector: VectorShuffleSelector::Mask(mask(&["L0", "Z", "Z", "Z"])),
            }),
        );
    }

    /// **Bits that are a constant's or zero are a constant.** A shuffle of a
    /// constant vector's lanes and zero lanes is the constant those lanes
    /// spell, one whose every lane is zero (a lane a merge cleared, another a
    /// zero constant's) is zero, and a lane of an unknown vector beside them
    /// is not a constant.
    #[test]
    fn bits_that_are_a_constants_or_zero_are_a_constant() {
        let words =
            |lanes: [u128; 4]| ConstValue::Vector(VectorConst::from_lanes(32, lanes).unwrap());
        let (ssa, vars) = function_of(
            &[
                MockType::V4I32,
                MockType::V4I32,
                MockType::V4I32,
                MockType::V4I32,
                MockType::V4I32,
            ],
            |v| {
                vec![
                    SsaOp::Const {
                        dest: v[1],
                        value: words([1, 2, 3, 4]),
                    },
                    SsaOp::VectorShuffle {
                        dest: v[2],
                        left: v[1],
                        right: None,
                        selector: VectorShuffleSelector::Mask(mask(&["L3", "Z", "L1", "L0"])),
                    },
                    SsaOp::VectorShuffle {
                        dest: v[3],
                        left: v[0],
                        right: Some(v[1]),
                        selector: VectorShuffleSelector::Mask(mask(&["Z", "Z", "Z", "Z"])),
                    },
                    SsaOp::VectorShuffle {
                        dest: v[4],
                        left: v[0],
                        right: Some(v[1]),
                        selector: VectorShuffleSelector::Mask(mask(&["L0", "R1", "Z", "R3"])),
                    },
                ]
            },
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[2])),
            Some(SsaOp::Const {
                dest: vars[2],
                value: words([4, 0, 2, 1]),
            })
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[3])),
            Some(SsaOp::Const {
                dest: vars[3],
                value: words([0; 4]),
            })
        );
        assert!(!matches!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[4])),
            Some(SsaOp::Const { .. })
        ));
    }

    /// **A constant wider than 128 bits is a constant.** A shuffle of a
    /// 512-bit constant's high 256-bit lane and a zero lane is the constant
    /// those lanes spell, read from and built into vectors wider than any one
    /// integer; a lane of an unknown vector beside them is not a constant.
    #[test]
    fn bits_of_a_wide_constant_are_a_constant() {
        let limbs = |limbs: [u128; 8]| {
            VectorConst::from_lanes(64, limbs)
                .and_then(|vector| vector.resized(256, 2))
                .map(ConstValue::Vector)
        };
        let source = [1, 2, 3, 4, 5, 6, 7, u128::from(u64::MAX)];
        let (ssa, vars) = function_of(
            &[
                MockType::V2I256,
                MockType::V2I256,
                MockType::V2I256,
                MockType::V2I256,
            ],
            |v| {
                vec![
                    SsaOp::Const {
                        dest: v[1],
                        value: limbs(source).unwrap_or(ConstValue::I32(0)),
                    },
                    SsaOp::VectorShuffle {
                        dest: v[2],
                        left: v[0],
                        right: Some(v[1]),
                        selector: VectorShuffleSelector::Mask(mask(&["R1", "Z"])),
                    },
                    SsaOp::VectorShuffle {
                        dest: v[3],
                        left: v[0],
                        right: Some(v[1]),
                        selector: VectorShuffleSelector::Mask(mask(&["L0", "R1"])),
                    },
                ]
            },
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[2])),
            limbs([5, 6, 7, u128::from(u64::MAX), 0, 0, 0, 0]).map(|value| SsaOp::Const {
                dest: vars[2],
                value,
            })
        );
        assert!(!matches!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[3])),
            Some(SsaOp::Const { .. })
        ));
    }

    /// **A register that only kept a nonzero value's lanes keeps them.** The
    /// control for the zero merge: a splat of one is not zero, so the merge
    /// has nothing to drop.
    #[test]
    fn a_merge_with_a_nonzero_register_keeps_both_inputs() {
        let (ssa, vars) = function_of(
            &[
                MockType::V4I32,
                MockType::I32,
                MockType::V4I32,
                MockType::V4I32,
            ],
            |v| {
                vec![
                    SsaOp::Const {
                        dest: v[1],
                        value: ConstValue::I32(1),
                    },
                    SsaOp::VectorSplat {
                        dest: v[2],
                        value: v[1],
                        vector_type: MockType::V4I32,
                    },
                    SsaOp::VectorShuffle {
                        dest: v[3],
                        left: v[0],
                        right: Some(v[2]),
                        selector: VectorShuffleSelector::Mask(mask(&["L0", "R1", "R2", "R3"])),
                    },
                ]
            },
        );
        assert_eq!(collapse_reshape(&ssa, &def_of(&ssa, vars[3])), None);
    }

    /// **A shuffle of a shuffle is one shuffle.** Reversing the lanes twice
    /// selects every lane where it was: the original.
    #[test]
    fn a_shuffle_of_a_shuffle_composes() {
        let (ssa, vars) = function_of(&[MockType::V4I32, MockType::V4I32, MockType::V4I32], |v| {
            vec![
                SsaOp::VectorShuffle {
                    dest: v[1],
                    left: v[0],
                    right: None,
                    selector: VectorShuffleSelector::Mask(mask(&["L3", "L2", "L1", "L0"])),
                },
                SsaOp::VectorShuffle {
                    dest: v[2],
                    left: v[1],
                    right: None,
                    selector: VectorShuffleSelector::Mask(mask(&["L3", "L2", "L1", "L0"])),
                },
            ]
        });
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[2])),
            Some(SsaOp::Copy {
                dest: vars[2],
                src: vars[0]
            }),
        );
    }

    /// **A lane read through a reinterpret reads the reinterpreted value's
    /// view.** An extract from a shuffle of a reinterpret names the lane where
    /// it originates, in the nearest value shaped with that lane width.
    #[test]
    fn an_extract_reads_the_lane_where_it_originates() {
        let (ssa, vars) = function_of(
            &[
                MockType::V4I32,
                MockType::V2F64,
                MockType::V4I32,
                MockType::V4I32,
                MockType::I32,
            ],
            |v| {
                vec![
                    SsaOp::VectorReinterpret {
                        dest: v[1],
                        value: v[0],
                        target_type: MockType::V2F64,
                    },
                    SsaOp::VectorReinterpret {
                        dest: v[2],
                        value: v[1],
                        target_type: MockType::V4I32,
                    },
                    SsaOp::VectorShuffle {
                        dest: v[3],
                        left: v[2],
                        right: None,
                        selector: VectorShuffleSelector::Mask(mask(&["L2", "L3", "Z", "Z"])),
                    },
                    SsaOp::VectorExtract {
                        dest: v[4],
                        vector: v[3],
                        lane: 1,
                    },
                ]
            },
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[4])),
            Some(SsaOp::VectorExtract {
                dest: vars[4],
                vector: vars[0],
                lane: 3,
            }),
        );
    }

    /// **A lane of anything else is left alone.** A shuffle of an entry value
    /// has nothing further up to read, and an undefined lane has no source.
    #[test]
    fn a_shuffle_of_an_origin_is_already_shortest() {
        let (ssa, vars) = function_of(&[MockType::V4I32, MockType::V4I32], |v| {
            vec![SsaOp::VectorShuffle {
                dest: v[1],
                left: v[0],
                right: None,
                selector: VectorShuffleSelector::Mask(mask(&["L1", "Z", "U", "L0"])),
            }]
        });
        assert_eq!(collapse_reshape(&ssa, &def_of(&ssa, vars[1])), None);
    }

    /// **A field inserted into a wider value and read back is the field.** The
    /// high word of `(hi << 32) | (lo & 0xffffffff)`, shifted down and
    /// truncated, is `hi`; its low word is `lo`. A register pair written half at
    /// a time and read whole or by half spells exactly this.
    #[test]
    fn a_field_read_back_out_of_a_merge_is_the_field() {
        let (ssa, vars) = function_of(
            &[
                MockType::U32, // 0 lo
                MockType::U32, // 1 hi
                MockType::U64, // 2 zext lo
                MockType::U64, // 3 zext hi
                MockType::U64, // 4 const 32
                MockType::U64, // 5 hi << 32
                MockType::U64, // 6 pair
                MockType::U64, // 7 pair >> 32
                MockType::U32, // 8 high word
                MockType::U32, // 9 low word
            ],
            |v| {
                vec![
                    SsaOp::IntConv {
                        dest: v[2],
                        operand: v[0],
                        target: MockType::U64,
                        overflow_check: false,
                        unsigned: true,
                    },
                    SsaOp::IntConv {
                        dest: v[3],
                        operand: v[1],
                        target: MockType::U64,
                        overflow_check: false,
                        unsigned: true,
                    },
                    SsaOp::Const {
                        dest: v[4],
                        value: ConstValue::U64(32),
                    },
                    SsaOp::Shl {
                        dest: v[5],
                        value: v[3],
                        amount: v[4],
                        flags: None,
                    },
                    SsaOp::Or {
                        dest: v[6],
                        left: v[5],
                        right: v[2],
                        flags: None,
                    },
                    SsaOp::Shr {
                        dest: v[7],
                        value: v[6],
                        amount: v[4],
                        unsigned: true,
                        flags: None,
                    },
                    SsaOp::IntConv {
                        dest: v[8],
                        operand: v[7],
                        target: MockType::U32,
                        overflow_check: false,
                        unsigned: true,
                    },
                    SsaOp::IntConv {
                        dest: v[9],
                        operand: v[6],
                        target: MockType::U32,
                        overflow_check: false,
                        unsigned: true,
                    },
                ]
            },
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[8])),
            Some(SsaOp::Copy {
                dest: vars[8],
                src: vars[1]
            }),
        );
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[9])),
            Some(SsaOp::Copy {
                dest: vars[9],
                src: vars[0]
            }),
        );
        // `pair >> 32` is the high word zero-extended: one conversion of it.
        assert_eq!(
            collapse_reshape(&ssa, &def_of(&ssa, vars[7])),
            Some(SsaOp::IntConv {
                dest: vars[7],
                operand: vars[1],
                target: MockType::U64,
                overflow_check: false,
                unsigned: true,
            }),
        );
    }

    /// **A sign extension is not a run of bits.** Its high bits copy one bit of
    /// the value, which a layout cannot say, so a field read out of it is left
    /// alone - the control for the merge above.
    #[test]
    fn a_sign_extension_is_not_traced() {
        let (ssa, vars) = function_of(
            &[MockType::I32, MockType::I64, MockType::I32, MockType::U64],
            |v| {
                vec![
                    SsaOp::IntConv {
                        dest: v[1],
                        operand: v[0],
                        target: MockType::I64,
                        overflow_check: false,
                        unsigned: false,
                    },
                    SsaOp::Const {
                        dest: v[3],
                        value: ConstValue::U64(32),
                    },
                    SsaOp::IntConv {
                        dest: v[2],
                        operand: v[1],
                        target: MockType::I32,
                        overflow_check: false,
                        unsigned: false,
                    },
                ]
            },
        );
        // The low word of the extension is the value, but that is the
        // conversion chain's to fold; the layout of the extension itself is
        // opaque above bit 31.
        let widened = def_of(&ssa, vars[1]);
        assert_eq!(collapse_reshape(&ssa, &widened), None);
    }

    /// One arm of [`diamond`]: emits the value the join takes from it, from
    /// the `double` arguments.
    type Arm = fn(&mut SsaBlockBuilder<'_, MockTarget>, &[SsaVarId]) -> crate::Result<SsaVarId>;

    /// A diamond whose join phi, of `joined`, takes `left`'s value and
    /// `right`'s, and whose exit returns the phi's bits read as `read`, so
    /// the phi is read only to move its bits. Arguments 0 to 2 are `double`s
    /// the entry loads for the arms; argument 3 picks the arm.
    fn diamond(joined: MockType, read: MockType, left: Arm, right: Arm) -> SsaFunction<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
        let args = builder
            .in_block(0, |block| {
                let args = (0..3)
                    .map(|index| block.load_arg(SsaDefSpec::tmp(MockType::F64), index))
                    .collect::<crate::Result<Vec<_>>>()?;
                let flag = block.load_arg(SsaDefSpec::tmp(MockType::I32), 3)?;
                block.branch(flag, 1, 2)?;
                Ok(args)
            })
            .expect("entry");
        let mut values = Vec::new();
        for (index, arm) in [(1, left), (2, right)] {
            values.push(
                builder
                    .in_block(index, |block| {
                        let value = arm(block, &args)?;
                        block.jump(3)?;
                        Ok(value)
                    })
                    .expect("arm"),
            );
        }
        builder
            .in_block(3, |block| {
                let joined_type = joined;
                let joined = block.phi(
                    SsaDefSpec::new(VariableOrigin::Local(0), joined),
                    [(1, values[0]), (2, values[1])],
                )?;
                let read = block.emit_def(SsaDefSpec::tmp(read), |dest| {
                    view(dest, joined, &joined_type, &read)
                })?;
                block.ret(Some(read))
            })
            .expect("exit");
        builder.finish().expect("finish")
    }

    /// The `double` `value` in both lanes of a `v2i64`.
    fn both_lanes(
        block: &mut SsaBlockBuilder<'_, MockTarget>,
        value: SsaVarId,
    ) -> crate::Result<SsaVarId> {
        let lane = block.emit_def(SsaDefSpec::tmp(MockType::V1I64), |dest| {
            SsaOp::VectorReinterpret {
                dest,
                value,
                target_type: MockType::V1I64,
            }
        })?;
        block.emit_def(SsaDefSpec::tmp(MockType::V2I64), |dest| {
            SsaOp::VectorShuffle {
                dest,
                left: lane,
                right: None,
                selector: VectorShuffleSelector::Mask(mask(&["L0", "L0"])),
            }
        })
    }

    /// The `double`s `low` and `high` in the two lanes of a `v2i64`.
    fn pair(
        block: &mut SsaBlockBuilder<'_, MockTarget>,
        low: SsaVarId,
        high: SsaVarId,
    ) -> crate::Result<SsaVarId> {
        let mut views = Vec::with_capacity(2);
        for value in [low, high] {
            views.push(block.emit_def(SsaDefSpec::tmp(MockType::V1I64), |dest| {
                SsaOp::VectorReinterpret {
                    dest,
                    value,
                    target_type: MockType::V1I64,
                }
            })?);
        }
        let (left, right) = (views[0], views[1]);
        block.emit_def(SsaDefSpec::tmp(MockType::V2I64), |dest| {
            SsaOp::VectorShuffle {
                dest,
                left,
                right: Some(right),
                selector: VectorShuffleSelector::Mask(mask(&["L0", "R0"])),
            }
        })
    }

    /// A `v2i64` constant of two lanes.
    fn lanes(
        block: &mut SsaBlockBuilder<'_, MockTarget>,
        low: u64,
        high: u64,
    ) -> crate::Result<SsaVarId> {
        block.const_value(
            SsaDefSpec::tmp(MockType::V2I64),
            ConstValue::Vector(
                VectorConst::from_lanes(64, [u128::from(low), u128::from(high)])
                    .ok_or_else(|| crate::Error::new("lanes"))?,
            ),
        )
    }

    /// **A web's phis join the values their joined values are reshapes of.**
    /// A phi of two `double`s each reinterpreted as an `i64` is one view of
    /// the phi of the `double`s. A phi of a `double` in both lanes of a
    /// register and a constant with one value in both lanes joins that value
    /// as a `double`, and the shuffle that rebuilds the register reads the
    /// source's lane twice. The controls: a constant whose two lanes differ is
    /// no `double` in both lanes, and registers of two `double`s each, the
    /// high one the same, have no one source; neither phi is a web.
    #[test]
    fn a_web_joins_the_sources_its_values_are_reshapes_of() {
        let ssa = diamond(
            MockType::I64,
            MockType::F64,
            |block, args| {
                let value = args[0];
                block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::Bitcast {
                    dest,
                    operand: value,
                    target: MockType::I64,
                })
            },
            |block, args| {
                let value = args[1];
                block.emit_def(SsaDefSpec::tmp(MockType::I64), |dest| SsaOp::Bitcast {
                    dest,
                    operand: value,
                    target: MockType::I64,
                })
            },
        );
        let webs = ReshapedPhis::of(&ssa, &Readers::of(&ssa));
        let [web] = webs.as_slice() else {
            panic!("one web: {webs:?}");
        };
        assert_eq!(web.source_type, MockType::F64);
        assert!(
            matches!(web.steps.as_slice(), [ReshapeStep::View(MockType::I64)]),
            "{web:?}"
        );
        assert!(
            web.phis.iter().all(|phi| phi
                .operands
                .iter()
                .all(|(_, _, source)| matches!(source, SourceOperand::Value(_)))),
            "{web:?}"
        );

        let two = 2.0_f64.to_bits();
        let ssa = diamond(
            MockType::V2I64,
            MockType::V4I32,
            |block, args| both_lanes(block, args[0]),
            |block, _| lanes(block, 2.0_f64.to_bits(), 2.0_f64.to_bits()),
        );
        let webs = ReshapedPhis::of(&ssa, &Readers::of(&ssa));
        let [web] = webs.as_slice() else {
            panic!("one web: {webs:?}");
        };
        assert_eq!(web.source_type, MockType::F64);
        assert!(
            matches!(
                web.steps.as_slice(),
                [
                    ReshapeStep::View(MockType::V1I64),
                    ReshapeStep::Shuffle(shuffle, MockType::V2I64),
                ] if *shuffle == mask(&["L0", "L0"])
            ),
            "{web:?}"
        );
        assert!(
            web.phis.iter().any(|phi| {
                phi.operands.iter().any(|(_, _, source)| matches!(
                source,
                SourceOperand::Constant(value) if *value == ConstValue::from(f64::from_bits(two))
            ))
            }),
            "the constant joins as the double in both its lanes: {web:?}"
        );

        let ssa = diamond(
            MockType::V2I64,
            MockType::V4I32,
            |block, args| both_lanes(block, args[0]),
            |block, _| lanes(block, 2.0_f64.to_bits(), 3.0_f64.to_bits()),
        );
        assert!(ReshapedPhis::of(&ssa, &Readers::of(&ssa)).is_empty());

        let ssa = diamond(
            MockType::V2I64,
            MockType::V4I32,
            |block, args| pair(block, args[0], args[2]),
            |block, args| pair(block, args[1], args[2]),
        );
        assert!(ReshapedPhis::of(&ssa, &Readers::of(&ssa)).is_empty());
    }

    /// **A reshape two webs join is released by both.** A branch's two arms
    /// each write a `double` into a register and leave for either of two
    /// exits, each exit joining the registers: two webs, neither reading the
    /// other, sharing the reshapes. Rewritten together, neither reshape is
    /// read any longer, so each hands its place to its `double`. The control:
    /// a reshape something else reads too stays its own.
    #[test]
    fn a_reshape_two_webs_join_is_released_by_both() {
        let build = |read_elsewhere: bool| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
            let args = builder
                .in_block(0, |block| {
                    let args = (0..2)
                        .map(|index| block.load_arg(SsaDefSpec::tmp(MockType::F64), index))
                        .collect::<crate::Result<Vec<_>>>()?;
                    let flag = block.load_arg(SsaDefSpec::tmp(MockType::I32), 3)?;
                    block.branch(flag, 1, 2)?;
                    Ok(args)
                })
                .expect("entry");
            let mut packs = Vec::new();
            for (index, value) in [(1, args[0]), (2, args[1])] {
                packs.push(
                    builder
                        .in_block(index, |block| {
                            let lanes = both_lanes(block, value)?;
                            if read_elsewhere && index == 1 {
                                block.emit_def(SsaDefSpec::tmp(MockType::V4I32), |dest| {
                                    SsaOp::VectorReinterpret {
                                        dest,
                                        value: lanes,
                                        target_type: MockType::V4I32,
                                    }
                                })?;
                            }
                            let flag = block.load_arg(SsaDefSpec::tmp(MockType::I32), 2)?;
                            block.branch(flag, 3, 4)?;
                            Ok(lanes)
                        })
                        .expect("arm"),
                );
            }
            for exit in [3, 4] {
                builder
                    .in_block(exit, |block| {
                        let joined = block.phi(
                            SsaDefSpec::new(VariableOrigin::Local(0), MockType::V2I64),
                            [(1, packs[0]), (2, packs[1])],
                        )?;
                        let read = block.emit_def(SsaDefSpec::tmp(MockType::V4I32), |dest| {
                            SsaOp::VectorReinterpret {
                                dest,
                                value: joined,
                                target_type: MockType::V4I32,
                            }
                        })?;
                        block.ret(Some(read))
                    })
                    .expect("exit");
            }
            (builder.finish().expect("finish"), args, packs)
        };
        let (ssa, args, packs) = build(false);
        let readers = Readers::of(&ssa);
        let webs = ReshapedPhis::of(&ssa, &readers);
        assert_eq!(webs.len(), 2, "{webs:?}");
        let mut released = ReshapedPhis::released(&webs, &readers);
        released.sort();
        let mut expected = vec![(packs[0], args[0]), (packs[1], args[1])];
        expected.sort();
        assert_eq!(released, expected);

        let (ssa, args, packs) = build(true);
        let readers = Readers::of(&ssa);
        let webs = ReshapedPhis::of(&ssa, &readers);
        assert_eq!(webs.len(), 2, "{webs:?}");
        assert_eq!(
            ReshapedPhis::released(&webs, &readers),
            vec![(packs[1], args[1])]
        );
    }
}
