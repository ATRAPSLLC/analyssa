//! Drops the operands of bitwise operations no reader demands the bits of.
//!
//! A value's readers decide which of its bits matter. `(uint8_t)(x | y)`
//! reads the low byte of the `or` and nothing else, so where `x` has no bit
//! in that byte, the `or` is `y` to every reader. Machine code writes such
//! values all the time: a write of a register's low byte (`setg cl`, `mov
//! al, 1`) keeps the rest of the register, a lift states it as `(r & ~0xff) |
//! b`, and the wider arithmetic after it (`and eax, ecx`) carries the kept
//! bits along until a narrowing (`movzx eax, al`) drops them. Each kept term
//! names the register the function was entered with, though no bit of it
//! reaches anything the function does.
//!
//! # Demand
//!
//! Each integer value of at most 64 bits gets the bits its readers demand:
//! the union, over every reader, of what that reader takes. A reader this
//! pass does not model, a store, a call, a return, a comparison or a
//! branch, demands every bit. The modelled readers hand on what their own
//! result is demanded of: a bitwise operation and a copy the same bits, a
//! select its two values, a shift by a constant the shifted bits (and an
//! arithmetic right shift the sign bit where it fills a demanded one), a
//! truncation the bits it keeps and an extension those of its operand an
//! extension does not fill (and the sign bit where it does), and an add,
//! subtract, multiply or negate every bit up to the highest demanded one,
//! since a carry runs upward only. An operation whose flags output something
//! reads demands every bit of its operands, as the flags depend on all of
//! them. The demands are the least fixpoint, from nothing up, so a value a
//! loop only feeds back to itself is demanded of nothing.
//!
//! # Lanes
//!
//! A value too wide for a bit per bit - a vector - is demanded by the byte
//! instead, which fits every vector up to 512 bits in the same mask. The
//! readers that hand on what they take are the lane ones: a shuffle through
//! the lanes its mask selects, an extract its lane, an insert everything but
//! the lane it writes, a splat the scalar it repeats, and a reinterpretation
//! or a bitcast the same bits restated in the units the operand counts.
//! Every other reader demands every byte. A span is always rounded out to
//! whole units, so a demand is never understated, and a value the mask
//! cannot state part by part - a lane of no stated width, or a vector of
//! more bytes than the mask has bits, as an x86 AMX tile is - is demanded
//! whole.
//!
//! This is the same shape of waste as the kept register bits, in the vector
//! file: a write of one view of a vector register - ARM's `Dj` of a `Qi`, an
//! x86 legacy-SSE write of an `xmm` - keeps the rest of the register, and the
//! lift states that as a shuffle of the register's previous value with the
//! written one. Where nothing reads the kept lanes, that shuffle is the only
//! reader of the previous value, and at a function's entry it is what makes
//! the register look like a parameter.
//!
//! # Rewrites
//!
//! An `or` or `xor` one of whose operands has no possibly set bit among the
//! demanded ones ([`Reader::known_bits`](crate::analysis::bits::Reader::known_bits)) is its other operand, and an `and` with a
//! constant that keeps every demanded bit is its other operand. The rewritten
//! operation is a copy, which copy propagation and dead-code elimination then
//! fold away with whatever only it read.
//!
//! A shuffle no demanded lane of which selects an input does not read that
//! input: those lanes read zero and the input goes. A shuffle left with no
//! input at all - every demanded lane zero or undefined - is the zero
//! vector, since nothing can tell an undefined lane from a zero one.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    Result,
    analysis::bits::Reader,
    ir::{
        function::SsaFunction,
        integer::BitWidth,
        lanes::VectorConst,
        ops::{SsaOp, VectorShuffleSelector},
        value::ConstValue,
        variable::SsaVarId,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::{Target, VectorShuffleLane, VectorShuffleMask},
};

/// The bits of each integer value its readers demand.
struct Demand {
    /// The demanded bits of each value; a value not here is demanded of
    /// nothing.
    bits: FxHashMap<SsaVarId, u64>,
    /// Every value something reads, taken off the function itself: a pass
    /// that removed a reader need not have kept the use lists current.
    read: FxHashSet<SsaVarId>,
}

impl Demand {
    /// The least fixpoint of every value's demand in `ssa`.
    fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let mut demand = Self {
            bits: FxHashMap::default(),
            read: ssa
                .blocks()
                .iter()
                .flat_map(|block| block.used_variables())
                .collect(),
        };
        // Where each value is defined: an instruction's operation, or the
        // operands of a phi.
        let mut definitions: FxHashMap<SsaVarId, Definition<'_, T>> = FxHashMap::default();
        let mut pending: Vec<SsaVarId> = Vec::new();
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                definitions.insert(phi.result(), Definition::Phi(phi.operands()));
                // A phi demands of its operands only what its result is
                // demanded of, which starts at nothing.
            }
            for instr in block.instructions() {
                let op = instr.op();
                for dest in op.defs() {
                    definitions.insert(dest, Definition::Op(op));
                }
                // What the operation demands of its operands before anything
                // demands of its result: everything, where it is a reader
                // this pass does not model, and the operands a modelled one
                // demands whole (a shift amount, a select's condition).
                for (operand, bits) in contributions(ssa, op, &demand) {
                    if demand.add(ssa, operand, bits) {
                        pending.push(operand);
                    }
                }
            }
        }
        while let Some(value) = pending.pop() {
            let Some(definition) = definitions.get(&value) else {
                continue;
            };
            let contributed = match definition {
                Definition::Op(op) => contributions(ssa, op, &demand),
                Definition::Phi(operands) => {
                    let bits = demand.of_value(value);
                    operands
                        .iter()
                        .map(|operand| (operand.value(), bits))
                        .collect()
                }
            };
            for (operand, bits) in contributed {
                if demand.add(ssa, operand, bits) {
                    pending.push(operand);
                }
            }
        }
        demand
    }

    /// The bits of `value` some reader demands.
    fn of_value(&self, value: SsaVarId) -> u64 {
        self.bits.get(&value).copied().unwrap_or(0)
    }

    /// Adds `bits` to what `value` is demanded of, cut to its width in its
    /// own units ([`Widths::unit_bits`]); `true` when that grew.
    fn add<T: Target>(&mut self, ssa: &SsaFunction<T>, value: SsaVarId, bits: u64) -> bool {
        let limit = Widths { ssa }
            .units(value)
            .map_or(u64::MAX, |units| BitWidth::new(units).mask());
        let bits = bits & limit;
        let entry = self.bits.entry(value).or_insert(0);
        let grown = *entry | bits;
        let changed = grown != *entry;
        *entry = grown;
        changed
    }

    /// The operand `op` is on every bit its result is demanded of, when it is
    /// an `or` or `xor` whose other operand has no possibly set bit there, or
    /// an `and` with a constant that keeps every demanded bit; `None`
    /// otherwise, or where the operation's flags are read or its result is
    /// not of the operand's type.
    fn kept_operand<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        op: &SsaOp<T>,
        constants: &std::collections::BTreeMap<SsaVarId, ConstValue<T>>,
    ) -> Option<SsaVarId> {
        let widths = Widths { ssa };
        let (dest, left, right, flags, and) = match op {
            SsaOp::Or {
                dest,
                left,
                right,
                flags,
            }
            | SsaOp::Xor {
                dest,
                left,
                right,
                flags,
            } => (*dest, *left, *right, *flags, false),
            SsaOp::And {
                dest,
                left,
                right,
                flags,
            } => (*dest, *left, *right, *flags, true),
            _ => return None,
        };
        if flags.is_some_and(|flags| self.read.contains(&flags)) {
            return None;
        }
        let demanded = self.of_value(dest);
        if demanded == 0 || widths.width(dest).is_none() {
            return None;
        }
        let kept = if and {
            let keeps = |mask: SsaVarId| {
                constants
                    .get(&mask)
                    .and_then(ConstValue::as_u64_bits)
                    .is_some_and(|mask| demanded & !mask == 0)
            };
            if keeps(right) {
                left
            } else if keeps(left) {
                right
            } else {
                return None;
            }
        } else {
            let silent = |term: SsaVarId| {
                Reader::new(ssa, constants)
                    .known_bits(term)
                    .is_some_and(|known| known.possibly_set() & demanded == 0)
            };
            if silent(left) {
                right
            } else if silent(right) {
                left
            } else {
                return None;
            }
        };
        let same_type = ssa
            .variable(kept)
            .zip(ssa.variable(dest))
            .is_some_and(|(kept, dest)| kept.var_type() == dest.var_type());
        same_type.then_some(kept)
    }

    /// The shuffle `op` with every input no demanded lane selects dropped,
    /// which is what makes it stop reading that input: the lanes that
    /// selected it read zero, and nothing can tell - no reader demands them.
    /// A shuffle left with no input at all is the zero vector.
    ///
    /// Zero rather than [`VectorShuffleLane::Undef`], which is what those
    /// lanes are: an undefined lane is as good as a zero one to a reader, and
    /// every reader can say what a zero lane holds. An undefined one stops
    /// the constant evaluator
    /// ([`VectorConst::shuffle`](crate::ir::lanes::VectorConst::shuffle))
    /// and the layout tracer
    /// ([`known_slice`](crate::analysis::reshape::known_slice)) alike, which
    /// would cost every lane of the value its answer to spare the lanes
    /// nobody reads.
    ///
    /// `None` where every input a demanded lane selects is read, where the
    /// result is demanded of nothing (dead-code elimination's), or where the
    /// lanes cannot be read: an index selector, an unknown width, a width
    /// that is no multiple of the lane count, or a lane past the ones a
    /// 64-bit mask can state ([`Widths::span_mask`]).
    fn shuffle_without_unread<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        op: &SsaOp<T>,
    ) -> Option<SsaOp<T>> {
        let widths = Widths { ssa };
        let SsaOp::VectorShuffle {
            dest,
            left,
            right,
            selector,
        } = op
        else {
            return None;
        };
        let mask = selector.mask()?;
        let demanded = self.of_value(*dest);
        if demanded == 0 {
            return None;
        }
        let lanes = mask.lanes();
        let lane_bits = widths.lane_bits(*dest, lanes.len())?;
        let mut reads_left = false;
        let mut reads_right = false;
        for (index, lane) in (0u32..).zip(lanes) {
            // A lane the mask cannot state is a lane that may be demanded,
            // so the shuffle stands as it is.
            if demanded & widths.lane_mask(*dest, index, lane_bits)? == 0 {
                continue;
            }
            match lane {
                VectorShuffleLane::Left(_) => reads_left = true,
                VectorShuffleLane::Right(_) => reads_right = true,
                VectorShuffleLane::Undef | VectorShuffleLane::Zero => {}
            }
        }
        match (reads_left, reads_right) {
            // Every demanded lane selects zero or nothing, so the whole
            // value is zero as far as anything can tell.
            (false, false) => Some(SsaOp::Const {
                dest: *dest,
                value: ConstValue::Vector(VectorConst::zeroed(
                    lane_bits,
                    u32::try_from(lanes.len()).ok()?,
                )?),
            }),
            // The second input alone is read: it becomes the first, and the
            // lanes that selected the first read zero.
            (false, true) => {
                let right = (*right)?;
                let lanes = lanes
                    .iter()
                    .map(|lane| match lane {
                        VectorShuffleLane::Right(source) => VectorShuffleLane::Left(*source),
                        VectorShuffleLane::Left(_) => VectorShuffleLane::Zero,
                        kept => *kept,
                    })
                    .collect();
                Some(SsaOp::VectorShuffle {
                    dest: *dest,
                    left: right,
                    right: None,
                    selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(lanes)),
                })
            }
            // The first input alone is read: the second goes, and the lanes
            // that selected it read zero.
            (true, false) if right.is_some() => {
                let lanes = lanes
                    .iter()
                    .map(|lane| match lane {
                        VectorShuffleLane::Right(_) => VectorShuffleLane::Zero,
                        kept => *kept,
                    })
                    .collect();
                Some(SsaOp::VectorShuffle {
                    dest: *dest,
                    left: *left,
                    right: None,
                    selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(lanes)),
                })
            }
            _ => None,
        }
    }
}

/// Where a value is defined.
enum Definition<'a, T: Target> {
    /// By an instruction's operation.
    Op(&'a SsaOp<T>),
    /// By a phi of these operands.
    Phi(&'a [crate::ir::phi::PhiOperand]),
}

/// What `op` demands of each of its operands, given what `demand` says its
/// result is demanded of.
fn contributions<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    demand: &Demand,
) -> Vec<(SsaVarId, u64)> {
    let widths = Widths { ssa };
    let all = |values: &[SsaVarId]| values.iter().map(|&value| (value, u64::MAX)).collect();
    let result = |dest: SsaVarId| demand.of_value(dest);
    let flags_read =
        |flags: &Option<SsaVarId>| flags.is_some_and(|flags| demand.read.contains(&flags));
    match op {
        SsaOp::And {
            dest,
            left,
            right,
            flags,
        }
        | SsaOp::Or {
            dest,
            left,
            right,
            flags,
        }
        | SsaOp::Xor {
            dest,
            left,
            right,
            flags,
        } => {
            if flags_read(flags) {
                return all(&[*left, *right]);
            }
            let bits = result(*dest);
            vec![(*left, bits), (*right, bits)]
        }
        SsaOp::Not {
            dest,
            operand,
            flags,
        } => {
            if flags_read(flags) {
                return all(&[*operand]);
            }
            vec![(*operand, result(*dest))]
        }
        SsaOp::Add {
            dest,
            left,
            right,
            flags,
        }
        | SsaOp::Sub {
            dest,
            left,
            right,
            flags,
        }
        | SsaOp::Mul {
            dest,
            left,
            right,
            flags,
            ..
        } => {
            if flags_read(flags) {
                return all(&[*left, *right]);
            }
            let bits = up_to_highest(result(*dest));
            vec![(*left, bits), (*right, bits)]
        }
        SsaOp::Neg {
            dest,
            operand,
            flags,
        } => {
            if flags_read(flags) {
                return all(&[*operand]);
            }
            vec![(*operand, up_to_highest(result(*dest)))]
        }
        SsaOp::Shl {
            dest,
            value,
            amount,
            flags,
        } => {
            let width = widths.width(*value);
            match (widths.constant_amount(*amount, width), flags_read(flags)) {
                (Some(shift), false) => {
                    vec![(*value, result(*dest) >> shift), (*amount, u64::MAX)]
                }
                _ => all(&[*value, *amount]),
            }
        }
        SsaOp::Shr {
            dest,
            value,
            amount,
            unsigned,
            flags,
        } => {
            let width = widths.width(*value);
            match (
                widths.constant_amount(*amount, width),
                width,
                flags_read(flags),
            ) {
                (Some(shift), Some(width), false) => {
                    let demanded = result(*dest);
                    let mut bits = BitWidth::new(width).low_bits(demanded << shift);
                    // The bits an arithmetic shift fills are copies of the
                    // sign bit.
                    let filled =
                        !(BitWidth::new(width).mask() >> shift) & BitWidth::new(width).mask();
                    if !*unsigned && demanded & filled != 0 {
                        bits |= sign_bit(width);
                    }
                    vec![(*value, bits), (*amount, u64::MAX)]
                }
                _ => all(&[*value, *amount]),
            }
        }
        SsaOp::IntConv {
            dest,
            operand,
            overflow_check: false,
            unsigned,
            ..
        } => match (widths.width(*operand), widths.width(*dest)) {
            (Some(from), Some(to)) => {
                let demanded = result(*dest);
                let mut bits = BitWidth::new(from).low_bits(demanded);
                // A sign extension fills the bits above the operand with
                // its sign bit.
                if !*unsigned && to > from && demanded & !BitWidth::new(from).mask() != 0 {
                    bits |= sign_bit(from);
                }
                vec![(*operand, bits)]
            }
            _ => all(&[*operand]),
        },
        SsaOp::Copy { dest, src } => vec![(*src, result(*dest))],
        // A restatement of the same bits in another shape - a vector read as
        // lanes of another width, a scalar read as the bits it is made of -
        // hands its demand on, in the units the operand counts: the bits the
        // reader takes are the bits the operand holds.
        SsaOp::VectorReinterpret { dest, value, .. } => {
            match widths.restated(*dest, *value, result(*dest)) {
                Some(bits) => vec![(*value, bits)],
                None => all(&[*value]),
            }
        }
        SsaOp::Bitcast { dest, operand, .. } => {
            match widths.restated(*dest, *operand, result(*dest)) {
                Some(bits) => vec![(*operand, bits)],
                None => all(&[*operand]),
            }
        }
        // A shuffle takes the lanes its mask selects, of the inputs it
        // selects them from: a result lane nothing demands selects nothing.
        // A shuffle whose selector is a vector of indices reads every lane of
        // both, since which is not known here.
        SsaOp::VectorShuffle {
            dest,
            left,
            right,
            selector,
        } => {
            let inputs: Vec<SsaVarId> = [Some(*left), *right, selector.index()]
                .into_iter()
                .flatten()
                .collect();
            let Some(mask) = selector.mask() else {
                return all(&inputs);
            };
            let demanded = result(*dest);
            let Some(lane_bits) = widths.lane_bits(*dest, mask.lanes().len()) else {
                return all(&inputs);
            };
            let mut taken: Vec<(SsaVarId, u64)> = Vec::new();
            let mut take = |value: SsaVarId, lane: u32| {
                // An input whose lane the mask cannot state is demanded
                // whole: more than the shuffle takes, never less.
                let span = widths.lane_mask(value, lane, lane_bits).unwrap_or(u64::MAX);
                match taken.iter_mut().find(|(held, _)| *held == value) {
                    Some((_, bits)) => *bits |= span,
                    None => taken.push((value, span)),
                }
            };
            for (index, lane) in (0u32..).zip(mask.lanes()) {
                // A result lane the mask cannot state may be demanded, and
                // then every input is read.
                let Some(covered) = widths.lane_mask(*dest, index, lane_bits) else {
                    return all(&inputs);
                };
                if demanded & covered == 0 {
                    continue;
                }
                match lane {
                    VectorShuffleLane::Left(source) => take(*left, *source),
                    VectorShuffleLane::Right(source) => {
                        if let Some(right) = *right {
                            take(right, *source);
                        }
                    }
                    VectorShuffleLane::Undef | VectorShuffleLane::Zero => {}
                }
            }
            taken
        }
        // An extract takes its lane, an insert everything but the lane it
        // writes, and a splat the scalar it repeats - where anything demands
        // the result at all.
        SsaOp::VectorExtract { dest, vector, lane } => {
            if result(*dest) == 0 {
                return Vec::new();
            }
            // The lane is as wide as the vector states its lanes, not as
            // the destination is: a destination of another width says
            // nothing about which bits the extract takes. Without a shape,
            // or a lane the mask cannot state, the whole vector is read.
            let Some(taken) = widths
                .vector_lane_bits(*vector)
                .and_then(|lane_bits| widths.lane_mask(*vector, *lane, lane_bits))
            else {
                return all(&[*vector]);
            };
            vec![(*vector, taken)]
        }
        SsaOp::VectorInsert {
            dest,
            vector,
            lane,
            value,
        } => {
            let demanded = result(*dest);
            if demanded == 0 {
                return Vec::new();
            }
            // The lane written is as wide as the result states its lanes,
            // not as the value written is. Without a shape, or a lane the
            // mask cannot state, the insert reads the vector and the value
            // both: the value above all, which an empty span would read as
            // unwritten.
            let Some(written) = widths
                .vector_lane_bits(*dest)
                .and_then(|lane_bits| widths.lane_mask(*dest, *lane, lane_bits))
            else {
                return all(&[*vector, *value]);
            };
            let mut taken = Vec::new();
            if demanded & !written != 0 {
                taken.push((*vector, demanded & !written));
            }
            if demanded & written != 0 {
                taken.push((*value, u64::MAX));
            }
            taken
        }
        SsaOp::VectorSplat { dest, value, .. } => {
            if result(*dest) == 0 {
                return Vec::new();
            }
            vec![(*value, u64::MAX)]
        }
        SsaOp::Select {
            dest,
            condition,
            true_val,
            false_val,
        } => {
            let bits = result(*dest);
            vec![
                (*condition, u64::MAX),
                (*true_val, bits),
                (*false_val, bits),
            ]
        }
        _ => op
            .uses()
            .into_iter()
            .map(|value| (value, u64::MAX))
            .collect(),
    }
}

/// The widths of one function's values as demand reads them: in lanes, in
/// the units a value is counted in, and as spans of those units.
#[derive(Clone, Copy)]
struct Widths<'a, T: Target> {
    /// The function whose values are read.
    ssa: &'a SsaFunction<T>,
}

impl<T: Target> Widths<'_, T> {
    /// The width of `value` when it is an integer of 1 to 64 bits.
    fn width(&self, value: SsaVarId) -> Option<u32> {
        self.ssa
            .bit_width(value)
            .filter(|width| (1..=64).contains(width))
    }

    /// How wide one lane of `value` is where it holds `lanes` of them; `None`
    /// where its width is unknown or no multiple of the count.
    ///
    /// This is a shuffle's own lane width, which its mask's length states and
    /// which its inputs share however many lanes each of them holds - not the
    /// lane width of a value's declared shape, which is
    /// [`Self::vector_lane_bits`] and what an operation naming a lane by its index
    /// means.
    fn lane_bits(&self, value: SsaVarId, lanes: usize) -> Option<u32> {
        let lanes = u32::try_from(lanes).ok()?;
        let width = self.ssa.bit_width(value)?;
        let whole = width.checked_rem(lanes)? == 0;
        whole.then(|| width.checked_div(lanes)).flatten()
    }

    /// The width of one lane of `value`, as its own type states its shape;
    /// `None` where it is no vector of a fixed shape. A reader's own width is no
    /// answer to this: [`VectorShape`](crate::target::VectorShape) carries the
    /// invariant that the lanes make up the whole, and a destination or a
    /// written value of another width would slice the vector wrongly.
    fn vector_lane_bits(&self, value: SsaVarId) -> Option<u32> {
        let variable = self.ssa.variable(value)?;
        T::vector_shape(variable.var_type()).map(|shape| shape.lane_bits)
    }

    /// How many bits of a value one bit of its demand mask stands for: one, for
    /// a value of at most 64 bits, and eight - a byte - for one too wide for
    /// that, which puts every vector up to 512 bits in the same 64-bit mask.
    fn unit_bits(&self, value: SsaVarId) -> u32 {
        match self.ssa.bit_width(value) {
            Some(width) if width <= 64 => 1,
            _ => 8,
        }
    }

    /// How many units of its own ([`Self::unit_bits`]) `value` is made of; `None`
    /// where its width is unknown, or where it has more units than a 64-bit mask
    /// can state - a vector wider than 512 bits. A mask cannot speak about such
    /// a value part by part, so every reader of one demands all of it.
    fn units(&self, value: SsaVarId) -> Option<u32> {
        let unit = self.unit_bits(value);
        self.ssa
            .bit_width(value)
            .map(|width| width.div_ceil(unit))
            .filter(|units| (1..=64).contains(units))
    }

    /// The mask of the bits one lane covers in `value`, where its lanes are
    /// `lane_bits` wide ([`Self::span_mask`]); `None` where the lane cannot be stated.
    fn lane_mask(&self, value: SsaVarId, lane: u32, lane_bits: u32) -> Option<u64> {
        if lane_bits == 0 {
            return None;
        }
        let from = lane.checked_mul(lane_bits)?;
        self.span_mask(value, from, from.checked_add(lane_bits)?)
    }

    /// The mask of the bits `[from, to)` of `value`, rounded out to whole units:
    /// demanding more of a value than a reader takes is sound, demanding less is
    /// not. Empty where the span is.
    ///
    /// `None` where the mask cannot state the span, because `value` has units a
    /// 64-bit mask cannot count or the span runs past the ones it can: a caller
    /// must then demand the whole value rather than read an empty mask as a
    /// demand for nothing.
    fn span_mask(&self, value: SsaVarId, from: u32, to: u32) -> Option<u64> {
        let units = self.units(value)?;
        if to <= from {
            return Some(0);
        }
        let unit = self.unit_bits(value).max(1);
        let first = from.checked_div(unit).unwrap_or(0);
        let last = to.div_ceil(unit);
        if last > units {
            return None;
        }
        let span = BitWidth::new(last.saturating_sub(first))
            .mask()
            .checked_shl(first)?;
        Some(BitWidth::new(units).low_bits(span))
    }

    /// `amount` as a constant shift below `width`.
    fn constant_amount(&self, amount: SsaVarId, width: Option<u32>) -> Option<u32> {
        let SsaOp::Const { value, .. } = self.ssa.get_definition(amount)? else {
            return None;
        };
        let shift = u32::try_from(value.as_u64_bits()?).ok()?;
        (shift < width?).then_some(shift)
    }

    /// `demanded`, stated of `held` in its units, restated of `same` - a value
    /// holding the very same bits in another shape; `None` where either value's
    /// units cannot be counted ([`Self::units`]) or the two are not of one width,
    /// in which case nothing here can say which bits of `same` the demand names.
    ///
    /// A unit of `held` covers a span of bits, and that span is the span of
    /// `same` the reader takes: rounded out to `same`'s own units, so a demand
    /// crossing from bits to bytes or back is never understated.
    fn restated(&self, held: SsaVarId, same: SsaVarId, demanded: u64) -> Option<u64> {
        let units = self.units(held)?;
        if self.ssa.bit_width(held)? != self.ssa.bit_width(same)? {
            return None;
        }
        let unit = self.unit_bits(held);
        let mut restated = 0;
        for index in 0..units {
            if demanded & 1u64.checked_shl(index)? == 0 {
                continue;
            }
            let from = index.checked_mul(unit)?;
            restated |= self.span_mask(same, from, from.checked_add(unit)?)?;
        }
        Some(restated)
    }
}

/// The sign bit of an integer `width` bits wide, at least one.
const fn sign_bit(width: u32) -> u64 {
    1 << width.saturating_sub(1)
}

/// Every bit up to and including the highest of `bits`: what a carry from
/// below can reach.
const fn up_to_highest(bits: u64) -> u64 {
    if bits == 0 {
        0
    } else {
        u64::MAX >> bits.leading_zeros()
    }
}

/// Pass that drops the operand of a bitwise operation no reader demands the
/// bits of.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - only replaces instructions,
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct DemandedBitsPass;

impl DemandedBitsPass {
    /// Creates a new demanded-bits pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for DemandedBitsPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "demanded-bits"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Drop the operand of a bitwise operation no reader demands the bits of"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Rewrites every `or`, `xor` and `and` of `ssa` that one of its operands
    /// decides on every bit its readers demand to a copy of that operand.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `_env` - The pass environment, which this pass does not read.
    ///
    /// # Returns
    ///
    /// `true` when at least one operation was rewritten.
    fn run(&self, ssa: &mut SsaFunction<T>, _env: &PassEnv<'_, T>) -> Result<bool> {
        let demand = Demand::of(ssa);
        let constants = ssa.find_constants();
        let rewrites: Vec<(usize, usize, SsaOp<T>)> = ssa
            .blocks()
            .iter()
            .enumerate()
            .flat_map(|(block_idx, block)| {
                block
                    .instructions()
                    .iter()
                    .enumerate()
                    .map(move |(instr_idx, instr)| (block_idx, instr_idx, instr.op()))
            })
            .filter_map(|(block_idx, instr_idx, op)| {
                if let Some(src) = demand.kept_operand(ssa, op, &constants) {
                    let dest = op.dest()?;
                    return Some((block_idx, instr_idx, SsaOp::Copy { dest, src }));
                }
                let shuffle = demand.shuffle_without_unread(ssa, op)?;
                Some((block_idx, instr_idx, shuffle))
            })
            .collect();
        if rewrites.is_empty() {
            return Ok(false);
        }
        Ok(ssa
            .edit(Default::default(), |editor| {
                for (block_idx, instr_idx, op) in rewrites {
                    editor.replace_instruction_op(block_idx, instr_idx, op)?;
                }
                Ok(())
            })
            .is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::NullListener,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// A function of one block, `ops` then a return of `returned`.
    fn function(
        types: &[MockType],
        ops: Vec<SsaOp<MockTarget>>,
        returned: usize,
    ) -> (SsaFunction<MockTarget>, Vec<SsaVarId>) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, types.len());
        let vars: Vec<SsaVarId> = (0u16..)
            .zip(types)
            .map(|(index, ty)| {
                ssa.create_variable(VariableOrigin::Local(index), 0, DefSite::entry(), *ty)
            })
            .collect();
        let mut block = SsaBlock::new(0);
        for op in ops {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(
            vars.get(returned).copied(),
        )));
        ssa.add_block(block);
        ssa.recompute_uses();
        (ssa, vars)
    }

    /// The operation defining `value`.
    fn definition(ssa: &SsaFunction<MockTarget>, value: SsaVarId) -> SsaOp<MockTarget> {
        ssa.get_definition(value).cloned().expect("a definition")
    }

    /// Puts `ops` at the head of the function's one block, before the return
    /// [`function`] left there, and renews its uses: what a test builds once
    /// it knows the variables' ids.
    fn insert(ssa: &mut SsaFunction<MockTarget>, ops: Vec<SsaOp<MockTarget>>) {
        let block = ssa.block_mut(0).expect("the block");
        for (at, op) in ops.into_iter().enumerate() {
            block
                .instructions_mut()
                .insert(at, SsaInstruction::synthetic(op));
        }
        ssa.recompute_uses();
    }

    /// The x86 `setg cl; and eax, ecx; movzx eax, al` shape: the low byte
    /// written over a register the function was entered with, an `and` at 32
    /// bits, and the low byte of that kept. `(c & ~0xff) | b` is `b` to the
    /// truncation, so the `or` is a copy of `b`, and `c` is read by nothing
    /// that reaches the result.
    #[test]
    fn a_kept_term_no_reader_demands_is_dropped() {
        // v0 = entry c, v1 = b (one byte, zero-extended), v2 = ~0xff, v3 = c &
        // ~0xff, v4 = v3 | v1, v5 = (u8)v4, v6 = (u32)v5.
        let types = [
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U8,
            MockType::U32,
        ];
        let (mut ssa, v) = function(&types, Vec::new(), 6);
        let ops = vec![
            SsaOp::Const {
                dest: v[2],
                value: ConstValue::U32(!0xff),
            },
            SsaOp::And {
                dest: v[3],
                left: v[0],
                right: v[2],
                flags: None,
            },
            SsaOp::Or {
                dest: v[4],
                left: v[3],
                right: v[1],
                flags: None,
            },
            SsaOp::IntConv {
                dest: v[5],
                operand: v[4],
                target: MockType::U8,
                overflow_check: false,
                unsigned: true,
            },
            SsaOp::IntConv {
                dest: v[6],
                operand: v[5],
                target: MockType::U32,
                overflow_check: false,
                unsigned: true,
            },
        ];
        let block = ssa.block_mut(0).expect("the block");
        for (at, op) in ops.into_iter().enumerate() {
            block
                .instructions_mut()
                .insert(at, SsaInstruction::synthetic(op));
        }
        ssa.recompute_uses();

        assert!(
            DemandedBitsPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true),
            "{ssa}"
        );
        assert_eq!(
            definition(&ssa, v[4]),
            SsaOp::Copy {
                dest: v[4],
                src: v[1]
            },
            "{ssa}"
        );
    }

    /// The control: read whole, `(c & ~0xff) | b` keeps `c`'s bits, and the
    /// `or` stays.
    #[test]
    fn a_term_a_reader_demands_is_kept() {
        let types = [
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
        ];
        let (mut ssa, v) = function(&types, Vec::new(), 4);
        let ops = vec![
            SsaOp::Const {
                dest: v[2],
                value: ConstValue::U32(!0xff),
            },
            SsaOp::And {
                dest: v[3],
                left: v[0],
                right: v[2],
                flags: None,
            },
            SsaOp::Or {
                dest: v[4],
                left: v[3],
                right: v[1],
                flags: None,
            },
        ];
        let block = ssa.block_mut(0).expect("the block");
        for (at, op) in ops.into_iter().enumerate() {
            block
                .instructions_mut()
                .insert(at, SsaInstruction::synthetic(op));
        }
        ssa.recompute_uses();

        assert!(
            !(DemandedBitsPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true)),
            "{ssa}"
        );
        assert!(matches!(definition(&ssa, v[4]), SsaOp::Or { .. }), "{ssa}");
    }

    /// An `and` with a constant that keeps every bit its readers demand is its
    /// operand: `(x & 0xffff)` read as a byte is `x` there.
    #[test]
    fn a_mask_keeping_every_demanded_bit_is_dropped() {
        let types = [MockType::U32, MockType::U32, MockType::U32, MockType::U8];
        let (mut ssa, v) = function(&types, Vec::new(), 3);
        let ops = vec![
            SsaOp::Const {
                dest: v[1],
                value: ConstValue::U32(0xffff),
            },
            SsaOp::And {
                dest: v[2],
                left: v[0],
                right: v[1],
                flags: None,
            },
            SsaOp::IntConv {
                dest: v[3],
                operand: v[2],
                target: MockType::U8,
                overflow_check: false,
                unsigned: true,
            },
        ];
        let block = ssa.block_mut(0).expect("the block");
        for (at, op) in ops.into_iter().enumerate() {
            block
                .instructions_mut()
                .insert(at, SsaInstruction::synthetic(op));
        }
        ssa.recompute_uses();

        assert!(
            DemandedBitsPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true),
            "{ssa}"
        );
        assert_eq!(
            definition(&ssa, v[2]),
            SsaOp::Copy {
                dest: v[2],
                src: v[0]
            },
            "{ssa}"
        );
    }

    /// A right shift by 8 of `(c & ~0xff) | b`, read whole, reads `c`'s bits:
    /// the `or` stays, and the `and`, whose mask keeps every bit the shift
    /// reads, is `c` there.
    #[test]
    fn demand_follows_shifts() {
        // v4 = (c & ~0xff) | b; v6 = v4 >> 8 (unsigned), returned whole.
        let types = [
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
        ];
        let (mut ssa, v) = function(&types, Vec::new(), 6);
        let ops = vec![
            SsaOp::Const {
                dest: v[2],
                value: ConstValue::U32(!0xff),
            },
            SsaOp::And {
                dest: v[3],
                left: v[0],
                right: v[2],
                flags: None,
            },
            SsaOp::Or {
                dest: v[4],
                left: v[3],
                right: v[1],
                flags: None,
            },
            SsaOp::Const {
                dest: v[5],
                value: ConstValue::U32(8),
            },
            SsaOp::Shr {
                dest: v[6],
                value: v[4],
                amount: v[5],
                unsigned: true,
                flags: None,
            },
        ];
        let block = ssa.block_mut(0).expect("the block");
        for (at, op) in ops.into_iter().enumerate() {
            block
                .instructions_mut()
                .insert(at, SsaInstruction::synthetic(op));
        }
        ssa.recompute_uses();

        assert!(
            DemandedBitsPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true),
            "{ssa}"
        );
        assert!(
            matches!(definition(&ssa, v[4]), SsaOp::Or { .. }),
            "the shift reads c's bits: {ssa}"
        );
        assert_eq!(
            definition(&ssa, v[3]),
            SsaOp::Copy {
                dest: v[3],
                src: v[0]
            },
            "{ssa}"
        );
    }

    /// An `or` whose flags are read reads every bit of its operands: the flags
    /// say whether the whole result is zero.
    #[test]
    fn an_operation_whose_flags_are_read_keeps_its_operands() {
        let types = [
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U32,
            MockType::U8,
            MockType::U32,
            MockType::U32,
        ];
        let (mut ssa, v) = function(&types, Vec::new(), 7);
        let ops = vec![
            SsaOp::Const {
                dest: v[2],
                value: ConstValue::U32(!0xff),
            },
            SsaOp::And {
                dest: v[3],
                left: v[0],
                right: v[2],
                flags: None,
            },
            SsaOp::Or {
                dest: v[4],
                left: v[3],
                right: v[1],
                flags: Some(v[6]),
            },
            SsaOp::IntConv {
                dest: v[5],
                operand: v[4],
                target: MockType::U8,
                overflow_check: false,
                unsigned: true,
            },
            SsaOp::Copy {
                dest: v[7],
                src: v[6],
            },
        ];
        let block = ssa.block_mut(0).expect("the block");
        for (at, op) in ops.into_iter().enumerate() {
            block
                .instructions_mut()
                .insert(at, SsaInstruction::synthetic(op));
        }
        ssa.recompute_uses();

        assert!(
            !(DemandedBitsPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true)),
            "{ssa}"
        );
    }
    /// **A merge stops reading the input no demanded lane selects.** The
    /// shape a partial vector write lifts to: two lanes written, the other
    /// two kept from what the register held - ARM's `Dj` of a `Qi`, an x86
    /// legacy-SSE write of an `xmm`. Reading only a written lane leaves the
    /// kept lanes undefined to every reader, so the register's previous value
    /// is no longer an input and the register stops being read at all;
    /// reading only a kept lane drops the written value the same way.
    ///
    /// The control: reading one lane of each keeps both inputs, so the pass
    /// is answering the demand and not merely rewriting every merge it meets.
    #[test]
    fn a_merge_stops_reading_the_input_no_demanded_lane_selects() {
        let lanes = |mask: [VectorShuffleLane; 4]| {
            VectorShuffleSelector::Mask(VectorShuffleMask::new(mask.to_vec()))
        };
        // The lanes read, and what the merge is left with: the written value
        // alone, the previous value alone, or both.
        for (read, expected) in [
            (&[3u32][..], Some(1usize)),
            (&[0][..], Some(0)),
            (&[0, 3][..], None),
        ] {
            let types = [
                MockType::V4I32,
                MockType::V4I32,
                MockType::V4I32,
                MockType::I32,
                MockType::I32,
                MockType::I32,
            ];
            // The function returns what every extract reaches, so that every
            // lane the case reads is a lane something demands: one extract
            // returned, two summed and the sum returned.
            let returned = if read.len() > 1 { 5 } else { 3 };
            let (mut ssa, v) = function(&types, Vec::new(), returned);
            let mut ops = vec![SsaOp::VectorShuffle {
                dest: v[2],
                left: v[0],
                right: Some(v[1]),
                selector: lanes([
                    VectorShuffleLane::Left(0),
                    VectorShuffleLane::Left(1),
                    VectorShuffleLane::Right(2),
                    VectorShuffleLane::Right(3),
                ]),
            }];
            for (at, lane) in read.iter().enumerate() {
                ops.push(SsaOp::VectorExtract {
                    dest: v[3usize.saturating_add(at)],
                    vector: v[2],
                    lane: *lane,
                });
            }
            if read.len() > 1 {
                ops.push(SsaOp::Add {
                    dest: v[5],
                    left: v[3],
                    right: v[4],
                    flags: None,
                });
            }
            insert(&mut ssa, ops);

            assert_eq!(
                DemandedBitsPass.run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                ) == Ok(true),
                expected.is_some(),
                "reading {read:?}: {ssa}"
            );
            let merged = definition(&ssa, v[2]);
            let SsaOp::VectorShuffle {
                left,
                right,
                ref selector,
                ..
            } = merged
            else {
                panic!("reading {read:?}: {merged:?}");
            };
            let Some(kept) = expected else {
                assert_eq!(
                    (left, right),
                    (v[0], Some(v[1])),
                    "the control: both inputs stand"
                );
                continue;
            };
            assert_eq!(
                (left, right),
                (v[kept], None),
                "reading {read:?} leaves input {kept}: {ssa}"
            );
            let standing = if kept == 1 {
                [
                    VectorShuffleLane::Zero,
                    VectorShuffleLane::Zero,
                    VectorShuffleLane::Left(2),
                    VectorShuffleLane::Left(3),
                ]
            } else {
                [
                    VectorShuffleLane::Left(0),
                    VectorShuffleLane::Left(1),
                    VectorShuffleLane::Zero,
                    VectorShuffleLane::Zero,
                ]
            };
            assert_eq!(
                selector.mask().expect("a fixed mask").lanes(),
                standing,
                "the lanes of the input that went read zero: {ssa}"
            );
        }
    }

    /// **A merge whose lanes the mask cannot state keeps every input.** The
    /// demand mask has 64 bits, one per byte of a value too wide for a bit
    /// each, so it can speak about a vector of at most 512 bits and about a
    /// lane whose width is known. Where it cannot - an x86 AMX tile is 16
    /// rows of 64 bytes, 1024 of them, and a reader may take a lane of no
    /// stated width - an unanswerable lane must read as "may be demanded":
    /// reading it as "demanded of nothing" drops an input the result still
    /// carries.
    ///
    /// The control: a merge every lane of which the mask does state loses
    /// the input no demanded lane selects, so the pass is answering the
    /// demand and not merely keeping every merge it meets.
    #[test]
    fn a_merge_whose_lanes_the_mask_cannot_state_keeps_every_input() {
        let mask = |lanes: Vec<VectorShuffleLane>| {
            VectorShuffleSelector::Mask(VectorShuffleMask::new(lanes))
        };
        // A vector of more bytes than the mask has bits, merged from two of
        // its kind and returned whole: every byte of both inputs is read,
        // though the mask can only speak about the first 64.
        let wide = (0u32..128)
            .map(|lane| {
                if lane < 64 {
                    VectorShuffleLane::Left(lane)
                } else {
                    VectorShuffleLane::Right(lane)
                }
            })
            .collect();
        let (mut ssa, v) = function(
            &[MockType::V128I8, MockType::V128I8, MockType::V128I8],
            Vec::new(),
            2,
        );
        insert(
            &mut ssa,
            vec![SsaOp::VectorShuffle {
                dest: v[2],
                left: v[0],
                right: Some(v[1]),
                selector: mask(wide),
            }],
        );
        assert!(
            !(DemandedBitsPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            ) == Ok(true)),
            "a vector wider than the mask: {ssa}"
        );
        let SsaOp::VectorShuffle { left, right, .. } = definition(&ssa, v[2]) else {
            panic!("the merge stands");
        };
        assert_eq!(
            (left, right),
            (v[0], Some(v[1])),
            "both inputs stand: {ssa}"
        );

        // A 128-bit merge read twice: once by a lane of known width, once by
        // a lane of none. The second says nothing about which lanes it
        // takes, so it takes them all.
        let four = || {
            vec![
                VectorShuffleLane::Left(0),
                VectorShuffleLane::Left(1),
                VectorShuffleLane::Right(2),
                VectorShuffleLane::Right(3),
            ]
        };
        for (lane_type, second_lane, dropped) in [
            (MockType::Unknown, 3u32, None),
            (MockType::I32, 1, Some(0usize)),
        ] {
            let (mut ssa, v) = function(
                &[
                    MockType::V4I32,
                    MockType::V4I32,
                    MockType::V4I32,
                    MockType::I32,
                    lane_type,
                    lane_type,
                ],
                Vec::new(),
                5,
            );
            insert(
                &mut ssa,
                vec![
                    SsaOp::VectorShuffle {
                        dest: v[2],
                        left: v[0],
                        right: Some(v[1]),
                        selector: mask(four()),
                    },
                    SsaOp::VectorExtract {
                        dest: v[3],
                        vector: v[2],
                        lane: 0,
                    },
                    SsaOp::VectorExtract {
                        dest: v[4],
                        vector: v[2],
                        lane: second_lane,
                    },
                    SsaOp::Add {
                        dest: v[5],
                        left: v[3],
                        right: v[4],
                        flags: None,
                    },
                ],
            );
            assert_eq!(
                DemandedBitsPass.run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                ) == Ok(true),
                dropped.is_some(),
                "{lane_type}: {ssa}"
            );
            let SsaOp::VectorShuffle { left, right, .. } = definition(&ssa, v[2]) else {
                panic!("{lane_type}: the merge stands");
            };
            let Some(kept) = dropped else {
                assert_eq!(
                    (left, right),
                    (v[0], Some(v[1])),
                    "a lane of no stated width reads both inputs: {ssa}"
                );
                continue;
            };
            assert_eq!(
                (left, right),
                (v[kept], None),
                "the control: the input no demanded lane selects goes: {ssa}"
            );
        }
    }

    /// **An insert whose lane the mask cannot state demands the value it
    /// writes.** The lane an insert writes is where it reads its value, so a
    /// lane the mask cannot state - one past the 64 bytes it counts - must
    /// read as "may be demanded" here too: read as unwritten, the value
    /// becomes one nothing demands, and whatever computed it is rewritten as
    /// if its result went nowhere.
    ///
    /// The control: an insert whose lane the mask does state, into a result
    /// read only elsewhere, demands nothing of the value - the pass answers
    /// the demand rather than giving up on every insert.
    #[test]
    fn an_insert_whose_lane_the_mask_cannot_state_demands_the_value_it_writes() {
        // `x | 0` is `x` on every bit, so the `or` folds to a copy wherever
        // its result is demanded at all: whether it does says whether the
        // insert demanded the value it writes.
        for (vector_type, lane, read_lane, folds) in [
            // A vector of more bytes than the mask has bits: the written
            // lane is past the ones it counts.
            (MockType::V128I8, 100u32, None, true),
            // The control: a lane the mask states, written into a result
            // nothing reads there.
            (MockType::V4I32, 1, Some(0u32), false),
        ] {
            let scalar = if vector_type == MockType::V128I8 {
                MockType::U8
            } else {
                MockType::U32
            };
            let (mut ssa, v) = function(
                &[vector_type, scalar, scalar, scalar, vector_type, scalar],
                Vec::new(),
                if read_lane.is_some() { 5 } else { 4 },
            );
            let mut ops = vec![
                SsaOp::Const {
                    dest: v[2],
                    value: if scalar == MockType::U8 {
                        ConstValue::U8(0)
                    } else {
                        ConstValue::U32(0)
                    },
                },
                SsaOp::Or {
                    dest: v[3],
                    left: v[1],
                    right: v[2],
                    flags: None,
                },
                SsaOp::VectorInsert {
                    dest: v[4],
                    vector: v[0],
                    lane,
                    value: v[3],
                },
            ];
            if let Some(read_lane) = read_lane {
                ops.push(SsaOp::VectorExtract {
                    dest: v[5],
                    vector: v[4],
                    lane: read_lane,
                });
            }
            insert(&mut ssa, ops);
            assert_eq!(
                DemandedBitsPass.run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                ) == Ok(true),
                folds,
                "{vector_type} lane {lane}: {ssa}"
            );
            let written = definition(&ssa, v[3]);
            if folds {
                assert!(
                    matches!(written, SsaOp::Copy { src, .. } if src == v[1]),
                    "the value the insert writes is demanded: {written:?} in {ssa}"
                );
            } else {
                assert!(
                    matches!(written, SsaOp::Or { .. }),
                    "the control: no demanded lane holds the value: {written:?} in {ssa}"
                );
            }
        }
    }

    /// **A restatement of the same bits hands the demand on.** The shape the
    /// ARM case has: a partial write of a vector register merges the written
    /// lanes with the register's previous value at the width of the write,
    /// the result is restated as the register's own cell, and a later read
    /// states its own view of it again. Every link is a
    /// [`SsaOp::VectorReinterpret`] of the same bits in another lane width,
    /// so a demand that stops there reads the whole register as read - and
    /// at a function's entry that is a parameter the function never had.
    ///
    /// The control is the pair: which input survives follows which lane the
    /// restated reader takes, so the pass is carrying the demand across the
    /// restatement and not simply dropping an input.
    #[test]
    fn a_restatement_of_the_same_bits_hands_the_demand_on() {
        for (read, kept, standing) in [
            (
                1u32,
                1usize,
                [
                    VectorShuffleLane::Zero,
                    VectorShuffleLane::Zero,
                    VectorShuffleLane::Left(2),
                    VectorShuffleLane::Left(3),
                ],
            ),
            (
                0,
                0,
                [
                    VectorShuffleLane::Left(0),
                    VectorShuffleLane::Left(1),
                    VectorShuffleLane::Zero,
                    VectorShuffleLane::Zero,
                ],
            ),
        ] {
            // v0 the register's previous value, v1 the written lanes, v2 the
            // merge at the width of the write, v3 the same bits as the
            // register's two 64-bit lanes, v4 the one lane read.
            let (mut ssa, v) = function(
                &[
                    MockType::V4I32,
                    MockType::V4I32,
                    MockType::V4I32,
                    MockType::V2F64,
                    MockType::F64,
                ],
                Vec::new(),
                4,
            );
            insert(
                &mut ssa,
                vec![
                    SsaOp::VectorShuffle {
                        dest: v[2],
                        left: v[0],
                        right: Some(v[1]),
                        selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                            VectorShuffleLane::Left(0),
                            VectorShuffleLane::Left(1),
                            VectorShuffleLane::Right(2),
                            VectorShuffleLane::Right(3),
                        ])),
                    },
                    SsaOp::VectorReinterpret {
                        dest: v[3],
                        value: v[2],
                        target_type: MockType::V2F64,
                    },
                    SsaOp::VectorExtract {
                        dest: v[4],
                        vector: v[3],
                        lane: read,
                    },
                ],
            );
            assert!(
                DemandedBitsPass.run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                ) == Ok(true),
                "reading lane {read}: {ssa}"
            );
            let SsaOp::VectorShuffle {
                left,
                right,
                ref selector,
                ..
            } = definition(&ssa, v[2])
            else {
                panic!("reading lane {read}: the merge stands");
            };
            assert_eq!(
                (left, right),
                (v[kept], None),
                "the 64-bit lane {read} is the 32-bit lanes of input {kept}: {ssa}"
            );
            assert_eq!(
                selector.mask().expect("a fixed mask").lanes(),
                standing,
                "the lanes of the input that went read zero: {ssa}"
            );
        }
    }

    /// **A lane is as wide as its vector states its lanes.** A reader's own
    /// width is no answer to how wide a lane is: an extract whose
    /// destination is narrower than the lane - which the verifier rejects,
    /// and which this pass must not act on wrongly all the same - would
    /// otherwise name a span low in the vector and drop whichever input the
    /// lanes there happen to select.
    ///
    /// The control: the same extract at the lane's own width answers the
    /// same, so the shape is what the demand follows.
    #[test]
    fn a_lane_is_as_wide_as_its_vector_states() {
        for extracted in [MockType::I8, MockType::I32] {
            let (mut ssa, v) = function(
                &[MockType::V4I32, MockType::V4I32, MockType::V4I32, extracted],
                Vec::new(),
                3,
            );
            insert(
                &mut ssa,
                vec![
                    SsaOp::VectorShuffle {
                        dest: v[2],
                        left: v[0],
                        right: Some(v[1]),
                        selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                            VectorShuffleLane::Left(0),
                            VectorShuffleLane::Left(1),
                            VectorShuffleLane::Right(2),
                            VectorShuffleLane::Right(3),
                        ])),
                    },
                    // Lane 3 holds the right input, whatever the
                    // destination's width says.
                    SsaOp::VectorExtract {
                        dest: v[3],
                        vector: v[2],
                        lane: 3,
                    },
                ],
            );
            assert!(
                DemandedBitsPass.run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                ) == Ok(true),
                "extracting into {extracted}: {ssa}"
            );
            let SsaOp::VectorShuffle { left, right, .. } = definition(&ssa, v[2]) else {
                panic!("extracting into {extracted}: the merge stands");
            };
            assert_eq!(
                (left, right),
                (v[1], None),
                "lane 3 is the right input's, so the left one goes: {ssa}"
            );
        }
    }

    /// **The top lane of a vector the mask just fits is demanded.** A
    /// 512-bit vector is 64 bytes, exactly the bits the mask has, so its
    /// last lane sits at the last unit the mask can state: the span of that
    /// lane must come out as the top bits of the mask and not as the empty
    /// span of a lane past it, which would read as a lane nobody demands and
    /// drop the input that lane selects.
    ///
    /// The control is the pair: reading the low lane instead drops the other
    /// input, so the answer follows the lane read at both ends of the mask.
    #[test]
    fn the_top_lane_of_a_vector_the_mask_just_fits_is_demanded() {
        for (read, kept) in [(1u32, 1usize), (0, 0)] {
            let (mut ssa, v) = function(
                &[
                    MockType::V2I256,
                    MockType::V2I256,
                    MockType::V2I256,
                    MockType::V2I256,
                ],
                Vec::new(),
                3,
            );
            insert(
                &mut ssa,
                vec![
                    SsaOp::VectorShuffle {
                        dest: v[2],
                        left: v[0],
                        right: Some(v[1]),
                        selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                            VectorShuffleLane::Left(0),
                            VectorShuffleLane::Right(1),
                        ])),
                    },
                    // The one lane anything reads: the 256-bit lane `read`,
                    // which is bytes 0..31 or bytes 32..63 of the merge.
                    SsaOp::VectorShuffle {
                        dest: v[3],
                        left: v[2],
                        right: None,
                        selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                            VectorShuffleLane::Left(read),
                            VectorShuffleLane::Zero,
                        ])),
                    },
                ],
            );
            assert!(
                DemandedBitsPass.run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
                ) == Ok(true),
                "reading lane {read}: {ssa}"
            );
            let merged = definition(&ssa, v[2]);
            let SsaOp::VectorShuffle { left, right, .. } = merged else {
                panic!("reading lane {read}: the merge stands, not {merged:?}");
            };
            assert_eq!(
                (left, right),
                (v[kept], None),
                "lane {read} of the merge is input {kept}'s: {ssa}"
            );
        }
    }
}
