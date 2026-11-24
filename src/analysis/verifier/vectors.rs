//! The vector checks of an operation: lane shapes, masks and bit storage.

use crate::{
    analysis::verifier::{Check, RegionExtent, VerifierError, operations::OperationScan},
    ir::{
        ops::{SsaOp, VectorCastKind, VectorRounding, VectorShuffleSelector},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{Target, VectorDescriptor, VectorLaneKind, VectorMaskDescriptor, VectorMaskShape},
};

impl<T: Target> OperationScan<'_, '_, T> {
    /// Checks vector operation shape consistency when operand types are known.
    ///
    /// Every check is conditioned on the operand type resolving to a known
    /// vector or mask descriptor: types are frequently unknown on lifted code,
    /// and a `None` shape means "cannot tell", not "wrong", so an unknown
    /// operand is skipped rather than rejected. Within that guard the match
    /// enforces the per-op shape rules - operand shapes agreeing with each other
    /// and with the destination, lane indices in bounds, element types matching
    /// a descriptor's lane type, mask lane counts matching their vector, and
    /// reinterpret preserving total bit width - reporting each mismatch as an
    /// [`VerifierError::InvalidVectorOperation`].
    pub fn check_vector_operation(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        // A lane operation computes on the lanes its element states: where its
        // operand's type lays out lanes of another width, one of the two is
        // wrong. A type of one lane states the register, not its lanes.
        if let Some((operand, element)) = op.vector_element()
            && element.bits != 0
            && !element.scalar
            && let Some(VectorDescriptor::Fixed(shape)) = self.var_vector_shape(operand)
            && shape.lane_count > 1
            && shape.lane_bits != element.bits
        {
            self.invalid_vector(
                block_idx,
                instr_idx,
                "the element's lane width differs from its operand's lanes",
            );
        }
        match op {
            // A computation of a whole value states the interface its kernel
            // fixes; one whose operands its row states has nothing more to
            // check here.
            // A contraction reads the accumulator its kind adds to and the two
            // factors, and defines one result.
            SsaOp::Contract(data) => {
                if data.inputs.len() != data.kind.input_count() || data.outputs.len() != 1 {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "contraction interface differs from its kind",
                    );
                }
            }
            SsaOp::Compute(data) if data.kind.lanes.is_none() => {
                let kind = data.kind;
                if !kind.is_valid()
                    || kind.input_count() != data.inputs.len()
                    || kind.output_count() != data.outputs.len()
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "computation interface differs from its kernel",
                    );
                }
            }
            SsaOp::Compute(data) => {
                let kind = data.kind;
                if !kind.is_valid()
                    || kind.input_count() != data.inputs.len()
                    || kind.output_count() != data.outputs.len()
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "controlled vector interface differs from its kernel",
                    );
                    return;
                }
                for &value in data
                    .inputs
                    .iter()
                    .skip(kind.control_input_count())
                    .chain(data.outputs.iter())
                {
                    self.check_vector_bit_storage(block_idx, instr_idx, value);
                }
                for &value in data.inputs.iter().take(2) {
                    self.check_region_integer(block_idx, instr_idx, value);
                }
                if matches!(
                    kind.rounding,
                    VectorRounding::FloatingEnvironment | VectorRounding::FixedEnvironment
                ) && let Some(&environment) = data
                    .inputs
                    .get(2usize.saturating_add(usize::from(kind.inactive().is_some())))
                {
                    self.check_region_integer(block_idx, instr_idx, environment);
                }
                if kind.mask_tail()
                    && let Some(&limit) = data
                        .inputs
                        .get(kind.control_input_count().saturating_sub(1))
                {
                    self.check_region_integer(block_idx, instr_idx, limit);
                }
                if kind.inactive().is_some()
                    && let Some(ty) = data.inputs.get(2).and_then(|&value| self.var_type(value))
                    && !T::is_unknown(ty)
                    && !T::is_integer(ty)
                    && T::vector_mask_descriptor(ty).is_none()
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "controlled vector predicate is not a packed bit mask",
                    );
                }
                if kind.inactive().is_some()
                    && let Some(capacity) = data
                        .inputs
                        .get(2)
                        .and_then(|&value| self.var_type(value))
                        .and_then(RegionExtent::of::<T>)
                {
                    let stride = u128::from(kind.predicate_bit(1).unwrap_or(1));
                    let values = data.outputs.iter().take(kind.data_outputs()).chain(
                        data.inputs.iter().skip(
                            kind.control_input_count()
                                .saturating_add(kind.passthrough_count()),
                        ),
                    );
                    for value in values {
                        if let Some(lanes) =
                            self.var_vector_shape(*value).and_then(RegionExtent::lanes)
                            && !lanes
                                .fixed
                                .checked_mul(stride)
                                .zip(lanes.scaled.checked_mul(stride))
                                .and_then(|(fixed, scaled)| {
                                    RegionExtent::new(fixed, scaled, lanes.divisor)
                                })
                                .is_some_and(|required| capacity.covers(required))
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "controlled vector predicate cannot cover its data elements",
                            );
                            break;
                        }
                    }
                }
                for (&output, &old) in data
                    .outputs
                    .iter()
                    .take(kind.passthrough_count())
                    .zip(data.inputs.iter().skip(kind.control_input_count()))
                {
                    if let Some(ty) = self.var_type(old).cloned() {
                        self.check_var_type(
                            block_idx,
                            instr_idx,
                            output,
                            &ty,
                            "controlled vector passthrough differs from its result layout",
                        );
                    }
                }
            }
            SsaOp::VectorSlice {
                dest,
                offset,
                length,
                target_type,
                ..
            } => {
                self.check_region_integer(block_idx, instr_idx, *offset);
                self.check_region_integer(block_idx, instr_idx, *length);
                if !T::is_unknown(target_type) {
                    self.check_var_type(
                        block_idx,
                        instr_idx,
                        *dest,
                        target_type,
                        "slice destination differs from its target layout",
                    );
                }
            }
            SsaOp::VectorInsertSlice {
                dest,
                vector,
                offset,
                target_type,
                ..
            } => {
                self.check_region_integer(block_idx, instr_idx, *offset);
                if !T::is_unknown(target_type) {
                    self.check_var_type(
                        block_idx,
                        instr_idx,
                        *dest,
                        target_type,
                        "insertion destination differs from its target layout",
                    );
                }
                let source = self.var_type(*vector).and_then(RegionExtent::of::<T>);
                let target = RegionExtent::of::<T>(target_type);
                if let (Some(source), Some(target)) = (source, target)
                    && source != target
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "insertion changes the base storage width",
                    );
                }
            }
            SsaOp::VectorConcat {
                dest,
                left,
                right,
                target_type,
            } => {
                if !T::is_unknown(target_type) {
                    self.check_var_type(
                        block_idx,
                        instr_idx,
                        *dest,
                        target_type,
                        "concatenation destination differs from its target layout",
                    );
                }
                let source = self
                    .var_type(*left)
                    .and_then(RegionExtent::of::<T>)
                    .and_then(|left| left.plus(RegionExtent::of::<T>(self.var_type(*right)?)?));
                let target = RegionExtent::of::<T>(target_type);
                if let (Some(source), Some(target)) = (source, target)
                    && source != target
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "concatenation width differs from the sum of its parts",
                    );
                }
            }
            // A vector constant is laid out in its variable's lanes, which an
            // operation that states no lanes of its own reads it by.
            SsaOp::Const {
                dest,
                value: ConstValue::Vector(vector),
            } => {
                if let Some(VectorDescriptor::Fixed(shape)) = self.var_vector_shape(*dest)
                    && (shape.lane_bits, shape.lane_count) != (vector.lane_bits(), vector.lanes())
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "a vector constant's lanes differ from its type's",
                    );
                }
            }
            SsaOp::VectorBinary {
                dest, left, right, ..
            }
            | SsaOp::VectorCompare {
                dest, left, right, ..
            } => {
                let left_shape = self.var_vector_shape(*left);
                let right_shape = self.var_vector_shape(*right);
                if let (Some(left_shape), Some(right_shape)) = (left_shape, right_shape)
                    && left_shape != right_shape
                {
                    self.invalid_vector(block_idx, instr_idx, "operand vector shapes differ");
                }
                if matches!(op, SsaOp::VectorBinary { .. }) {
                    self.check_shape(block_idx, instr_idx, *dest, left_shape);
                } else if let Some(shape) = left_shape {
                    self.check_mask_dest_shape(
                        block_idx,
                        instr_idx,
                        *dest,
                        shape.mask_descriptor(),
                    );
                }
            }
            SsaOp::VectorUnary { dest, value, .. } => {
                self.check_shape(block_idx, instr_idx, *dest, self.var_vector_shape(*value));
            }
            SsaOp::VectorTernary {
                dest,
                first,
                second,
                third,
                ..
            } => {
                let a = self.var_vector_shape(*first);
                let b = self.var_vector_shape(*second);
                let c = self.var_vector_shape(*third);
                if let (Some(a), Some(b), Some(c)) = (a, b, c)
                    && (a != b || a != c)
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "ternary operand vector shapes differ",
                    );
                }
                self.check_shape(block_idx, instr_idx, *dest, a);
            }
            SsaOp::VectorLoad {
                dest, vector_type, ..
            }
            | SsaOp::VectorBroadcastLoad {
                dest, vector_type, ..
            } => {
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "load vector_type is not a supported vector",
                    );
                }
                self.check_shape(block_idx, instr_idx, *dest, shape);
            }
            SsaOp::VectorMaskedLoad {
                dest,
                mask,
                passthrough,
                vector_type,
                ..
            }
            | SsaOp::VectorGather {
                dest,
                mask,
                passthrough,
                vector_type,
                ..
            } => {
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "masked load vector_type is not a supported vector",
                    );
                }
                self.check_shape(block_idx, instr_idx, *dest, shape);
                if let Some(shape) = shape {
                    self.check_mask_source_shape(
                        block_idx,
                        instr_idx,
                        *mask,
                        shape.mask_descriptor(),
                    );
                    if let Some(passthrough) = passthrough {
                        self.check_shape(block_idx, instr_idx, *passthrough, Some(shape));
                    }
                }
            }
            SsaOp::VectorFaultingLoad {
                dest,
                fault,
                mask,
                passthrough,
                vector_type,
                ..
            } => {
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "faulting load vector_type is not a supported vector",
                    );
                }
                self.check_shape(block_idx, instr_idx, *dest, shape);
                if let Some(shape) = shape {
                    let mask_shape = shape.mask_descriptor();
                    if let Some(mask) = mask {
                        self.check_mask_source_shape(block_idx, instr_idx, *mask, mask_shape);
                    }
                    if let Some(fault) = fault {
                        self.check_mask_dest_shape(block_idx, instr_idx, *fault, mask_shape);
                    }
                    if let Some(passthrough) = passthrough {
                        self.check_shape(block_idx, instr_idx, *passthrough, Some(shape));
                    }
                }
            }
            SsaOp::VectorSegmentLoad {
                dests,
                mask,
                vector_type,
                segments,
                ..
            } => {
                if *segments == 0 {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "segment load must load at least one segment",
                    );
                }
                if usize::try_from(*segments).ok() != Some(dests.len()) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "segment load destination count does not match segments",
                    );
                }
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "segment load vector_type is not a supported vector",
                    );
                }
                for dest in dests {
                    self.check_shape(block_idx, instr_idx, *dest, shape);
                }
                if let (Some(mask), Some(shape)) = (mask, shape) {
                    self.check_mask_source_shape(
                        block_idx,
                        instr_idx,
                        *mask,
                        shape.mask_descriptor(),
                    );
                }
            }
            SsaOp::VectorStore {
                value, vector_type, ..
            } => {
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "store vector_type is not a supported vector",
                    );
                }
                self.check_shape(block_idx, instr_idx, *value, shape);
            }
            SsaOp::VectorMaskedStore {
                value,
                mask,
                vector_type,
                ..
            }
            | SsaOp::VectorScatter {
                value,
                mask,
                vector_type,
                ..
            } => {
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "masked store vector_type is not a supported vector",
                    );
                }
                self.check_shape(block_idx, instr_idx, *value, shape);
                if let Some(shape) = shape {
                    self.check_mask_source_shape(
                        block_idx,
                        instr_idx,
                        *mask,
                        shape.mask_descriptor(),
                    );
                }
            }
            SsaOp::VectorSegmentStore {
                values,
                mask,
                vector_type,
                segments,
                ..
            } => {
                if *segments == 0 {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "segment store must store at least one segment",
                    );
                }
                if usize::try_from(*segments).ok() != Some(values.len()) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "segment store value count does not match segments",
                    );
                }
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "segment store vector_type is not a supported vector",
                    );
                }
                for value in values {
                    self.check_shape(block_idx, instr_idx, *value, shape);
                }
                if let (Some(mask), Some(shape)) = (mask, shape) {
                    self.check_mask_source_shape(
                        block_idx,
                        instr_idx,
                        *mask,
                        shape.mask_descriptor(),
                    );
                }
            }
            SsaOp::VectorExtract { dest, vector, lane } => {
                if let Some(shape) = self.var_vector_shape(*vector) {
                    if shape
                        .fixed_lane_count()
                        .is_some_and(|lane_count| *lane >= lane_count)
                    {
                        self.invalid_vector(block_idx, instr_idx, "extract lane is out of bounds");
                    }
                    if let Some(expected) = T::vector_descriptor_lane_type(shape) {
                        self.check_var_type(
                            block_idx,
                            instr_idx,
                            *dest,
                            &expected,
                            "extract destination type does not match lane type",
                        );
                    }
                }
            }
            SsaOp::VectorInsert {
                dest,
                vector,
                lane,
                value,
            } => {
                let shape = self.var_vector_shape(*vector);
                if let Some(shape) = shape {
                    if shape
                        .fixed_lane_count()
                        .is_some_and(|lane_count| *lane >= lane_count)
                    {
                        self.invalid_vector(block_idx, instr_idx, "insert lane is out of bounds");
                    }
                    if let Some(expected) = T::vector_descriptor_lane_type(shape) {
                        self.check_var_type(
                            block_idx,
                            instr_idx,
                            *value,
                            &expected,
                            "insert value type does not match lane type",
                        );
                    }
                }
                self.check_shape(block_idx, instr_idx, *dest, shape);
            }
            SsaOp::VectorSplat {
                dest,
                value,
                vector_type,
            } => {
                let shape = T::vector_descriptor(vector_type);
                if shape.is_none() && !T::is_unknown(vector_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "splat vector_type is not a supported vector",
                    );
                }
                if let Some(shape) = shape
                    && let Some(expected) = T::vector_descriptor_lane_type(shape)
                {
                    self.check_var_type(
                        block_idx,
                        instr_idx,
                        *value,
                        &expected,
                        "splat value type does not match lane type",
                    );
                }
                self.check_shape(block_idx, instr_idx, *dest, shape);
            }
            SsaOp::VectorShuffle {
                dest,
                left,
                right,
                selector: VectorShuffleSelector::Index(index),
            } => {
                // The inputs share a lane shape the result takes, and the
                // result has a lane per integer index lane.
                let left_shape = self.var_vector_shape(*left);
                let right_shape = right.and_then(|v| self.var_vector_shape(v));
                let index_shape = self.var_vector_shape(*index);
                let same_lanes = |a: VectorDescriptor, b: VectorDescriptor| {
                    a.lane_bits() == b.lane_bits() && a.lane_kind() == b.lane_kind()
                };
                if let (Some(left_shape), Some(right_shape)) = (left_shape, right_shape)
                    && !same_lanes(left_shape, right_shape)
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "shuffle inputs differ in lane shape",
                    );
                }
                if let Some(index_shape) = index_shape
                    && matches!(index_shape.lane_kind(), VectorLaneKind::Float(_))
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "shuffle index lanes are no integers",
                    );
                }
                if let Some(dest_shape) = self.var_vector_shape(*dest) {
                    if let Some(left_shape) = left_shape
                        && !same_lanes(dest_shape, left_shape)
                    {
                        self.invalid_vector(
                            block_idx,
                            instr_idx,
                            "shuffle destination lane shape differs from input",
                        );
                    }
                    if let Some(index_shape) = index_shape
                        && (dest_shape.fixed_lane_count() != index_shape.fixed_lane_count()
                            || dest_shape.min_lane_count() != index_shape.min_lane_count())
                    {
                        self.invalid_vector(
                            block_idx,
                            instr_idx,
                            "shuffle destination lane count differs from its index",
                        );
                    }
                }
            }
            SsaOp::VectorShuffle {
                dest,
                left,
                right,
                selector: VectorShuffleSelector::Mask(mask),
            } => {
                let left_shape = self.var_vector_shape(*left);
                let right_shape = right.and_then(|v| self.var_vector_shape(v));
                if let Some(left_shape) = left_shape {
                    if !mask.is_valid_for(
                        left_shape.min_lane_count(),
                        right_shape.map(VectorDescriptor::min_lane_count),
                    ) {
                        self.invalid_vector(
                            block_idx,
                            instr_idx,
                            "shuffle mask selects an invalid lane",
                        );
                    }
                    if let Some(dest_shape) = self.var_vector_shape(*dest) {
                        if dest_shape
                            .fixed_lane_count()
                            .is_some_and(|lane_count| lane_count != mask.lanes().len() as u32)
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "shuffle destination lane count does not match mask",
                            );
                        }
                        if dest_shape.lane_bits() != left_shape.lane_bits()
                            || dest_shape.lane_kind() != left_shape.lane_kind()
                        {
                            self.invalid_vector(
                                block_idx,
                                instr_idx,
                                "shuffle destination lane shape differs from input",
                            );
                        }
                    }
                }
            }
            SsaOp::VectorCast {
                dest,
                value,
                target_type,
                from,
                to,
                saturate,
                ..
            } => {
                // Saturation clamps to an integer range.
                let integer = |kind: &VectorCastKind| {
                    matches!(kind, VectorCastKind::Signed | VectorCastKind::Unsigned)
                };
                if *saturate && !(integer(from) && integer(to)) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "a saturating cast is not between integers",
                    );
                }
                let target_shape = T::vector_descriptor(target_type);
                if target_shape.is_none() && !T::is_unknown(target_type) {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "cast target_type is not a supported vector",
                    );
                }
                // A cast converts each lane into a lane: a narrowing or widening
                // that also picks lanes (a half, the even ones) selects them
                // first, with a shuffle.
                let source_lanes = self
                    .var_vector_shape(*value)
                    .and_then(VectorDescriptor::fixed_lane_count);
                let target_lanes = target_shape.and_then(VectorDescriptor::fixed_lane_count);
                if let (Some(source), Some(target)) = (source_lanes, target_lanes)
                    && source != target
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "cast source and target lane counts differ",
                    );
                }
                self.check_shape(block_idx, instr_idx, *dest, target_shape);
            }
            SsaOp::VectorReinterpret {
                dest,
                value,
                target_type,
            } => {
                let source_shape = self.var_vector_shape(*value);
                let target_shape = T::vector_descriptor(target_type);
                if let (Some(source), Some(target)) = (source_shape, target_shape)
                    && source.total_bits().is_some()
                    && target.total_bits().is_some()
                    && source.total_bits() != target.total_bits()
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "reinterpret source and target widths differ",
                    );
                }
                self.check_shape(block_idx, instr_idx, *dest, target_shape);
            }
            // A grouped reduction's result is a vector of wider lanes.
            SsaOp::VectorReduce {
                dest,
                value,
                initial,
                groups: None,
                ..
            } => {
                if let Some(shape) = self.var_vector_shape(*value)
                    && let Some(expected) = T::vector_descriptor_lane_type(shape)
                {
                    self.check_var_type(
                        block_idx,
                        instr_idx,
                        *dest,
                        &expected,
                        "reduction destination type does not match lane type",
                    );
                    if let Some(initial) = initial {
                        self.check_var_type(
                            block_idx,
                            instr_idx,
                            *initial,
                            &expected,
                            "reduction initial value type does not match lane type",
                        );
                    }
                }
            }
            SsaOp::VectorBitmask { dest, value, .. } => {
                if let Some(ty) = self.var_type(*dest)
                    && !T::is_unknown(ty)
                    && !T::is_integer(ty)
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "bitmask destination must be an integer scalar",
                    );
                }
                if self.var_vector_shape(*value).is_none()
                    && self.var_mask_shape(*value).is_none()
                    && let Some(ty) = self.var_type(*value)
                    && !T::is_unknown(ty)
                {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "bitmask source must be a vector or mask",
                    );
                }
            }
            _ => {}
        }
    }

    /// Checks that a known controlled-vector value has an addressable bit representation.
    fn check_vector_bit_storage(&mut self, block: usize, instr_idx: usize, value: SsaVarId) {
        if let Some(ty) = self.var_type(value)
            && !T::is_unknown(ty)
            && !T::is_integer(ty)
            && !T::is_floating(ty)
            && !T::is_pointer(ty)
            && T::vector_descriptor(ty).is_none()
            && T::vector_mask_descriptor(ty).is_none()
        {
            self.invalid_vector(
                block,
                instr_idx,
                "controlled vector data has no numeric, address or predicate bit representation",
            );
        }
    }

    /// Returns the vector descriptor for `var` when its registered type is a known vector.
    fn var_vector_shape(&self, var: SsaVarId) -> Option<VectorDescriptor> {
        self.var_type(var).and_then(T::vector_descriptor)
    }

    /// Returns the mask descriptor for `var` when its registered type is a known mask.
    fn var_mask_shape(&self, var: SsaVarId) -> Option<VectorMaskDescriptor> {
        let ty = self.var_type(var)?;
        T::vector_mask_descriptor(ty).or_else(|| {
            (1..=u32::BITS).find_map(|lanes| {
                let shape = VectorMaskShape::new(lanes, 1)?;
                (T::vector_mask_type(shape).as_ref() == Some(ty))
                    .then_some(VectorMaskDescriptor::Fixed(shape))
            })
        })
    }

    /// Records a vector verifier error.
    pub fn invalid_vector(&mut self, block: usize, instr_idx: usize, reason: &str) {
        self.found.push(
            Check::VectorOperations,
            VerifierError::InvalidVectorOperation {
                block,
                instr_idx,
                reason: reason.to_owned(),
            },
        );
    }

    /// Checks that a variable has the expected vector shape when known.
    fn check_shape(
        &mut self,
        block: usize,
        instr_idx: usize,
        var: SsaVarId,
        expected: Option<VectorDescriptor>,
    ) {
        if let (Some(actual), Some(expected)) = (self.var_vector_shape(var), expected)
            && actual != expected
        {
            self.invalid_vector(
                block,
                instr_idx,
                "variable vector shape does not match expected shape",
            );
        }
    }

    /// Checks that a mask destination has a compatible shape when known.
    fn check_mask_dest_shape(
        &mut self,
        block: usize,
        instr_idx: usize,
        dest: SsaVarId,
        expected: VectorMaskDescriptor,
    ) {
        // A `None` expected mask type means the target does not model a
        // distinct mask type for this lane count - the mask lives in a plain
        // vector register (e.g. an AVX sign-bit mask), so there is no mask type
        // to compare against. Skip, mirroring `check_shape`'s Some/Some
        // guard; rejecting here would false-positive every masked op on such a
        // target.
        let Some(expected_ty) = T::vector_mask_descriptor_type(expected) else {
            return;
        };
        if let Some(ty) = self.var_type(dest) {
            if T::is_unknown(ty) {
                return;
            }
            if &expected_ty != ty {
                self.invalid_vector(
                    block,
                    instr_idx,
                    "compare destination is not a compatible vector mask",
                );
            }
        }
    }

    /// Checks that a mask source has a compatible shape when known.
    fn check_mask_source_shape(
        &mut self,
        block: usize,
        instr_idx: usize,
        mask: SsaVarId,
        expected: VectorMaskDescriptor,
    ) {
        // A `None` expected mask type means the target does not model a
        // distinct mask type for this lane count - the mask lives in a plain
        // vector register (e.g. an AVX2 `vmaskmov`/gather sign-bit mask), so
        // there is no mask type to compare against. Skip, mirroring
        // `check_source_shape`'s Some/Some guard; rejecting here would
        // false-positive every masked op whose mask register carries a
        // concrete vector type on such a target.
        let Some(expected_ty) = T::vector_mask_descriptor_type(expected) else {
            return;
        };
        if let Some(ty) = self.var_type(mask) {
            if T::is_unknown(ty) {
                return;
            }
            if &expected_ty != ty {
                self.invalid_vector(
                    block,
                    instr_idx,
                    "mask operand is not compatible with vector lane count",
                );
            }
        }
    }
}
