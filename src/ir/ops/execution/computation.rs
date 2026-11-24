//! A lane-wise computation: its kernel, lane control and how it rounds and
//! raises ([`Computation`]).

use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    ir::ops::{
        BcdAdjustKind, CarrylessPart, MemoryTagKind, SsaEffectKind, SsaEffects, VectorBinaryKind,
        VectorCastKind, VectorElement, VectorElementKind, VectorFaultMode, VectorReduceKind,
        VectorUnaryKind,
        execution::{
            BitfieldPosition, VectorFixedKind, VectorInactivePolicy, VectorLaneAction,
            VectorPredicateLayout, VectorRounding, VectorSlideSource, VectorStatus, kernel::Kernel,
        },
    },
    target::VectorTailPolicy,
};

/// The active interval and lane policies a computation runs its kernel under
/// for the elements of a vector (RISC-V's vector extension).
///
/// Prestart elements always retain their old value. When start is at or beyond
/// length, no vector result element changes, including agnostic tails. Inactive
/// elements never execute the kernel or produce exceptions. Structured kernels
/// such as reductions and slides additionally apply their own output-index rules.
/// Scalar mask summaries write their result even when the active length is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LaneControl {
    /// Whether mask tails may also contain a hypothetical computed result.
    ///
    /// Consumes an unsigned element limit after predicate and rounding controls.
    /// Tail indices below that limit may hold the old bit, one, or the result
    /// computed as if the active length equalled the limit. Masked-off elements
    /// permit only the old bit or one. Tail computation never produces exceptions.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "flag_absent"))]
    pub mask_tail: bool,
    /// The policy for elements at or beyond the active length.
    pub tail: VectorTailPolicy,
    /// Presence and treatment of a governing execution predicate.
    pub inactive: Option<VectorInactivePolicy>,
    /// Whether the operation takes no passthrough inputs.
    ///
    /// Every element the operation does not execute - a prestart, tail or
    /// inactive element, or every element of an execution whose restart is at
    /// or beyond its length - then holds an unspecified value rather than the
    /// old one. That is exact only where no reader observes such an element,
    /// which is where [`crate::passes::passthrough`] states it.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "flag_absent"))]
    pub no_passthrough: bool,
}

/// The complete static semantics of a computation: its kernel, the element
/// it computes on, its rounding and status, and, for one that runs over the
/// lanes of a vector, the lane control.
///
/// The runtime interface is outputs-first, then inputs. Under lane control
/// the inputs start with the restart index and active length, followed by
/// the governing predicate when `inactive` is present; then, with or without
/// it, a rounding mode when an environment is selected. An optional mask-tail
/// computation limit follows the rounding input. Passthrough data results
/// follow, unless `no_passthrough` drops them, then the kernel's data inputs.
/// Status and first-fault results follow data outputs and have no passthrough
/// input. A computation without lane control reads the rounding mode, where
/// it rounds by the environment, then its data inputs.
///
/// Every known runtime data value has integer, floating-point, pointer, vector
/// or predicate storage. Opaque managed objects require an explicit extraction
/// of their bit representation before participating in controlled execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct Computation {
    /// What is computed.
    pub kernel: Kernel,
    /// The data element class and width used by the kernel.
    pub element: VectorElement,
    /// Placement of governing and result predicate bits.
    pub predicate: VectorPredicateLayout,
    /// Static rounding choice or explicit rounding environment.
    pub rounding: VectorRounding,
    /// Additional status computed only from executed elements.
    pub status: VectorStatus,
    /// Replaces computed floating-point NaNs with the format's positive quiet NaN.
    ///
    /// Applies only to executed arithmetic results. Preserved elements and
    /// inactive or tail choices retain their bit patterns, including all ones.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "flag_absent"))]
    pub canonical_nan: bool,
    /// The active interval and lane policies, for a computation over the
    /// lanes of a vector; `None` for one of a whole value.
    pub lanes: Option<LaneControl>,
}

#[cfg(feature = "serde")]
pub fn flag_absent(value: &bool) -> bool {
    !*value
}

/// `compute.<name>`, the parameters the kernel carries beside its name, the
/// element it computes on (`.f32`, `.e8u`) where stated, its rounding and
/// status where it states them, and the lane control of a computation over
/// lanes.
impl fmt::Display for Computation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compute.{}", self.kernel.name())?;
        if self.element.bits != 0 {
            let class = if self.element.kind.is_float() {
                "f"
            } else {
                "e"
            };
            write!(
                f,
                ".{class}{}{}{}",
                self.element.bits,
                self.element.signedness_suffix(),
                self.element.rounding_suffix()
            )?;
        }
        match self.kernel {
            Kernel::TernaryLogic { table } => write!(f, " table={table:#04x}")?,
            Kernel::MultiSad { offsets } => write!(f, " offsets={offsets:#04x}")?,
            Kernel::DoubleBlockSad { selector } => write!(f, " selector={selector:#04x}")?,
            Kernel::StringCompare { control, .. } | Kernel::FloatControlled { control, .. } => {
                write!(f, " control={control:#04x}")?;
            }
            Kernel::CounterPredicate { part } => write!(f, " part={part}")?,
            Kernel::PredicateCounter { element_bits } => write!(f, " element={element_bits}")?,
            Kernel::Bitfield {
                position: BitfieldPosition::Immediate { index, length },
                ..
            } => write!(f, " index={index} length={length}")?,
            Kernel::DecimalAdjust {
                kind: BcdAdjustKind::AsciiMulAdjust | BcdAdjustKind::AsciiDivAdjust,
                base,
            } => write!(f, " base={base}")?,
            kernel if self.lanes_only() || self.lanes.is_some() => write!(f, " {kernel:?}")?,
            _ => {}
        }
        if !matches!(self.rounding, VectorRounding::None) {
            write!(f, " rounding={:?}", self.rounding)?;
        }
        if !matches!(self.status, VectorStatus::None) {
            write!(f, " status={:?}", self.status)?;
        }
        if self.canonical_nan {
            write!(f, " canonical_nan")?;
        }
        if let Some(lanes) = self.lanes {
            write!(f, " predicate={:?} {lanes:?}", self.predicate)?;
        }
        Ok(())
    }
}

impl Computation {
    /// A computation of `kernel` over a whole value of `element`, rounding as
    /// `rounding` states and reporting no status: no lane control, packed
    /// predicate bits, NaNs as computed.
    #[must_use]
    pub const fn whole(kernel: Kernel, element: VectorElement, rounding: VectorRounding) -> Self {
        Self {
            kernel,
            element,
            predicate: VectorPredicateLayout::Packed,
            rounding,
            status: VectorStatus::None,
            canonical_nan: false,
            lanes: None,
        }
    }

    /// The tail policy of a computation over lanes; `Undisturbed` for one of
    /// a whole value, which has no tail.
    #[must_use]
    pub const fn tail(self) -> VectorTailPolicy {
        match self.lanes {
            Some(lanes) => lanes.tail,
            None => VectorTailPolicy::Undisturbed,
        }
    }

    /// The inactive-element policy of a computation over lanes, `None` for
    /// one without a governing predicate or of a whole value.
    #[must_use]
    pub const fn inactive(self) -> Option<VectorInactivePolicy> {
        match self.lanes {
            Some(lanes) => lanes.inactive,
            None => None,
        }
    }

    /// Whether mask tails may hold a computed result
    /// ([`LaneControl::mask_tail`]).
    #[must_use]
    pub const fn mask_tail(self) -> bool {
        matches!(self.lanes, Some(lanes) if lanes.mask_tail)
    }

    /// Returns whether the selected lane action canonicalizes an arithmetic NaN.
    ///
    /// An empty reduction selection copies a seed rather than computing an
    /// arithmetic result. That copy preserves the seed even when its output
    /// lane action is `Execute`.
    #[must_use]
    pub const fn canonicalizes_nan(self, action: VectorLaneAction) -> bool {
        self.canonical_nan && matches!(action, VectorLaneAction::Execute)
    }

    /// Returns the action at an output index, preserving upward-slide holes.
    ///
    /// `auxiliary` is the full unsigned offset of an upward slide without scalar
    /// fill, or the selected source count for compression. Other kernels ignore it.
    #[must_use]
    pub const fn lane_action(
        self,
        index: u64,
        start: u64,
        length: u64,
        predicate: bool,
        tail_limit: u64,
        auxiliary: u64,
    ) -> VectorLaneAction {
        if matches!(self.kernel, Kernel::MaskSummary { .. }) {
            return if start == 0 && index == 0 {
                VectorLaneAction::Execute
            } else {
                VectorLaneAction::Preserve
            };
        }
        if start >= length || index < start {
            return VectorLaneAction::Preserve;
        }
        if matches!(
            self.kernel,
            Kernel::Reduce { .. } | Kernel::ReduceWide { .. }
        ) {
            return if index == 0 {
                VectorLaneAction::Execute
            } else {
                match self.tail() {
                    VectorTailPolicy::Undisturbed => VectorLaneAction::Preserve,
                    VectorTailPolicy::Agnostic => VectorLaneAction::Agnostic,
                }
            };
        }
        if matches!(
            self.kernel,
            Kernel::Slide {
                up: true,
                scalar_fill: false
            }
        ) && index < length
            && index < auxiliary
        {
            return VectorLaneAction::Preserve;
        }
        if index >= length
            || matches!(self.kernel, Kernel::InsertScalar) && index != 0
            || matches!(self.kernel, Kernel::Compress) && index >= auxiliary
        {
            if self.mask_tail() && index < tail_limit && (self.inactive().is_none() || predicate) {
                return VectorLaneAction::MaskTail;
            }
            return match self.tail() {
                VectorTailPolicy::Undisturbed => VectorLaneAction::Preserve,
                VectorTailPolicy::Agnostic => VectorLaneAction::Agnostic,
            };
        }
        if !predicate {
            return match self.inactive() {
                None => VectorLaneAction::Execute,
                Some(VectorInactivePolicy::Undisturbed) => VectorLaneAction::Preserve,
                Some(VectorInactivePolicy::Zero) => VectorLaneAction::Zero,
                Some(VectorInactivePolicy::Agnostic) => VectorLaneAction::Agnostic,
            };
        }
        VectorLaneAction::Execute
    }

    /// Returns the data source for a body element of a slide kernel.
    ///
    /// Source capacity is independent of active length. Large offsets do not
    /// wrap. Returns `None` for other kernels or result elements outside the
    /// active interval; restart and predicate policies are applied separately.
    #[must_use]
    pub const fn slide_source(
        self,
        index: u64,
        length: u64,
        capacity: u64,
        offset: u64,
    ) -> Option<VectorSlideSource> {
        let Kernel::Slide { up, scalar_fill } = self.kernel else {
            return None;
        };
        if index >= length {
            return None;
        }
        if scalar_fill
            && (up && index == 0
                || !up && matches!(index.checked_add(1), Some(last) if last == length))
        {
            return Some(VectorSlideSource::Scalar);
        }
        let amount = if scalar_fill { 1 } else { offset };
        let source = if up {
            match index.checked_sub(amount) {
                Some(source) => source,
                None => return Some(VectorSlideSource::Preserve),
            }
        } else {
            match index.checked_add(amount) {
                Some(source) => source,
                None => return Some(VectorSlideSource::Zero),
            }
        };
        Some(if source < capacity {
            VectorSlideSource::Element(source)
        } else {
            VectorSlideSource::Zero
        })
    }

    /// Returns the predicate bit governing an element, refusing invalid spacing or overflow.
    #[must_use]
    pub const fn predicate_bit(self, index: u64) -> Option<u64> {
        let bits = if matches!(self.kernel, Kernel::ReduceWide { .. }) {
            self.element.bits / 2
        } else {
            self.element.bits
        };
        self.predicate.bit(index, bits)
    }

    /// Returns the number of vector or scalar data results.
    #[must_use]
    pub const fn data_outputs(self) -> usize {
        match self.kernel {
            Kernel::Memory {
                load: true, fields, ..
            } => fields as usize,
            Kernel::Memory { load: false, .. } => 0,
            Kernel::Intersect => 2,
            Kernel::Crypto(kind) if self.lanes.is_none() => kind.data_outputs(),
            _ => 1,
        }
    }

    /// Returns the number of passthrough inputs: one per data result under
    /// lane control, or none where the operation states
    /// [`LaneControl::no_passthrough`] or computes a whole value.
    #[must_use]
    pub const fn passthrough_count(self) -> usize {
        match self.lanes {
            Some(lanes) if !lanes.no_passthrough => self.data_outputs(),
            _ => 0,
        }
    }

    /// Returns the number of results including exception and first-fault
    /// status, and the flags a computation writes: a string compare's, a
    /// decimal adjust's, a random draw's success, a crypto step's, and the
    /// generator state a random tag advances.
    #[must_use]
    pub const fn output_count(self) -> usize {
        let fault = matches!(
            self.kernel,
            Kernel::Memory {
                fault: VectorFaultMode::FaultOnlyFirst
                    | VectorFaultMode::FirstFault
                    | VectorFaultMode::NoFault,
                ..
            }
        );
        let flags = match self.kernel {
            // A random draw states whether it succeeded in the flags.
            Kernel::StringCompare { .. } | Kernel::DecimalAdjust { .. } | Kernel::Random { .. } => {
                1
            }
            Kernel::Crypto(kind) if self.lanes.is_none() => kind.writes_flags() as usize,
            // A random tag advances the generator whose state it draws from.
            Kernel::MemoryTag(MemoryTagKind::Random) => 1,
            _ => 0,
        };
        self.data_outputs()
            .saturating_add(fault as usize)
            .saturating_add(flags)
            .saturating_add((!matches!(self.status, VectorStatus::None)) as usize)
    }

    /// Returns the position among the inputs of the rounding mode a
    /// computation that rounds by the environment reads
    /// ([`VectorRounding::FloatingEnvironment`]): after the restart index and
    /// the active length under lane control, and after the governing
    /// predicate where inactive lanes are governed. `None` for a computation
    /// that reads no mode.
    #[must_use]
    pub const fn environment_input(self) -> Option<usize> {
        if !matches!(self.rounding, VectorRounding::FloatingEnvironment) {
            return None;
        }
        let lanes = if self.lanes.is_some() { 2usize } else { 0 };
        Some(lanes.saturating_add(self.inactive().is_some() as usize))
    }

    /// Returns the input prefix occupied by execution controls and the
    /// optional mask-tail limit: under lane control the restart index, the
    /// active length and the governing predicate, and with or without it the
    /// rounding mode of a computation that rounds by the environment.
    #[must_use]
    pub const fn control_input_count(self) -> usize {
        let environment = matches!(
            self.rounding,
            VectorRounding::FloatingEnvironment | VectorRounding::FixedEnvironment
        );
        let lanes = if self.lanes.is_some() { 2usize } else { 0 };
        lanes
            .saturating_add(self.inactive().is_some() as usize)
            .saturating_add(environment as usize)
            .saturating_add(self.mask_tail() as usize)
    }

    /// Returns the complete runtime input count, including control and
    /// passthrough.
    #[must_use]
    pub const fn input_count(self) -> usize {
        let data = match self.kernel {
            Kernel::Crypto(kind) => match (self.lanes, kind.group_input_count()) {
                (Some(_), Some(count)) => count,
                (Some(_), None) => 0,
                (None, _) => kind.whole_input_count(),
            },
            Kernel::TernaryLogic { .. } => 3,
            Kernel::MultiSad { .. }
            | Kernel::DoubleBlockSad { .. }
            | Kernel::ShuffleBits
            | Kernel::Intersect => 2,
            Kernel::HorizontalMinPos | Kernel::CounterPredicate { .. } => 1,
            Kernel::PredicateCounter { .. } => 3,
            Kernel::StringCompare {
                explicit_length, ..
            } => {
                if explicit_length {
                    4
                } else {
                    2
                }
            }
            Kernel::BitDeposit
            | Kernel::BitExtract
            | Kernel::BitGroup
            | Kernel::Checksum(_)
            | Kernel::VectorLengthChoice
            | Kernel::VectorFloatSupported
            | Kernel::VectorCryptoSupported => 2,
            Kernel::VectorShapeSupported => 3,
            Kernel::ScalarCipher(kind) => kind.input_count(),
            Kernel::FloatHelper(helper) => helper.input_count(),
            Kernel::FloatControlled { kind, .. } => kind.input_count(self.element.scalar),
            // The accumulator, then the carry and adjust flags an add or a
            // subtract adjust reads.
            Kernel::DecimalAdjust { kind, .. } => {
                1usize.saturating_add(kind.reads_flags() as usize)
            }
            // A random draw reads nothing a value states.
            Kernel::Random { .. } => 0,
            Kernel::PointerAuth(kind) => kind.input_count(),
            Kernel::FloatToIntModular => 1,
            // The address, the exclusion mask (`irg`'s register, zero where it
            // names none) or the address and tag offsets (`addg`), and the
            // exclusion register GCR_EL1; a random tag reads the generator's
            // state too, RGSR_EL1.
            Kernel::MemoryTag(MemoryTagKind::Random) => 4,
            Kernel::MemoryTag(MemoryTagKind::Add) => 4,
            // The data, after the destination for an insert, then the
            // control of a register form.
            Kernel::Bitfield { insert, position } => 1usize
                .saturating_add(insert as usize)
                .saturating_add(matches!(position, BitfieldPosition::Register) as usize),
            Kernel::Move
            | Kernel::InsertScalar
            | Kernel::Unary(_)
            | Kernel::Cast { .. }
            | Kernel::FloatClassify
            | Kernel::FloatToInteger { .. }
            | Kernel::Estimate7 { .. }
            | Kernel::Extend { .. }
            | Kernel::MaskPrefix(_)
            | Kernel::MaskIota
            | Kernel::MaskSummary { .. } => 1,
            Kernel::Binary(_)
            | Kernel::CarrylessMultiply { .. }
            | Kernel::MaskBinary(_)
            | Kernel::SignInject { .. }
            | Kernel::NarrowShift { .. }
            | Kernel::Fixed { .. }
            | Kernel::IntegerDivide { .. }
            | Kernel::ModuloShift { .. }
            | Kernel::MultiplyHigh { .. }
            | Kernel::Compare { .. }
            | Kernel::WidenBinary { .. }
            | Kernel::Reduce { .. }
            | Kernel::ReduceWide { .. }
            | Kernel::Compress
            | Kernel::Gather
            | Kernel::Slide { .. } => 2,
            Kernel::FusedMultiplyAdd { .. }
            | Kernel::IntegerMultiplyAdd { .. }
            | Kernel::WidenMultiplyAdd { .. }
            | Kernel::IntegerCarry { .. }
            | Kernel::Select => 3,
            Kernel::Index => 0,
            Kernel::Memory { load: true, .. } => 2,
            Kernel::Memory {
                load: false,
                fields,
                ..
            } => (fields as usize).saturating_add(2),
        };
        self.control_input_count()
            .saturating_add(self.passthrough_count())
            .saturating_add(data)
    }

    /// Whether the kernel runs only under lane control: the kernels the
    /// generic operations state for a whole value, which exist here for the
    /// lane-controlled form alone.
    #[must_use]
    const fn lanes_only(self) -> bool {
        matches!(
            self.kernel,
            Kernel::Move
                | Kernel::Unary(_)
                | Kernel::Binary(_)
                | Kernel::IntegerDivide { .. }
                | Kernel::ModuloShift { .. }
                | Kernel::MultiplyHigh { .. }
                | Kernel::Compare { .. }
                | Kernel::Cast { .. }
                | Kernel::Extend { .. }
                | Kernel::WidenBinary { .. }
                | Kernel::FusedMultiplyAdd { .. }
                | Kernel::IntegerMultiplyAdd { .. }
                | Kernel::Reduce { .. }
                | Kernel::Gather
                | Kernel::Slide { .. }
                | Kernel::Index
                | Kernel::Memory { .. }
                | Kernel::InsertScalar
                | Kernel::NarrowShift { .. }
                | Kernel::Fixed { .. }
                | Kernel::WidenMultiplyAdd { .. }
                | Kernel::MaskBinary(_)
                | Kernel::IntegerCarry { .. }
                | Kernel::Select
                | Kernel::MaskPrefix(_)
                | Kernel::MaskIota
                | Kernel::MaskSummary { .. }
                | Kernel::ReduceWide { .. }
                | Kernel::Compress
                | Kernel::SignInject { .. }
                | Kernel::FloatClassify
                | Kernel::FloatToInteger { .. }
                | Kernel::Estimate7 { .. }
        )
    }

    /// Whether the kernel computes only a whole value: the scalar and
    /// whole-register computations no lane-controlled form states.
    #[must_use]
    const fn whole_only(self) -> bool {
        !self.lanes_only()
            && !matches!(
                self.kernel,
                Kernel::CarrylessMultiply { .. } | Kernel::Crypto(_)
            )
    }

    /// Returns whether the static descriptor names a well-formed computation.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        if self.lanes_only() && self.lanes.is_none() || self.whole_only() && self.lanes.is_some() {
            return false;
        }
        // A kernel computing on an integer states it: of the width the
        // kernel fixes, or for a random draw of a register's width.
        let integer_bits = match self.kernel {
            Kernel::Random { .. } => matches!(self.element.bits, 16 | 32 | 64),
            kernel => match kernel.fixed_integer_bits() {
                Some(bits) => self.element.bits == bits,
                None => true,
            },
        };
        let states_integer = matches!(self.kernel, Kernel::Random { .. })
            || self.kernel.fixed_integer_bits().is_some();
        if !integer_bits
            || states_integer
                && (!matches!(self.element.kind, VectorElementKind::Integer)
                    || self.element.signed.is_some()
                    || !matches!(self.rounding, VectorRounding::None)
                    || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        // Rounding to odd is the direction of a narrowing between float
        // formats (Arm `fcvtxn`, RISC-V `vfncvt.rod.f.f.w`), which keeps the
        // sticky bit for a second rounding; no other kernel rounds by it.
        if matches!(self.rounding, VectorRounding::ToOdd)
            && !matches!(
                self.kernel,
                Kernel::Cast { from, to } if from.is_float() && to.is_float()
            )
        {
            return false;
        }
        let numeric_width = self.element.bits >= 8
            && self.element.bits <= 128
            && self.element.bits.is_power_of_two();
        if let Kernel::SignInject { negate, xor } = self.kernel
            && (!numeric_width
                || self.element.bits < 16
                || negate && xor
                || !matches!(
                    self.element.kind,
                    VectorElementKind::Float | VectorElementKind::BFloat16
                )
                || self.element.signed.is_some()
                || !matches!(self.rounding, VectorRounding::None)
                || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        if matches!(self.kernel, Kernel::FloatClassify)
            && (!numeric_width
                || self.element.bits < 16
                || !matches!(self.element.kind, VectorElementKind::Integer)
                || self.element.signed.is_some()
                || !matches!(self.rounding, VectorRounding::None)
                || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        if let Kernel::FloatToInteger { unsigned, .. } = self.kernel
            && (!numeric_width
                || !matches!(self.element.kind, VectorElementKind::Integer)
                || !matches!(
                    (self.element.signed, unsigned),
                    (Some(true), false) | (Some(false), true)
                )
                || !matches!(
                    self.rounding,
                    VectorRounding::FloatingEnvironment | VectorRounding::TowardZero
                )
                || !matches!(self.status, VectorStatus::FloatingExceptions))
        {
            return false;
        }
        if let Kernel::Estimate7 { square_root } = self.kernel
            && (!numeric_width
                || self.element.bits < 16
                || !matches!(
                    self.element.kind,
                    VectorElementKind::Float | VectorElementKind::BFloat16
                )
                || self.element.signed.is_some()
                || !matches!(self.status, VectorStatus::FloatingExceptions)
                || if square_root {
                    !matches!(self.rounding, VectorRounding::None)
                } else {
                    !matches!(
                        self.rounding,
                        VectorRounding::FloatingEnvironment | VectorRounding::TowardZero
                    )
                })
        {
            return false;
        }
        if self.canonical_nan
            && (!matches!(
                self.element.kind,
                VectorElementKind::Float | VectorElementKind::BFloat16
            ) || !matches!(
                self.kernel,
                Kernel::Binary(
                    VectorBinaryKind::Add
                        | VectorBinaryKind::Sub
                        | VectorBinaryKind::Mul
                        | VectorBinaryKind::Div
                        | VectorBinaryKind::Min
                        | VectorBinaryKind::Max
                        | VectorBinaryKind::MinNum
                        | VectorBinaryKind::MaxNum
                        | VectorBinaryKind::MinimumNumber
                        | VectorBinaryKind::MaximumNumber
                        | VectorBinaryKind::MinimumMagnitude
                        | VectorBinaryKind::MaximumMagnitude
                        | VectorBinaryKind::MinimumMagnitudeNumber
                        | VectorBinaryKind::MaximumMagnitudeNumber
                        | VectorBinaryKind::MinAbs
                        | VectorBinaryKind::MaxAbs
                ) | Kernel::Unary(VectorUnaryKind::Sqrt)
                    | Kernel::FusedMultiplyAdd { .. }
                    | Kernel::Estimate7 { .. }
                    | Kernel::Cast {
                        to: VectorCastKind::Float | VectorCastKind::BFloat16,
                        ..
                    }
                    | Kernel::Reduce { .. }
                    | Kernel::ReduceWide { .. }
            ))
        {
            return false;
        }
        if matches!(
            self.kernel,
            Kernel::Select | Kernel::Slide { .. } | Kernel::Gather | Kernel::Compress
        ) && (self.element.bits < 8
            || self.element.bits > 128
            || !self.element.bits.is_power_of_two()
            || self.element.signed.is_some()
            || !matches!(self.rounding, VectorRounding::None)
            || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        if matches!(
            self.kernel,
            Kernel::MaskPrefix(_) | Kernel::MaskSummary { .. } | Kernel::MaskIota
        ) && (!matches!(self.element.kind, VectorElementKind::Integer)
            || self.element.signed.is_some()
            || !matches!(self.rounding, VectorRounding::None)
            || !matches!(self.status, VectorStatus::None)
            || if matches!(self.kernel, Kernel::MaskIota) {
                self.element.bits < 8
                    || self.element.bits > 128
                    || !self.element.bits.is_power_of_two()
            } else {
                !matches!(self.predicate, VectorPredicateLayout::Packed)
                    && (self.element.bits == 0 || !self.element.bits.is_multiple_of(8))
                    || matches!(self.predicate, VectorPredicateLayout::Packed)
                        && self.element.bits != 1
            })
        {
            return false;
        }
        if matches!(self.kernel, Kernel::Compress) && self.inactive().is_some() {
            return false;
        }
        if let Kernel::ReduceWide { kind, .. } = self.kernel
            && (!matches!(
                self.element.kind,
                VectorElementKind::Integer | VectorElementKind::Float
            ) || self.element.bits < 16
                || self.element.bits > 128
                || !self.element.bits.is_power_of_two()
                || !matches!(kind, VectorReduceKind::Add)
                || matches!(self.element.kind, VectorElementKind::Float)
                    && !matches!(self.element.bits, 32 | 64 | 128)
                || matches!(self.element.kind, VectorElementKind::Integer)
                    && (self.element.signed.is_none()
                        || !matches!(self.rounding, VectorRounding::None)
                        || !matches!(self.status, VectorStatus::None)))
        {
            return false;
        }
        if matches!(self.kernel, Kernel::IntegerCarry { .. })
            && (!matches!(self.element.kind, VectorElementKind::Integer)
                || self.element.bits < 8
                || self.element.bits > 128
                || !self.element.bits.is_power_of_two()
                || self.element.signed.is_some()
                || !matches!(self.predicate, VectorPredicateLayout::Packed)
                || self.inactive().is_some()
                || !matches!(self.rounding, VectorRounding::None)
                || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        if matches!(self.kernel, Kernel::MaskBinary(_))
            && (!matches!(self.element.kind, VectorElementKind::Integer)
                || !matches!(
                    (self.predicate, self.element.bits),
                    (VectorPredicateLayout::Packed, 1) | (VectorPredicateLayout::ByteGranular, 8)
                )
                || self.element.signed.is_some()
                || !matches!(self.rounding, VectorRounding::None)
                || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        if self.mask_tail()
            && (!matches!(
                self.kernel,
                Kernel::Compare { .. }
                    | Kernel::MaskBinary(_)
                    | Kernel::MaskPrefix(_)
                    | Kernel::IntegerCarry {
                        mask_result: true,
                        ..
                    }
            ) || !matches!(self.tail(), VectorTailPolicy::Agnostic))
        {
            return false;
        }
        // Under lane control the element is a vector's, of a known width; a
        // whole value may be a scalar.
        if self.lanes.is_some() && (self.element.bits == 0 || self.element.scalar)
            || (matches!(self.predicate, VectorPredicateLayout::ByteGranular)
                && !self.element.bits.is_multiple_of(8))
        {
            return false;
        }
        if matches!(
            self.kernel,
            Kernel::InsertScalar
                | Kernel::Extend { .. }
                | Kernel::WidenBinary { .. }
                | Kernel::NarrowShift { .. }
                | Kernel::IntegerDivide { .. }
                | Kernel::IntegerMultiplyAdd { .. }
                | Kernel::WidenMultiplyAdd { .. }
                | Kernel::ModuloShift { .. }
                | Kernel::MultiplyHigh { .. }
                | Kernel::Unary(
                    VectorUnaryKind::TrailingZeros
                        | VectorUnaryKind::ByteSwap
                        | VectorUnaryKind::BitReverseIn8
                )
        ) && (!matches!(self.element.kind, VectorElementKind::Integer)
            || !matches!(self.rounding, VectorRounding::None)
            || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        // A group of words under lane control (RISC-V Zvk) states its whole
        // width; a whole register's step (`aesenc`) is of the register.
        if let Kernel::Crypto(kind) = self.kernel
            && self.lanes.is_some()
            && (!kind.accepts_group_bits(self.element.bits)
                || !matches!(self.element.kind, VectorElementKind::Integer)
                || self.element.signed.is_some()
                || !matches!(self.rounding, VectorRounding::None)
                || !matches!(self.status, VectorStatus::None)
                || self.mask_tail()
                || self.canonical_nan)
        {
            return false;
        }
        // Lane by lane the product keeps the element's width: its low or
        // high half.
        if let Kernel::CarrylessMultiply { part } = self.kernel
            && self.lanes.is_some()
            && matches!(part, CarrylessPart::Reversed | CarrylessPart::Full)
        {
            return false;
        }
        if matches!(self.kernel, Kernel::CarrylessMultiply { .. })
            && (!numeric_width
                || !matches!(self.element.kind, VectorElementKind::Integer)
                || self.element.bits < 8
                || self.element.signed.is_some()
                || !matches!(self.rounding, VectorRounding::None)
                || !matches!(self.status, VectorStatus::None))
        {
            return false;
        }
        if let Kernel::NarrowShift { unsigned } = self.kernel
            && (self.element.bits < 8
                || self.element.bits > 64
                || !self.element.bits.is_power_of_two()
                || matches!(self.element.signed, Some(signed) if signed == unsigned))
        {
            return false;
        }
        if let Kernel::WidenBinary {
            kind,
            wide_left,
            signed_left,
            signed_right,
        } = self.kernel
            && (self.element.bits < 16
                || self.element.bits > 128
                || !self.element.bits.is_power_of_two()
                || !matches!(
                    kind,
                    VectorBinaryKind::Add
                        | VectorBinaryKind::Sub
                        | VectorBinaryKind::Mul
                        | VectorBinaryKind::Shl
                )
                || matches!(kind, VectorBinaryKind::Shl)
                    && (wide_left || signed_left || signed_right)
                || wide_left && matches!(kind, VectorBinaryKind::Mul)
                || signed_left != signed_right
                    && (!signed_left || !matches!(kind, VectorBinaryKind::Mul))
                || matches!(self.element.signed, Some(sign) if sign != signed_left))
        {
            return false;
        }
        if matches!(
            self.kernel,
            Kernel::Binary(VectorBinaryKind::SatAdd | VectorBinaryKind::SatSub)
        ) && (!matches!(self.element.kind, VectorElementKind::Integer)
            || !matches!(self.rounding, VectorRounding::None)
            || matches!(self.status, VectorStatus::FloatingExceptions))
        {
            return false;
        }
        if let Kernel::WidenMultiplyAdd { signed_left, .. } = self.kernel
            && (self.element.bits < 16
                || self.element.bits > 128
                || !self.element.bits.is_power_of_two()
                || matches!(self.element.signed, Some(signed) if signed != signed_left))
        {
            return false;
        }
        if let Kernel::Fixed { kind } = self.kernel
            && (!matches!(self.element.kind, VectorElementKind::Integer)
                || self.element.bits < 8
                || self.element.bits > 64
                || !self.element.bits.is_power_of_two()
                || self.element.signed.is_none()
                || matches!(kind, VectorFixedKind::FractionalMultiply)
                    && !matches!(self.element.signed, Some(true))
                || !matches!(self.rounding, VectorRounding::FixedEnvironment)
                || !matches!(
                    (kind.saturates(), self.status),
                    (true, VectorStatus::Saturation) | (false, VectorStatus::None)
                ))
        {
            return false;
        }
        if matches!(self.kernel, Kernel::InsertScalar) && self.inactive().is_some() {
            return false;
        }
        if matches!(
            self.kernel,
            Kernel::Unary(VectorUnaryKind::ByteSwap | VectorUnaryKind::BitReverseIn8)
        ) && !self.element.bits.is_multiple_of(8)
        {
            return false;
        }
        if let Kernel::IntegerDivide { unsigned, .. } = self.kernel
            && !matches!(
                (self.element.signed, unsigned),
                (Some(true), false) | (Some(false), true)
            )
        {
            return false;
        }
        if let Kernel::MultiplyHigh {
            signed_left,
            signed_right,
        } = self.kernel
            && ((!signed_left && signed_right)
                || !matches!(
                    (self.element.signed, signed_left),
                    (Some(true), true) | (Some(false), false)
                ))
        {
            return false;
        }
        if let Kernel::ModuloShift { kind } = self.kernel
            && (!self.element.bits.is_power_of_two()
                || !matches!(
                    kind,
                    VectorBinaryKind::Shl
                        | VectorBinaryKind::Shr
                        | VectorBinaryKind::Rol
                        | VectorBinaryKind::Ror
                ))
        {
            return false;
        }
        if let Kernel::Memory {
            load,
            fields,
            fault,
            access_bits,
            sign_extend,
            byte_stream,
            indexed,
            coalescible,
            ..
        } = self.kernel
        {
            return fields != 0
                && access_bits != 0
                && access_bits.is_multiple_of(8)
                && access_bits as u32 <= self.element.bits
                && (!sign_extend || load)
                && (!byte_stream || !sign_extend && access_bits as u32 == self.element.bits)
                && (!coalescible || !indexed)
                && (load || matches!(fault, VectorFaultMode::Normal))
                && matches!(self.rounding, VectorRounding::None)
                && matches!(self.status, VectorStatus::None);
        }
        true
    }

    /// Returns effects, retaining unspecified policy choices as nondeterminism.
    #[must_use]
    pub fn effects(self) -> SsaEffects {
        match self.kernel {
            Kernel::Memory { load: true, .. } => SsaEffects::new(SsaEffectKind::Read, true),
            Kernel::Memory { load: false, .. } => SsaEffects::new(SsaEffectKind::Write, true),
            // A random source (`rdrand`, `rdseed`, `irg`) draws the value the
            // hardware supplies where control reaches it, reading no memory,
            // so a draw is never folded, moved or shared with another.
            Kernel::Random { .. } | Kernel::MemoryTag(MemoryTagKind::Random) => {
                SsaEffects::new(SsaEffectKind::RuntimeDefined, false)
            }
            _ if self.tail() == VectorTailPolicy::Agnostic
                || self.inactive() == Some(VectorInactivePolicy::Agnostic) =>
            {
                SsaEffects::new(SsaEffectKind::RuntimeDefined, false)
            }
            _ => SsaEffects::pure(),
        }
    }
}
