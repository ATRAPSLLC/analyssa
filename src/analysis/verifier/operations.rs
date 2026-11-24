//! The checks one operation must satisfy on its own ([`OperationScan`]).

use crate::{
    analysis::verifier::{Check, Findings, VerifierError},
    ir::{
        function::SsaFunction,
        ops::{
            AtomicAccessWidth, AtomicOrdering, AtomicRmwOp, ControlEffect, MemoryAccessSemantics,
            MemoryEffectLocation, SsaEffectKind, SsaEffects, SsaOp,
        },
        variable::{SsaVarId, SsaVariable},
    },
    target::Target,
};

/// The checks one operation must satisfy on its own, whatever the rest of the
/// function holds: shapes, widths, orderings, conversions and effects.
pub struct OperationScan<'s, 'a, T: Target> {
    /// The function the operation belongs to, for the types of its operands.
    pub ssa: &'a SsaFunction<T>,
    /// The scan's findings, which these checks add to.
    pub found: &'s mut Findings,
}

impl<T: Target> OperationScan<'_, '_, T> {
    /// Hands one instruction to the checks on what a single operation must
    /// satisfy on its own.
    pub fn run(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        self.check_vector_operation(block_idx, instr_idx, op);
        self.check_stated_rounding(block_idx, instr_idx, op);
        self.check_atomic_operation(block_idx, instr_idx, op);
        self.check_wide_arithmetic(block_idx, instr_idx, op);
        self.check_operand_widths(block_idx, instr_idx, op);
        self.check_conversion(block_idx, instr_idx, op);
        self.check_native_effects(block_idx, instr_idx, op);
        self.check_flag_masks(block_idx, instr_idx, op);
    }

    /// Checks that a lane-wise operation states a rounding
    /// ([`crate::ir::ops::VectorElement::rounding`]) only on float lanes of an
    /// operation whose result depends on it, and reads the environment's
    /// rounding mode ([`SsaOp::environment`]) only where it rounds such lanes by
    /// it. A conversion reads it only where it states no rounding of its own.
    fn check_stated_rounding(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        let (element, rounds) = match op {
            SsaOp::VectorUnary { kind, element, .. }
            | SsaOp::VectorPredicatedUnary { kind, element, .. } => {
                (element, kind.reads_rounding())
            }
            SsaOp::VectorBinary { kind, element, .. }
            | SsaOp::VectorPredicatedBinary { kind, element, .. } => {
                (element, kind.reads_rounding())
            }
            SsaOp::VectorTernary { kind, element, .. }
            | SsaOp::VectorPredicatedTernary { kind, element, .. } => {
                (element, kind.reads_rounding())
            }
            SsaOp::VectorReduce { kind, element, .. } => (element, kind.reads_rounding()),
            SsaOp::VectorCast {
                rounding,
                environment: Some(_),
                format: None,
                ..
            }
            | SsaOp::FloatConv {
                rounding,
                environment: Some(_),
                ..
            }
            | SsaOp::IntToFloat {
                rounding,
                environment: Some(_),
                ..
            } => {
                if rounding.is_some() {
                    self.invalid_vector(
                        block_idx,
                        instr_idx,
                        "a conversion stating its rounding reads the environment's mode",
                    );
                }
                return;
            }
            _ => return,
        };
        if element.rounding.is_some() && !(rounds && element.kind.is_float()) {
            self.invalid_vector(
                block_idx,
                instr_idx,
                "a rounding stated on an operation that does not round",
            );
        }
        // The environment's mode is read by an operation that rounds float
        // lanes and states no rounding of its own.
        if op.environment().is_some()
            && !(rounds && element.kind.is_float() && element.rounding.is_none())
        {
            self.invalid_vector(
                block_idx,
                instr_idx,
                "the environment's rounding mode read by an operation that does not round by it",
            );
        }
    }

    /// Checks native atomic ordering, width, and output-shape constraints.
    ///
    /// For each atomic op the address must be pointer-like and every value must
    /// match the declared access width, both only when the type is known.
    /// A read-modify-write or exchange may take any ordering, a release one
    /// included (RISC-V `amoadd.rl`, AArch64 `ldaddl`, a C++ `fetch_add` with
    /// `memory_order_release`). A compare-exchange failure path stores
    /// nothing, so it cannot carry release semantics, and its failure ordering
    /// cannot be stronger than its success ordering (compared through
    /// `ordering_rank`).
    /// Violations are reported as [`VerifierError::InvalidAtomicOperation`].
    fn check_atomic_operation(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        match op {
            SsaOp::AtomicExchange {
                dest,
                addr,
                value,
                width,
                ..
            }
            | SsaOp::AtomicLockRmw {
                dest,
                addr,
                value,
                width,
                ..
            } => {
                self.check_atomic_address(block_idx, instr_idx, *addr);
                self.check_atomic_width(block_idx, instr_idx, *dest, *width);
                self.check_atomic_width(block_idx, instr_idx, *value, *width);
            }
            SsaOp::AtomicCondRmw {
                old,
                addr,
                compare,
                value,
                width,
                ..
            } => {
                self.check_atomic_address(block_idx, instr_idx, *addr);
                for operand in [*old, *compare, *value] {
                    self.check_atomic_width(block_idx, instr_idx, operand, *width);
                }
            }
            SsaOp::AtomicCmpXchg {
                old,
                success,
                addr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
                width,
                ..
            } => {
                self.check_atomic_address(block_idx, instr_idx, *addr);
                self.check_atomic_width(block_idx, instr_idx, *old, *width);
                self.check_atomic_width(block_idx, instr_idx, *expected, *width);
                self.check_atomic_width(block_idx, instr_idx, *desired, *width);
                if let Some(success) = success
                    && let Some(ty) = self.var_type(*success)
                    && !T::is_unknown(ty)
                    && !T::is_integer(ty)
                {
                    self.invalid_atomic(
                        block_idx,
                        instr_idx,
                        "compare-exchange success output must be an integer boolean",
                    );
                }
                self.check_compare_exchange_ordering(
                    block_idx,
                    instr_idx,
                    *success_ordering,
                    *failure_ordering,
                );
            }
            SsaOp::AtomicPairRmw { addr, .. } => {
                self.check_atomic_address(block_idx, instr_idx, *addr);
            }
            SsaOp::AtomicCheckedRmw {
                old,
                addr,
                compare,
                value,
                op,
                rcw_mask,
                rcws_mask,
                width,
                ..
            } => {
                self.check_atomic_address(block_idx, instr_idx, *addr);
                // A read-check-write accesses 64 bits, or 128 as a pair, and
                // clears, sets or replaces; only a replacement compares.
                if !matches!(
                    width,
                    AtomicAccessWidth::Bits64 | AtomicAccessWidth::Bits128
                ) {
                    self.invalid_atomic(
                        block_idx,
                        instr_idx,
                        "read-check-write accesses 64 or 128 bits",
                    );
                }
                if !matches!(
                    op,
                    AtomicRmwOp::AndNot | AtomicRmwOp::Or | AtomicRmwOp::Xchg
                ) || (compare.is_some() && *op != AtomicRmwOp::Xchg)
                {
                    self.invalid_atomic(
                        block_idx,
                        instr_idx,
                        "read-check-write clears, sets, swaps or compares and swaps",
                    );
                }
                for var in [
                    Some(*old),
                    Some(*value),
                    Some(*rcw_mask),
                    *compare,
                    *rcws_mask,
                ]
                .into_iter()
                .flatten()
                {
                    self.check_atomic_width(block_idx, instr_idx, var, *width);
                }
            }
            SsaOp::AtomicPairCmpXchg {
                addr,
                success_ordering,
                failure_ordering,
                ..
            } => {
                self.check_atomic_address(block_idx, instr_idx, *addr);
                self.check_compare_exchange_ordering(
                    block_idx,
                    instr_idx,
                    *success_ordering,
                    *failure_ordering,
                );
            }
            _ => {}
        }
    }

    /// Checks a compare-exchange's orderings, a pair one's included: its
    /// failure path stores nothing, so it cannot release, and it cannot be
    /// stronger than the success path.
    fn check_compare_exchange_ordering(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        success: AtomicOrdering,
        failure: AtomicOrdering,
    ) {
        if matches!(failure, AtomicOrdering::Release | AtomicOrdering::AcqRel) {
            self.invalid_atomic(
                block_idx,
                instr_idx,
                "compare-exchange failure ordering cannot release",
            );
        }
        if ordering_rank(failure) > ordering_rank(success) {
            self.invalid_atomic(
                block_idx,
                instr_idx,
                "compare-exchange failure ordering is stronger than success ordering",
            );
        }
    }

    /// Checks that a bitcast keeps the width it reads, and that a conversion
    /// states its sign once.
    ///
    /// A bitcast reads its operand's bits as another type of the same width;
    /// bits placed into or read out of a wider value are a reshape
    /// (`VectorShuffle`, `VectorExtract`, `IntConv`), which states where they
    /// go.
    ///
    /// A float-to-integer conversion's `unsigned` flag is the destination's
    /// interpretation, and an integer target states the same thing; the two
    /// must agree, or a reader of the type and a reader of the flag convert the
    /// same bits differently. An integer-to-float conversion is not checked:
    /// its flag reads the source, and a source type need not state a sign (a
    /// CIL stack integer does not).
    ///
    /// An integer conversion reads an integer and gives one: an operand or a
    /// target the target classifies as a float or a vector makes it a bit
    /// move or a numeric float conversion that its readers take for an
    /// integer extension or truncation. A type the target classifies as
    /// neither is not judged: a pointer is an integer to it, and a host's
    /// value type may be one whose representation is an integer (a CIL
    /// enumeration) without the target stating it.
    fn check_conversion(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        if let SsaOp::IntConv {
            operand, target, ..
        } = op
        {
            let foreign = |ty: &T::Type| T::is_floating(ty) || T::is_vector(ty);
            let reason = if foreign(target) {
                Some(format!("an integer conversion to {target:?}"))
            } else {
                self.var_type(*operand)
                    .filter(|source| foreign(source))
                    .map(|source| format!("an integer conversion of {source:?}"))
            };
            if let Some(reason) = reason {
                self.found.push(
                    Check::Conversions,
                    VerifierError::InvalidConversion {
                        block: block_idx,
                        instr_idx,
                        reason,
                    },
                );
            }
        }
        if let SsaOp::Bitcast {
            operand, target, ..
        } = op
            && let (Some(source), Some(bits)) = (self.ssa.bit_width(*operand), T::bit_width(target))
            && source != bits
        {
            self.found.push(
                Check::Conversions,
                VerifierError::InvalidConversion {
                    block: block_idx,
                    instr_idx,
                    reason: format!("a bitcast of {source} bits reads them as {bits}"),
                },
            );
        }
        if let SsaOp::FloatToInt {
            target, unsigned, ..
        } = op
            && T::is_integer(target)
            && T::is_signed(target) == *unsigned
        {
            self.found.push(
                Check::Conversions,
                VerifierError::InvalidConversion {
                    block: block_idx,
                    instr_idx,
                    reason: "a float-to-integer flag contradicts its target's sign".to_owned(),
                },
            );
        }
    }

    /// Checks native implicit-width arithmetic operand and result relationships.
    ///
    /// Wide multiply and wide divide split a double-width result across two
    /// half-width registers, so every operand and output that participates must
    /// share one integer width. The per-op arms gather the relevant variables
    /// and defer to `check_same_width`, which tolerates unknown types and
    /// accepts pointer-width scalars, reporting a mismatch as an
    /// [`VerifierError::InvalidWideArithmetic`].
    fn check_wide_arithmetic(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        match op {
            SsaOp::WideMul {
                low,
                high,
                left,
                right,
                ..
            } => {
                self.check_same_width(
                    block_idx,
                    instr_idx,
                    &[*low, *high, *left, *right],
                    "wide multiply operands and outputs must have matching half width",
                );
            }
            SsaOp::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                ..
            } => {
                self.check_same_width(
                    block_idx,
                    instr_idx,
                    &[*quotient, *remainder, *high, *low, *divisor],
                    "wide divide operands and outputs must have matching half width",
                );
            }
            _ => {}
        }
    }

    /// Checks effect-summary consistency and terminator agreement.
    ///
    /// An operation's effect summary is the only channel a pass consults, so
    /// the invariants it has to satisfy - a fence declaring fence semantics, a
    /// block-ending control effect belonging to a terminator and only to one -
    /// are checked against the operation itself rather than against a parallel
    /// restatement of them. An [`SsaOp::FpuControl`] is a state image moved
    /// through the address it names first.
    fn check_native_effects(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        let effects = op.effects();
        self.check_effect_summary(block_idx, instr_idx, effects);

        let is_terminator = op.is_terminator();
        let ends_block = matches!(
            effects.control,
            ControlEffect::Terminator | ControlEffect::Return | ControlEffect::Throw
        );
        if is_terminator && !ends_block {
            self.invalid_native(
                Check::NativeEffects,
                block_idx,
                instr_idx,
                "terminator op must declare a block-ending control effect",
            );
        }
        if ends_block && !is_terminator {
            self.invalid_native(
                Check::NativeEffects,
                block_idx,
                instr_idx,
                "op with a block-ending control effect must be a terminator",
            );
        }
        // An FPU control operation of the SSA moves a state image, at the
        // address it names first; the unit's other operations are the
        // registers they move.
        if let SsaOp::FpuControl(data) = op {
            if !data.kind.moves_image() {
                self.invalid_native(
                    Check::NativeEffects,
                    block_idx,
                    instr_idx,
                    "an FPU control operation moves a state image",
                );
            } else if data.inputs.is_empty() {
                self.invalid_native(
                    Check::NativeEffects,
                    block_idx,
                    instr_idx,
                    "a state image move names its address",
                );
            }
        }
    }

    /// Checks that flag-selecting operations name flags the IR defines.
    ///
    /// Two shapes, both of which make an operation claim something about flags
    /// that nothing else in the crate can interpret.
    ///
    /// A [`SsaOp::ReadFlags`] or [`SsaOp::MergeFlags`] mask with bits outside
    /// [`FlagsMask::NAMED`] selects a flag this crate has no name for, and an
    /// [`SsaOp::InsertFlag`] sets one boolean, so its mask names exactly one
    /// defined bit. [`FlagsMask::from_bits`] is
    /// unmasked on purpose - a persisted mask must round-trip so the bits can be
    /// *reported* rather than silently dropped - which makes reporting them
    /// this check's job. The empty mask stays legal: selecting no flags is a
    /// legal thing to ask for.
    ///
    /// [`FlagsMask::NAMED`]: crate::ir::ops::FlagsMask::NAMED
    /// [`FlagsMask::from_bits`]: crate::ir::ops::FlagsMask::from_bits
    fn check_flag_masks(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        match op {
            SsaOp::ReadFlags { mask, .. } | SsaOp::MergeFlags { mask, .. } if !mask.is_valid() => {
                self.invalid_native(
                    Check::FlagMasks,
                    block_idx,
                    instr_idx,
                    "flag mask selects bits that name no defined flag",
                );
            }
            SsaOp::InsertFlag { mask, .. } if !mask.is_valid() || !mask.is_single_bit() => {
                self.invalid_native(
                    Check::FlagMasks,
                    block_idx,
                    instr_idx,
                    "an inserted flag names exactly one defined bit",
                );
            }
            SsaOp::FloatRaise {
                dest,
                op,
                raises,
                layout,
                ..
            } => {
                let reason = if op.defs().any(|def| def != *dest) {
                    "a raise's operation defines a variable other than the raise"
                } else if matches!(**op, SsaOp::FloatRaise { .. })
                    || op.is_terminator()
                    || !op.effects().is_pure()
                {
                    "a raise is of a computation, which neither raises nor has an effect"
                } else if !layout.holds().contains(*raises) {
                    "a raise sets a flag its layout keeps at no bit"
                } else {
                    return;
                };
                self.invalid_native(Check::FlagMasks, block_idx, instr_idx, reason);
            }
            _ => {}
        }
    }

    /// Returns the registered type for `var`.
    pub fn var_type(&self, var: SsaVarId) -> Option<&T::Type> {
        self.ssa.variable(var).map(SsaVariable::var_type)
    }

    /// Records a native/effect verifier error under `check`.
    fn invalid_native(&mut self, check: Check, block: usize, instr_idx: usize, reason: &str) {
        self.found.push(
            check,
            VerifierError::InvalidNativeOperation {
                block,
                instr_idx,
                reason: reason.to_owned(),
            },
        );
    }

    /// Checks effect-summary invariants that passes rely on.
    ///
    /// The effect summary is the only channel a pass consults before reordering
    /// or eliminating an operation, so an internally inconsistent summary is a
    /// miscompilation waiting to happen. An effect that touches no memory - pure
    /// or runtime-defined - must reference no memory, carry no memory
    /// semantics, and declare no ordering; an atomic or fence
    /// effect must carry the matching memory semantics and an explicit ordering;
    /// and a volatile effect must actually touch memory. Each inconsistency is
    /// reported as an [`VerifierError::InvalidNativeOperation`].
    ///
    /// # Arguments
    ///
    /// * `block` - Index of the block containing the instruction, for error
    ///   reporting.
    /// * `instr_idx` - Index of the instruction within its block.
    /// * `effects` - The operation's effect summary to validate.
    pub fn check_effect_summary(&mut self, block: usize, instr_idx: usize, effects: SsaEffects) {
        if effects.removable_when_unused() {
            if effects.memory != MemoryEffectLocation::None {
                self.invalid_native(
                    Check::NativeEffects,
                    block,
                    instr_idx,
                    "memory-free effect cannot reference memory",
                );
            }
            if effects.memory_semantics != MemoryAccessSemantics::None {
                self.invalid_native(
                    Check::NativeEffects,
                    block,
                    instr_idx,
                    "memory-free effect cannot have memory semantics",
                );
            }
            if effects.ordering.is_some() {
                self.invalid_native(
                    Check::NativeEffects,
                    block,
                    instr_idx,
                    "memory-free effect cannot have ordering",
                );
            }
        }

        match effects.kind {
            SsaEffectKind::Atomic => {
                if effects.memory_semantics != MemoryAccessSemantics::Atomic {
                    self.invalid_native(
                        Check::NativeEffects,
                        block,
                        instr_idx,
                        "atomic effect must use atomic memory semantics",
                    );
                }
                if effects.ordering.is_none() {
                    self.invalid_native(
                        Check::NativeEffects,
                        block,
                        instr_idx,
                        "atomic effect must declare ordering",
                    );
                }
            }
            SsaEffectKind::Fence => {
                if effects.memory_semantics != MemoryAccessSemantics::Fence {
                    self.invalid_native(
                        Check::NativeEffects,
                        block,
                        instr_idx,
                        "fence effect must use fence memory semantics",
                    );
                }
                if effects.ordering.is_none() {
                    self.invalid_native(
                        Check::NativeEffects,
                        block,
                        instr_idx,
                        "fence effect must declare ordering",
                    );
                }
            }
            SsaEffectKind::Pure
            | SsaEffectKind::RuntimeDefined
            | SsaEffectKind::Read
            | SsaEffectKind::Write
            | SsaEffectKind::ReadWrite
            | SsaEffectKind::Call
            | SsaEffectKind::Opaque => {}
        }

        if effects.volatile && effects.memory_semantics == MemoryAccessSemantics::None {
            self.invalid_native(
                Check::NativeEffects,
                block,
                instr_idx,
                "volatile effect must access memory",
            );
        }
    }

    /// Records an atomic verifier error.
    fn invalid_atomic(&mut self, block: usize, instr_idx: usize, reason: &str) {
        self.found.push(
            Check::AtomicOperations,
            VerifierError::InvalidAtomicOperation {
                block,
                instr_idx,
                reason: reason.to_owned(),
            },
        );
    }

    /// Checks that an atomic address is pointer-like when the type is known.
    fn check_atomic_address(&mut self, block: usize, instr_idx: usize, addr: SsaVarId) {
        if let Some(ty) = self.var_type(addr)
            && !T::is_unknown(ty)
            && !T::is_pointer(ty)
        {
            self.invalid_atomic(block, instr_idx, "atomic address must be pointer-like");
        }
    }

    /// Checks that an atomic value matches the declared access width when known.
    fn check_atomic_width(
        &mut self,
        block: usize,
        instr_idx: usize,
        var: SsaVarId,
        width: AtomicAccessWidth,
    ) {
        if let Some(expected) = width.bits()
            && let Some(ty) = self.var_type(var)
            && !T::is_unknown(ty)
            && T::bit_width(ty) != Some(expected)
        {
            self.invalid_atomic(
                block,
                instr_idx,
                "atomic value type width does not match access width",
            );
        }
    }

    /// Checks that all known integer widths in `vars` match, each at the
    /// width it computes at ([`Target::operand_width`]).
    fn check_same_width(
        &mut self,
        block: usize,
        instr_idx: usize,
        vars: &[SsaVarId],
        reason: &str,
    ) {
        let mut expected = None;
        for var in vars {
            let Some(ty) = self.var_type(*var) else {
                continue;
            };
            if T::is_unknown(ty) {
                continue;
            }
            // Native wide multiply/divide operate on the machine word as a raw
            // integer bit-pattern. A pointer is an integer-width scalar and is a
            // faithful operand here (e.g. `div` of an address, pointer hashing):
            // accept integer *or* pointer types and let the width check below
            // enforce consistency. Rejecting pointers outright produced false
            // positives on lifted `div`/`mul` whose accumulator held a pointer.
            if !T::is_integer(ty) && !T::is_pointer(ty) {
                self.invalid_wide(block, instr_idx, "wide arithmetic values must be integers");
                continue;
            }
            let Some(width) = T::operand_width(ty) else {
                continue;
            };
            if let Some(expected) = expected {
                if width != expected {
                    self.invalid_wide(block, instr_idx, reason);
                    return;
                }
            } else {
                expected = Some(width);
            }
        }
    }

    /// Checks that a scalar operation computes at one width.
    ///
    /// An addition, subtraction, multiplication, division, remainder, logical
    /// operation, negation or complement reads its operands and writes its
    /// result at one width; a shift or rotate writes the width of the value it
    /// shifts, its amount any width; a comparison reads two values of one
    /// width, whether it yields a boolean, a branch or a float compare's
    /// flags; a select chooses between two values of its result's width; a
    /// float function (an x87 `fpatan`, `fprem` or `fscale`) reads and writes
    /// its floats at one width, beside the integers some of its results are. A
    /// mismatch is a lift that typed one operand wrongly (a 64-bit constant
    /// against a 32-bit register), which no fold can evaluate. Each operand
    /// counts at the width it computes at ([`Target::operand_width`]), so a
    /// target that widens what its operations read is held to its own rule.
    /// Unknown, vector and non-scalar types are skipped, as is an operand
    /// whose width is not fixed.
    fn check_operand_widths(&mut self, block_idx: usize, instr_idx: usize, op: &SsaOp<T>) {
        let functions;
        let mut floats_only = false;
        let vars: &[SsaVarId] = match op {
            SsaOp::Add {
                dest, left, right, ..
            }
            | SsaOp::AddOvf {
                dest, left, right, ..
            }
            | SsaOp::Sub {
                dest, left, right, ..
            }
            | SsaOp::SubOvf {
                dest, left, right, ..
            }
            | SsaOp::Mul {
                dest, left, right, ..
            }
            | SsaOp::MulOvf {
                dest, left, right, ..
            }
            | SsaOp::Div {
                dest, left, right, ..
            }
            | SsaOp::Rem {
                dest, left, right, ..
            }
            | SsaOp::And {
                dest, left, right, ..
            }
            | SsaOp::Or {
                dest, left, right, ..
            }
            | SsaOp::Xor {
                dest, left, right, ..
            } => &[*dest, *left, *right],
            SsaOp::Neg { dest, operand, .. } | SsaOp::Not { dest, operand, .. } => {
                &[*dest, *operand]
            }
            SsaOp::Shl { dest, value, .. }
            | SsaOp::Shr { dest, value, .. }
            | SsaOp::Rol { dest, value, .. }
            | SsaOp::Ror { dest, value, .. } => &[*dest, *value],
            SsaOp::Ceq { left, right, .. }
            | SsaOp::Clt { left, right, .. }
            | SsaOp::Cgt { left, right, .. }
            | SsaOp::BranchCmp { left, right, .. }
            | SsaOp::FloatCompareFlags { left, right, .. } => &[*left, *right],
            SsaOp::Select {
                dest,
                true_val,
                false_val,
                ..
            } => &[*dest, *true_val, *false_val],
            SsaOp::FloatFunction(data) => {
                floats_only = true;
                functions = data
                    .inputs
                    .iter()
                    .chain(&data.outputs)
                    .copied()
                    .collect::<Vec<_>>();
                &functions
            }
            _ => return,
        };
        let mut widths = vars.iter().filter_map(|var| {
            let ty = self.var_type(*var)?;
            (T::is_floating(ty) || !floats_only && (T::is_integer(ty) || T::is_pointer(ty)))
                .then(|| T::operand_width(ty))
                .flatten()
        });
        let Some(first) = widths.next() else {
            return;
        };
        if let Some(other) = widths.find(|width| *width != first) {
            self.found.push(
                Check::OperandWidths,
                VerifierError::InvalidOperandWidths {
                    block: block_idx,
                    instr_idx,
                    reason: format!(
                        "{} computes over {first} and {other} bits",
                        op.opcode_name()
                    ),
                },
            );
        }
    }

    /// Records a wide arithmetic verifier error.
    fn invalid_wide(&mut self, block: usize, instr_idx: usize, reason: &str) {
        self.found.push(
            Check::WideArithmetic,
            VerifierError::InvalidWideArithmetic {
                block,
                instr_idx,
                reason: reason.to_owned(),
            },
        );
    }

    /// Checks that a known bit offset or length is a scalar integer.
    pub fn check_region_integer(&mut self, block: usize, instr_idx: usize, offset: SsaVarId) {
        if let Some(ty) = self.var_type(offset)
            && !T::is_unknown(ty)
            && !T::is_integer(ty)
        {
            self.invalid_vector(
                block,
                instr_idx,
                "bit region position or length is not a scalar integer",
            );
        }
    }

    /// Checks a variable's registered type against an expected scalar type.
    pub fn check_var_type(
        &mut self,
        block: usize,
        instr_idx: usize,
        var: SsaVarId,
        expected: &T::Type,
        reason: &str,
    ) {
        if let Some(actual) = self.var_type(var)
            && !T::is_unknown(actual)
            && actual != expected
        {
            self.invalid_vector(block, instr_idx, reason);
        }
    }
}

/// Returns a conservative ordering strength rank for compare-exchange checks.
///
/// The compare-exchange check only needs to compare two orderings by strength,
/// not reconstruct the full memory-model lattice, so this projects the five
/// orderings onto a total `0..=3` scale. Acquire and Release share rank 1
/// deliberately: they are incomparable in the model, and collapsing them to
/// equal strength keeps the "failure ordering stronger than success" test from
/// firing on a pairing the model does not actually order.
///
/// # Arguments
///
/// * `ordering` - The atomic ordering to rank.
///
/// # Returns
///
/// A rank in `0..=3`, where a higher value means stronger ordering.
const fn ordering_rank(ordering: AtomicOrdering) -> u8 {
    match ordering {
        AtomicOrdering::Relaxed => 0,
        AtomicOrdering::Acquire => 1,
        AtomicOrdering::Release => 1,
        AtomicOrdering::AcqRel => 2,
        AtomicOrdering::SeqCst => 3,
    }
}
