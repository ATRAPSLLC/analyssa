//! Choosing the operation that models a conversion.
//!
//! A front end that knows only the endpoints of a coercion - their value
//! domains, or their types - asks here for the op that crosses them, so the
//! classification lives in one place: [`SsaOp::IntConv`] within the integers,
//! [`SsaOp::IntToPtr`] and [`SsaOp::PtrToInt`] across the pointer boundary,
//! [`SsaOp::IntToFloat`], [`SsaOp::FloatToInt`] and [`SsaOp::FloatConv`] across
//! and within the floating-point one.

use crate::{
    ir::{
        float::{FloatFormat, FloatRounding},
        ops::{VectorCastKind, VectorElement, def::SsaOp},
        variable::SsaVarId,
    },
    target::Target,
};

/// The value domain of a conversion endpoint: integer, pointer, floating
/// point, or vector. The domains a value crosses select which conversion op
/// models it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionDomain {
    /// An integer or a boolean - and anything that is none of the others (an aggregate).
    Int,
    /// A pointer or a managed reference.
    Ptr,
    /// A floating-point value.
    Float,
    /// A vector, fixed or scalable. Its bits are its lanes, so a conversion
    /// into or out of one restates them: [`SsaOp::VectorReinterpret`] between
    /// vectors and [`SsaOp::Bitcast`] across a scalar.
    Vector,
}

impl ConversionDomain {
    /// The domain of a value of type `ty`, as the target classifies it.
    #[must_use]
    pub fn of<T: Target>(ty: &T::Type) -> Self {
        if T::is_vector(ty) {
            Self::Vector
        } else if T::is_pointer(ty) || T::is_reference(ty) {
            Self::Ptr
        } else if T::is_floating(ty) {
            Self::Float
        } else {
            Self::Int
        }
    }
}

/// What a conversion a front end chose before its operand's type was known
/// does to that operand.
///
/// The op names a domain it reads, and with no type to read from, the front
/// end's choice says only what it wanted done. Two front ends want different
/// things: a machine-code lift coercing a register's contents into the type an
/// instruction reads moves bits, while a CIL `conv.i8` converts the value it is
/// handed - of a `float64`, it truncates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionIntent {
    /// The conversion moves its operand's bits. A conversion of bits
    /// ([`SsaOp::IntConv`] checking no overflow, [`SsaOp::PtrToInt`],
    /// [`SsaOp::IntToPtr`]) reads them by the op their holder takes, and a
    /// conversion of values ([`SsaOp::IntToFloat`], [`SsaOp::FloatToInt`],
    /// [`SsaOp::FloatConv`]) reads its own domain through them.
    Bits,
    /// The conversion converts its operand's value: it is the op
    /// [`SsaOp::conversion_between`] chooses for the operand's type and the
    /// target.
    Values,
}

/// A conversion restated for the type its operand turned out to have
/// ([`SsaOp::restated_for_operand`]).
#[derive(Debug, Clone)]
pub struct Restated<T: Target> {
    /// The type the operand's bits are first read as, by a [`SsaOp::Bitcast`]
    /// whose result [`Self::op`] then reads in the operand's place; `None`
    /// when it reads the operand itself.
    pub through: Option<T::Type>,
    /// The conversion, defining the same value and reading the operand.
    pub op: SsaOp<T>,
}

/// How a conversion's operand holds its bits, as far as choosing the op that
/// reads them goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Held {
    /// An integer or a boolean.
    Integer,
    /// A pointer or a managed reference.
    Address,
    /// A floating-point value.
    Float,
    /// A vector of lanes.
    Vector,
}

impl Held {
    /// How a value of type `ty` holds its bits; `None` for a type that says
    /// nothing about them, such as an unknown one.
    fn of<T: Target>(ty: &T::Type) -> Option<Self> {
        if T::is_pointer(ty) || T::is_reference(ty) {
            Some(Self::Address)
        } else if T::is_floating(ty) {
            Some(Self::Float)
        } else if T::is_vector(ty) {
            Some(Self::Vector)
        } else if T::is_integer(ty) {
            Some(Self::Integer)
        } else {
            None
        }
    }
}

impl<T: Target> SsaOp<T> {
    /// The rounding mode of the floating-point environment the operation
    /// rounds by, where it reads one: the `environment` operand of a lane
    /// operation, a reduction, a vector cast, a float conversion or an
    /// integer-to-float conversion. `None` for an operation that states its
    /// rounding, rounds nothing, or reads no mode.
    #[must_use]
    pub const fn environment(&self) -> Option<SsaVarId> {
        match self {
            Self::VectorUnary { environment, .. }
            | Self::VectorBinary { environment, .. }
            | Self::VectorTernary { environment, .. }
            | Self::VectorReduce { environment, .. }
            | Self::VectorCast { environment, .. }
            | Self::VectorPredicatedUnary { environment, .. }
            | Self::VectorPredicatedBinary { environment, .. }
            | Self::VectorPredicatedTernary { environment, .. }
            | Self::FloatConv { environment, .. }
            | Self::IntToFloat { environment, .. } => *environment,
            _ => None,
        }
    }

    /// The value that names a format this operation does not state
    /// (FP8's encoding and scale), where it has one.
    #[must_use]
    pub const fn format_source(&self) -> Option<SsaVarId> {
        match self {
            Self::VectorCast { format, .. } => *format,
            _ => None,
        }
    }

    /// Whether the operation rounds by the floating-point environment's mode
    /// where it states none: a lane operation whose result depends on the
    /// rounding (`reads_rounding`) on float lanes whose rounding the element
    /// leaves to the environment, a vector cast to or from floats that can be
    /// inexact, and a float conversion or an integer-to-float conversion that
    /// can be inexact. A conversion whose every value the target holds
    /// ([`FloatFormat::includes`], [`FloatFormat::includes_integer`]) rounds nothing.
    /// `type_of` gives an operand's type, where known; an operand of unknown
    /// type is taken to be one that can round.
    #[must_use]
    pub fn rounds_by_environment(&self, type_of: impl Fn(SsaVarId) -> Option<T::Type>) -> bool {
        let float = |element: &VectorElement| {
            element.kind.is_float() && !element.kind.opens_specials() && element.rounding.is_none()
        };
        let lanes = |ty: &T::Type| T::vector_shape(ty).map(|shape| shape.lane_bits);
        match self {
            Self::VectorUnary { kind, element, .. }
            | Self::VectorPredicatedUnary { kind, element, .. } => {
                float(element) && kind.reads_rounding()
            }
            Self::VectorBinary { kind, element, .. }
            | Self::VectorPredicatedBinary { kind, element, .. } => {
                float(element) && kind.reads_rounding()
            }
            Self::VectorTernary { kind, element, .. }
            | Self::VectorPredicatedTernary { kind, element, .. } => {
                float(element) && kind.reads_rounding()
            }
            Self::VectorReduce { kind, element, .. } => float(element) && kind.reads_rounding(),
            Self::VectorCast {
                value,
                target_type,
                from,
                to,
                rounding,
                saturate,
                ..
            } => {
                if rounding.is_some()
                    || *saturate
                    || from.opens_specials()
                    || to.opens_specials()
                    || !(from.is_float() || to.is_float())
                {
                    return false;
                }
                let source = type_of(*value).as_ref().and_then(lanes);
                let target = lanes(target_type);
                let format = |kind: VectorCastKind, bits: Option<u32>| match kind {
                    VectorCastKind::BFloat16 => Some(FloatFormat::BFloat16),
                    _ => FloatFormat::of_width(bits?),
                };
                match (from.is_float(), to.is_float()) {
                    (true, true) => !matches!(
                        (format(*from, source), format(*to, target)),
                        (Some(from), Some(to)) if to.includes(from)
                    ),
                    (false, true) => !matches!(
                        (source, format(*to, target)),
                        (Some(bits), Some(to))
                            if to.includes_integer(bits, *from == VectorCastKind::Signed)
                    ),
                    _ => true,
                }
            }
            Self::FloatConv {
                operand,
                target,
                rounding,
                source,
                ..
            } => {
                rounding.is_none()
                    && !matches!(
                        (
                            source.or_else(|| type_of(*operand).as_ref().and_then(T::float_format)),
                            T::float_format(target),
                        ),
                        (Some(from), Some(to)) if to.includes(from)
                    )
            }
            Self::IntToFloat {
                operand,
                target,
                unsigned,
                rounding,
                ..
            } => {
                rounding.is_none()
                    && !matches!(
                        (
                            type_of(*operand).as_ref().and_then(T::bit_width),
                            T::float_format(target),
                        ),
                        (Some(bits), Some(to)) if to.includes_integer(bits, !*unsigned)
                    )
            }
            _ => false,
        }
    }

    /// The `environment` operand of an operation that may round by the
    /// floating-point environment's mode ([`Self::environment`]), to restate
    /// it; `None` for any other operation.
    pub fn environment_mut(&mut self) -> Option<&mut Option<SsaVarId>> {
        match self {
            Self::VectorUnary { environment, .. }
            | Self::VectorBinary { environment, .. }
            | Self::VectorTernary { environment, .. }
            | Self::VectorReduce { environment, .. }
            | Self::VectorCast { environment, .. }
            | Self::VectorPredicatedUnary { environment, .. }
            | Self::VectorPredicatedBinary { environment, .. }
            | Self::VectorPredicatedTernary { environment, .. }
            | Self::FloatConv { environment, .. }
            | Self::IntToFloat { environment, .. } => Some(environment),
            _ => None,
        }
    }

    /// This operation rounding by `rounding` in place of the environment's
    /// mode it read ([`Self::environment`]): the rounding stated where it
    /// states one, its lanes' element's or the conversion's, and the mode
    /// operand gone. An operation reading no mode as it is.
    #[must_use]
    pub fn rounded_by(mut self, rounding: FloatRounding) -> Self {
        let stated = match &mut self {
            Self::VectorUnary { element, .. }
            | Self::VectorBinary { element, .. }
            | Self::VectorTernary { element, .. }
            | Self::VectorReduce { element, .. }
            | Self::VectorPredicatedUnary { element, .. }
            | Self::VectorPredicatedBinary { element, .. }
            | Self::VectorPredicatedTernary { element, .. } => &mut element.rounding,
            Self::VectorCast { rounding, .. }
            | Self::FloatConv { rounding, .. }
            | Self::IntToFloat { rounding, .. } => rounding,
            _ => return self,
        };
        *stated = Some(rounding);
        if let Some(environment) = self.environment_mut() {
            *environment = None;
        }
        self
    }

    /// The value a conversion converts: the operand of an [`Self::IntConv`],
    /// [`Self::PtrToInt`], [`Self::IntToPtr`], [`Self::IntToFloat`],
    /// [`Self::FloatToInt`] or [`Self::FloatConv`]; `None` for any other op.
    #[must_use]
    pub const fn converted_operand(&self) -> Option<SsaVarId> {
        match self {
            Self::IntConv { operand, .. }
            | Self::PtrToInt { operand, .. }
            | Self::IntToPtr { operand, .. }
            | Self::IntToFloat { operand, .. }
            | Self::FloatToInt { operand, .. }
            | Self::FloatConv { operand, .. } => Some(*operand),
            _ => None,
        }
    }

    /// The rounding a value conversion that can be inexact states: that of
    /// an [`Self::IntToFloat`], a [`Self::FloatConv`] or a
    /// [`Self::VectorCast`]. `None` for the environment's current mode, and for
    /// any other op.
    #[must_use]
    pub const fn stated_rounding(&self) -> Option<FloatRounding> {
        match self {
            Self::IntToFloat { rounding, .. }
            | Self::FloatConv { rounding, .. }
            | Self::VectorCast { rounding, .. } => *rounding,
            _ => None,
        }
    }

    /// The format a conversion of a float states its operand is in: that of
    /// a [`Self::FloatToInt`] or a [`Self::FloatConv`]. `None` where it states
    /// none, and for any other op.
    #[must_use]
    pub const fn stated_source(&self) -> Option<FloatFormat> {
        match self {
            Self::FloatToInt { source, .. } | Self::FloatConv { source, .. } => *source,
            _ => None,
        }
    }

    /// This op stating `source` as its operand's format, where it is a
    /// conversion of a float ([`Self::stated_source`]); any other op as it is.
    #[must_use]
    pub fn with_source(mut self, stated: Option<FloatFormat>) -> Self {
        if let Self::FloatToInt { source, .. } | Self::FloatConv { source, .. } = &mut self {
            *source = stated;
        }
        self
    }

    /// This op reading the environment's rounding mode `mode`, where it is an
    /// operation that can round by it ([`Self::environment`]); any other op
    /// as it is.
    #[must_use]
    pub fn with_environment(mut self, mode: Option<SsaVarId>) -> Self {
        if let Some(environment) = self.environment_mut() {
            *environment = mode;
        }
        self
    }

    /// This op stating `rounding`, where it is a conversion that can be
    /// inexact ([`Self::stated_rounding`]); any other op as it is.
    #[must_use]
    pub fn with_rounding(mut self, stated: Option<FloatRounding>) -> Self {
        if let Self::IntToFloat { rounding, .. }
        | Self::FloatConv { rounding, .. }
        | Self::VectorCast { rounding, .. } = &mut self
        {
            *rounding = stated;
        }
        self
    }

    /// This conversion as it reads an operand of type `operand_type`, when
    /// that is not the op it is.
    ///
    /// A front end chooses a conversion from the type its operand has when it
    /// lifts the instruction. A value whose definition construction places
    /// later has the reading instruction's guess then, or no type at all, and
    /// the definition it turns out to be can hold its bits otherwise: a stack
    /// slot read as an integer that holds a stored `double`, a frame register
    /// guessed to hold an address that holds a count. What the conversion
    /// becomes is what `intent` says the front end meant.
    ///
    /// For [`ConversionIntent::Bits`]:
    ///
    /// - A conversion of bits - an [`Self::IntConv`] that checks no
    ///   overflow, a [`Self::PtrToInt`], an [`Self::IntToPtr`] - is the
    ///   operand's bits at its target type, so it becomes the op that reads
    ///   them from what the operand holds: [`Self::PtrToInt`] from an
    ///   address, [`Self::IntConv`] from an integer, and from a float or a
    ///   vector a [`Self::Bitcast`] - through the unsigned integer of the
    ///   operand's width when the target is not that wide. An integer
    ///   conversion of a `double` otherwise reads as its value converted.
    /// - A conversion of values - [`Self::IntToFloat`], [`Self::FloatToInt`],
    ///   [`Self::FloatConv`] - converts from the domain its operation names,
    ///   whatever holds the operand: an integer in a float register
    ///   (AArch64's `scvtf s0, s0`) is read through the integer of its width,
    ///   with the sign the conversion reads it by, and a float in an integer
    ///   one through the float of its width.
    /// - An [`Self::IntConv`] that checks for overflow states a value range,
    ///   not a reinterpretation, and is left as it is.
    ///
    /// For [`ConversionIntent::Values`], every conversion becomes the one
    /// [`Self::conversion_between`] chooses from the operand's type, an
    /// unknown type read as an integer, as that function reads one.
    ///
    /// # Arguments
    ///
    /// * `operand_type` - The type of the value the conversion converts.
    /// * `intent` - What the front end meant the conversion to do.
    ///
    /// # Returns
    ///
    /// The restated conversion, or `None` when the op already is it, or, for
    /// bits, when the operand's type says nothing about them or the target
    /// does not name an integer or float type the restatement needs.
    #[must_use]
    pub fn restated_for_operand(
        &self,
        operand_type: &T::Type,
        intent: ConversionIntent,
    ) -> Option<Restated<T>> {
        if intent == ConversionIntent::Values {
            let (dest, operand, target, unsigned, overflow_check) = match self {
                Self::IntConv {
                    dest,
                    operand,
                    target,
                    unsigned,
                    overflow_check,
                }
                | Self::FloatToInt {
                    dest,
                    operand,
                    target,
                    unsigned,
                    overflow_check,
                    ..
                } => (*dest, *operand, target, *unsigned, *overflow_check),
                Self::IntToFloat {
                    dest,
                    operand,
                    target,
                    unsigned,
                    ..
                } => (*dest, *operand, target, *unsigned, false),
                Self::PtrToInt {
                    dest,
                    operand,
                    target,
                }
                | Self::IntToPtr {
                    dest,
                    operand,
                    target,
                } => (*dest, *operand, target, true, false),
                Self::FloatConv {
                    dest,
                    operand,
                    target,
                    ..
                } => (*dest, *operand, target, false, false),
                _ => return None,
            };
            let known = (!T::is_unknown(operand_type)).then_some(operand_type);
            // The rounding, the environment's mode and the source format the
            // conversion states go with it into whichever conversion it
            // becomes.
            let chosen = Self::conversion_between(
                dest,
                operand,
                target.clone(),
                known,
                unsigned,
                overflow_check,
            )
            .with_rounding(self.stated_rounding())
            .with_environment(self.environment())
            .with_source(
                self.stated_source()
                    .or_else(|| known.and_then(T::float_format)),
            );
            return (chosen != *self).then_some(Restated {
                through: None,
                op: chosen,
            });
        }
        let held = Held::of::<T>(operand_type)?;
        let width = T::bit_width(operand_type);
        // The bits of a float or a vector, at the target type: one bitcast
        // when it is as wide, else through the integer of their width.
        let bits_as = |dest: SsaVarId, operand: SsaVarId, target: &T::Type, then: Self| {
            if T::is_integer(target) && width.is_some() && T::bit_width(target) == width {
                return Some(Restated {
                    through: None,
                    op: Self::Bitcast {
                        dest,
                        operand,
                        target: target.clone(),
                    },
                });
            }
            Some(Restated {
                through: Some(T::integer_type(width?, false)?),
                op: then,
            })
        };
        match (self, held) {
            (
                Self::IntConv {
                    dest,
                    operand,
                    target,
                    overflow_check: false,
                    ..
                },
                Held::Address,
            ) => Some(Restated {
                through: None,
                op: Self::PtrToInt {
                    dest: *dest,
                    operand: *operand,
                    target: target.clone(),
                },
            }),
            (
                Self::IntConv {
                    dest,
                    operand,
                    target,
                    overflow_check: false,
                    ..
                }
                | Self::IntToPtr {
                    dest,
                    operand,
                    target,
                },
                Held::Float | Held::Vector,
            ) => bits_as(*dest, *operand, target, self.clone()),
            (
                Self::IntToPtr {
                    dest,
                    operand,
                    target,
                },
                Held::Address,
            ) => Some(Restated {
                through: None,
                op: Self::Bitcast {
                    dest: *dest,
                    operand: *operand,
                    target: target.clone(),
                },
            }),
            (
                Self::PtrToInt {
                    dest,
                    operand,
                    target,
                },
                Held::Integer | Held::Float | Held::Vector,
            ) => {
                let conversion = Self::IntConv {
                    dest: *dest,
                    operand: *operand,
                    target: target.clone(),
                    overflow_check: false,
                    unsigned: true,
                };
                if held == Held::Integer {
                    Some(Restated {
                        through: None,
                        op: conversion,
                    })
                } else {
                    bits_as(*dest, *operand, target, conversion)
                }
            }
            (Self::IntToFloat { unsigned, .. }, Held::Float | Held::Vector) => Some(Restated {
                through: Some(T::integer_type(width?, !*unsigned)?),
                op: self.clone(),
            }),
            // The source's own type is gone, so its format is the one the
            // conversion states, or else the IEEE one of the width it is held
            // at. Not at 16 bits, where binary16 and bfloat16 meet: there the
            // width says nothing, and the conversion is left as it is.
            (Self::FloatToInt { .. } | Self::FloatConv { .. }, Held::Integer | Held::Vector) => {
                let format = match self.stated_source() {
                    Some(format) => format,
                    None => FloatFormat::of_width(width?).filter(|_| width != Some(16))?,
                };
                Some(Restated {
                    through: Some(T::float_type(format)?),
                    op: self.clone(),
                })
            }
            _ => None,
        }
    }

    /// Builds the conversion that crosses from the `from` domain to the `to`
    /// domain.
    ///
    /// # Arguments
    ///
    /// * `dest` - The value receiving the result.
    /// * `operand` - The value converted.
    /// * `target` - The type converted to, recorded on the op.
    /// * `from` - The domain of the value converted.
    /// * `to` - The domain converted to.
    /// * `unsigned` - Whether an integer endpoint is read as unsigned
    ///   (zero-extended) rather than signed. A float-to-integer conversion
    ///   takes it from the integer type it converts to instead: the flag names
    ///   the destination's interpretation, which that type states, and a CIL
    ///   `conv.*.un` - the source read as unsigned - means nothing for a float
    ///   source.
    /// * `overflow_check` - Whether the op traps when the value does not fit
    ///   the destination.
    #[must_use]
    pub fn conversion(
        dest: SsaVarId,
        operand: SsaVarId,
        target: T::Type,
        from: ConversionDomain,
        to: ConversionDomain,
        unsigned: bool,
        overflow_check: bool,
    ) -> Self {
        match (from, to) {
            (ConversionDomain::Vector, ConversionDomain::Vector) => Self::VectorReinterpret {
                dest,
                value: operand,
                target_type: target,
            },
            (ConversionDomain::Vector, _) | (_, ConversionDomain::Vector) => Self::Bitcast {
                dest,
                operand,
                target,
            },
            (_, ConversionDomain::Ptr) => Self::IntToPtr {
                dest,
                operand,
                target,
            },
            (ConversionDomain::Float, ConversionDomain::Float) => Self::FloatConv {
                dest,
                operand,
                target,
                rounding: None,
                source: None,
                environment: None,
            },
            (ConversionDomain::Int | ConversionDomain::Ptr, ConversionDomain::Float) => {
                Self::IntToFloat {
                    dest,
                    operand,
                    target,
                    unsigned,
                    rounding: None,
                    environment: None,
                }
            }
            (ConversionDomain::Ptr, ConversionDomain::Int) => Self::PtrToInt {
                dest,
                operand,
                target,
            },
            (ConversionDomain::Float, ConversionDomain::Int) => Self::FloatToInt {
                dest,
                operand,
                source: None,
                unsigned: if T::is_integer(&target) {
                    !T::is_signed(&target)
                } else {
                    unsigned
                },
                target,
                overflow_check,
            },
            (ConversionDomain::Int, ConversionDomain::Int) => Self::IntConv {
                dest,
                operand,
                target,
                overflow_check,
                unsigned,
            },
        }
    }

    /// Builds the conversion from a value of type `from` to `target`: the one
    /// [`Self::conversion`] chooses for their domains. An unknown source is
    /// taken to be an integer.
    #[must_use]
    pub fn conversion_between(
        dest: SsaVarId,
        operand: SsaVarId,
        target: T::Type,
        from: Option<&T::Type>,
        unsigned: bool,
        overflow_check: bool,
    ) -> Self {
        let to = ConversionDomain::of::<T>(&target);
        // A float's format is the operand's, stated so it outlives the type.
        let source = from.and_then(T::float_format);
        let from = from.map_or(ConversionDomain::Int, ConversionDomain::of::<T>);
        Self::conversion(dest, operand, target, from, to, unsigned, overflow_check)
            .with_source(source)
    }

    /// Builds the op that reads a value of type `from` as `target`, a type of
    /// the same width: the bits unchanged, only their type restated.
    ///
    /// Across the pointer boundary that is [`Self::PtrToInt`] or
    /// [`Self::IntToPtr`], which say an address is being held as a number (or
    /// read back as an address) - the forms a type analysis reads the address
    /// through. Between integers it is [`Self::IntConv`] at the same width,
    /// which range and constant reasoning see through. A vector read in
    /// another lane layout is [`Self::VectorReinterpret`], and a vector read as
    /// a scalar or the reverse a [`Self::Bitcast`]: a lane-wise conversion
    /// would convert each lane, and a vector of eight bytes read as four
    /// halfwords has no lane for half of them. Anything else - an integer's
    /// bits read as a float, or the reverse - is [`Self::Bitcast`]; a value
    /// conversion there would change the bits.
    ///
    /// # Arguments
    ///
    /// * `dest` - The value receiving the result.
    /// * `operand` - The value reinterpreted.
    /// * `target` - The type it is read as.
    /// * `from` - The type it holds.
    #[must_use]
    pub fn reinterpretation(
        dest: SsaVarId,
        operand: SsaVarId,
        target: T::Type,
        from: &T::Type,
    ) -> Self {
        match (T::is_vector(from), T::is_vector(&target)) {
            (true, true) => {
                return Self::VectorReinterpret {
                    dest,
                    value: operand,
                    target_type: target,
                };
            }
            (true, false) | (false, true) => {
                return Self::Bitcast {
                    dest,
                    operand,
                    target,
                };
            }
            (false, false) => {}
        }
        match (
            ConversionDomain::of::<T>(from),
            ConversionDomain::of::<T>(&target),
        ) {
            (ConversionDomain::Ptr, ConversionDomain::Int) => Self::PtrToInt {
                dest,
                operand,
                target,
            },
            (ConversionDomain::Int, ConversionDomain::Ptr) => Self::IntToPtr {
                dest,
                operand,
                target,
            },
            (ConversionDomain::Int, ConversionDomain::Int) => Self::IntConv {
                dest,
                operand,
                unsigned: !T::is_signed(&target),
                target,
                overflow_check: false,
            },
            _ => Self::Bitcast {
                dest,
                operand,
                target,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConversionDomain::{self, Float, Int, Ptr, Vector},
        *,
    };
    use crate::testing::{MockTarget, MockType};

    /// A reinterpretation keeps the bits: an address held as a number is a
    /// `ptrtoint`, not a bitcast a type analysis cannot read the address
    /// through; an integer's bits read as a float stay a bitcast, where a
    /// value conversion would change them.
    #[test]
    fn a_reinterpretation_names_what_the_bits_are() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let op = |from: MockType, to: MockType| {
            SsaOp::<MockTarget>::reinterpretation(dest, operand, to, &from)
        };
        assert!(matches!(
            op(MockType::Ptr, MockType::U64),
            SsaOp::PtrToInt { .. }
        ));
        assert!(matches!(
            op(MockType::U64, MockType::Ptr),
            SsaOp::IntToPtr { .. }
        ));
        assert!(matches!(
            op(MockType::I32, MockType::U32),
            SsaOp::IntConv { unsigned: true, .. }
        ));
        assert!(matches!(
            op(MockType::U64, MockType::F64),
            SsaOp::Bitcast { .. }
        ));
        assert!(matches!(
            op(MockType::F64, MockType::U64),
            SsaOp::Bitcast { .. }
        ));
    }

    /// A vector's bits read in another lane layout are reinterpreted whole,
    /// never converted lane by lane: four 32-bit lanes read as two 64-bit
    /// floats, or as sixteen 4-bit lanes of the low half, keep every bit,
    /// where an `IntConv` would convert each lane it pairs. A vector read as a
    /// scalar of its width, and the reverse, is a bitcast. The control, two
    /// scalar integers, is the `IntConv` range reasoning sees through.
    #[test]
    fn a_vector_reinterpretation_keeps_its_bits_whole() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let op = |from: MockType, to: MockType| {
            SsaOp::<MockTarget>::reinterpretation(dest, operand, to, &from)
        };
        assert!(matches!(
            op(MockType::V4I32, MockType::V2F64),
            SsaOp::VectorReinterpret { .. }
        ));
        assert!(matches!(
            op(MockType::V2F64, MockType::V4I32),
            SsaOp::VectorReinterpret { .. }
        ));
        assert!(matches!(
            op(MockType::V4I32, MockType::U128),
            SsaOp::Bitcast { .. }
        ));
        assert!(matches!(
            op(MockType::U128, MockType::V4I32),
            SsaOp::Bitcast { .. }
        ));
        assert!(matches!(
            op(MockType::U64, MockType::V16I4),
            SsaOp::Bitcast { .. }
        ));
        assert!(matches!(
            op(MockType::U64, MockType::I64),
            SsaOp::IntConv {
                unsigned: false,
                ..
            }
        ));
    }

    /// Each pair of domains is crossed by its own op.
    #[test]
    fn each_domain_crossing_has_its_op() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let op = |from, to| {
            SsaOp::<MockTarget>::conversion(dest, operand, MockType::I64, from, to, false, false)
        };
        assert!(matches!(op(Int, Int), SsaOp::IntConv { .. }));
        assert!(matches!(op(Int, Ptr), SsaOp::IntToPtr { .. }));
        assert!(matches!(op(Ptr, Int), SsaOp::PtrToInt { .. }));
        assert!(matches!(op(Int, Float), SsaOp::IntToFloat { .. }));
        assert!(matches!(op(Float, Int), SsaOp::FloatToInt { .. }));
        assert!(matches!(op(Float, Float), SsaOp::FloatConv { .. }));
        // A vector's bits are its lanes: no integer or value conversion
        // crosses into or out of one, whatever the other side.
        assert!(matches!(
            op(Vector, Vector),
            SsaOp::VectorReinterpret { .. }
        ));
        for scalar in [Int, Ptr, Float] {
            assert!(matches!(op(Vector, scalar), SsaOp::Bitcast { .. }));
            assert!(matches!(op(scalar, Vector), SsaOp::Bitcast { .. }));
        }
    }

    /// A float-to-integer conversion states the sign of the integer it
    /// converts to, whatever flag the front end passed - CIL's `conv.u4`
    /// passes the source's `.un`, which a float source does not have. The
    /// control, an integer-to-float conversion, keeps the flag: CIL's
    /// `conv.r.un` reads a sign-agnostic stack integer as unsigned.
    #[test]
    fn a_float_to_integer_conversion_takes_its_targets_sign() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let to_int = |target, unsigned| {
            SsaOp::<MockTarget>::conversion(dest, operand, target, Float, Int, unsigned, false)
        };
        assert!(matches!(
            to_int(MockType::U32, false),
            SsaOp::FloatToInt { unsigned: true, .. }
        ));
        assert!(matches!(
            to_int(MockType::I32, true),
            SsaOp::FloatToInt {
                unsigned: false,
                ..
            }
        ));
        let to_float =
            SsaOp::<MockTarget>::conversion(dest, operand, MockType::F64, Int, Float, true, false);
        assert!(matches!(to_float, SsaOp::IntToFloat { unsigned: true, .. }));
    }

    /// Each conversion as an operand of each holding reads it: bits through
    /// the op that reads them from what holds them, values from the domain
    /// their operation names.
    #[test]
    fn a_conversion_is_restated_for_what_its_operand_holds() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let restated = |op: SsaOp<MockTarget>, held: MockType| {
            op.restated_for_operand(&held, ConversionIntent::Bits)
                .map(|restated| (restated.through, restated.op))
        };
        let int_conv = |target, overflow_check| SsaOp::<MockTarget>::IntConv {
            dest,
            operand,
            target,
            overflow_check,
            unsigned: true,
        };
        // An integer conversion of an address is the address as a number.
        assert!(matches!(
            restated(int_conv(MockType::U64, false), MockType::Ptr),
            Some((
                None,
                SsaOp::PtrToInt {
                    target: MockType::U64,
                    ..
                }
            ))
        ));
        // Of a double: its bits, at once when as wide, else through them.
        assert!(matches!(
            restated(int_conv(MockType::U64, false), MockType::F64),
            Some((
                None,
                SsaOp::Bitcast {
                    target: MockType::U64,
                    ..
                }
            ))
        ));
        assert!(matches!(
            restated(int_conv(MockType::U32, false), MockType::F64),
            Some((
                Some(MockType::U64),
                SsaOp::IntConv {
                    target: MockType::U32,
                    ..
                }
            ))
        ));
        // Of a 128-bit vector: through the integer as wide.
        assert!(matches!(
            restated(int_conv(MockType::U32, false), MockType::V4I32),
            Some((
                Some(MockType::U128),
                SsaOp::IntConv {
                    target: MockType::U32,
                    ..
                }
            ))
        ));
        // A checked conversion states a range, and an integer is what it reads.
        assert!(restated(int_conv(MockType::U32, true), MockType::F64).is_none());
        assert!(restated(int_conv(MockType::U32, false), MockType::I64).is_none());
        // An address taken as a number from an integer is the integer.
        let ptr_to_int = SsaOp::<MockTarget>::PtrToInt {
            dest,
            operand,
            target: MockType::U64,
        };
        assert!(matches!(
            restated(ptr_to_int.clone(), MockType::I64),
            Some((None, SsaOp::IntConv { unsigned: true, .. }))
        ));
        assert!(restated(ptr_to_int, MockType::Ptr).is_none());
        // A number taken as an address from an address is that address.
        let int_to_ptr = SsaOp::<MockTarget>::IntToPtr {
            dest,
            operand,
            target: MockType::Ptr,
        };
        assert!(matches!(
            restated(int_to_ptr.clone(), MockType::Ptr),
            Some((
                None,
                SsaOp::Bitcast {
                    target: MockType::Ptr,
                    ..
                }
            ))
        ));
        assert!(matches!(
            restated(int_to_ptr, MockType::F64),
            Some((Some(MockType::U64), SsaOp::IntToPtr { .. }))
        ));
        // A value conversion reads its domain through the operand's bits, the
        // integer side with the sign it is read by.
        let to_float = |unsigned| SsaOp::<MockTarget>::IntToFloat {
            dest,
            operand,
            target: MockType::F64,
            unsigned,
            rounding: None,
            environment: None,
        };
        assert!(matches!(
            restated(to_float(false), MockType::F64),
            Some((Some(MockType::I64), SsaOp::IntToFloat { .. }))
        ));
        assert!(matches!(
            restated(to_float(true), MockType::F64),
            Some((Some(MockType::U64), SsaOp::IntToFloat { .. }))
        ));
        assert!(restated(to_float(false), MockType::I64).is_none());
        let to_int = SsaOp::<MockTarget>::FloatToInt {
            dest,
            operand,
            target: MockType::I64,
            unsigned: false,
            overflow_check: false,
            source: None,
        };
        assert!(matches!(
            restated(to_int.clone(), MockType::U64),
            Some((Some(MockType::F64), SsaOp::FloatToInt { .. }))
        ));
        assert!(restated(to_int, MockType::F64).is_none());
        // A type that says nothing about its bits leaves the conversion alone.
        assert!(restated(int_conv(MockType::U64, false), MockType::Unknown).is_none());
    }

    /// **A stated rounding goes with the conversion it is restated as.** An
    /// integer-to-float conversion rounding up, whose operand turns out to be a
    /// double, becomes the float conversion of that double rounding up, and it
    /// shows its rounding as it displays.
    #[test]
    fn a_restated_conversion_keeps_its_rounding() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let up = SsaOp::<MockTarget>::IntToFloat {
            dest,
            operand,
            target: MockType::BF16,
            unsigned: false,
            rounding: Some(FloatRounding::Up),
            environment: None,
        };
        assert!(up.to_string().contains("inttofloat.bf16.up"), "{up}");
        let restated = up
            .restated_for_operand(&MockType::F64, ConversionIntent::Values)
            .expect("a double operand converts as a float")
            .op;
        assert!(matches!(restated, SsaOp::FloatConv { .. }), "{restated}");
        assert_eq!(restated.stated_rounding(), Some(FloatRounding::Up));
        assert!(restated.to_string().contains(".up"), "{restated}");
        // An op that rounds nothing takes no rounding.
        let copy = SsaOp::<MockTarget>::Copy { dest, src: operand };
        assert_eq!(copy.clone().with_rounding(Some(FloatRounding::Up)), copy);
    }

    /// **A float held as an integer is read in the format its conversion
    /// states.** A bfloat16 stored and loaded back as a halfword reaches its
    /// conversion as an integer of 16 bits, and the conversion stating
    /// bfloat16 reads it through bfloat16. One stating no format is not
    /// restated at 16 bits, where binary16 and bfloat16 meet; at 64 bits, the
    /// control, the width names the double it is.
    #[test]
    fn a_float_held_as_an_integer_is_read_in_its_stated_format() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let truncate = |source: Option<FloatFormat>| SsaOp::<MockTarget>::FloatToInt {
            dest,
            operand,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
            source,
        };
        let stated = truncate(Some(FloatFormat::BFloat16))
            .restated_for_operand(&MockType::I16, ConversionIntent::Bits)
            .expect("read through the format it states");
        assert_eq!(stated.through, Some(MockType::BF16));
        assert!(
            truncate(None)
                .restated_for_operand(&MockType::I16, ConversionIntent::Bits)
                .is_none(),
            "sixteen bits name no format"
        );
        let wide = truncate(None)
            .restated_for_operand(&MockType::I64, ConversionIntent::Bits)
            .expect("sixty-four bits are a double");
        assert_eq!(wide.through, Some(MockType::F64));
    }

    /// Converting values, a conversion is the one its operand's type and its
    /// target choose: `conv.i8` of a `float64` truncates it, and of an
    /// integer it stays an integer conversion.
    #[test]
    fn a_value_conversion_is_chosen_from_both_endpoints() {
        let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        let int_conv = SsaOp::<MockTarget>::IntConv {
            dest,
            operand,
            target: MockType::I64,
            overflow_check: false,
            unsigned: false,
        };
        let values = |op: &SsaOp<MockTarget>, held: MockType| {
            op.restated_for_operand(&held, ConversionIntent::Values)
                .map(|restated| (restated.through, restated.op))
        };
        assert!(matches!(
            values(&int_conv, MockType::F64),
            Some((
                None,
                SsaOp::FloatToInt {
                    target: MockType::I64,
                    ..
                }
            ))
        ));
        assert!(values(&int_conv, MockType::I32).is_none());
        // The same op moving bits reads the double's bits instead.
        assert!(matches!(
            int_conv
                .restated_for_operand(&MockType::F64, ConversionIntent::Bits)
                .map(|restated| restated.op),
            Some(SsaOp::Bitcast { .. })
        ));
        let to_float = SsaOp::<MockTarget>::IntToFloat {
            dest,
            operand,
            target: MockType::F64,
            unsigned: false,
            rounding: None,
            environment: None,
        };
        assert!(matches!(
            values(&to_float, MockType::F64),
            Some((None, SsaOp::FloatConv { .. }))
        ));
        assert!(values(&to_float, MockType::Unknown).is_none());
    }

    /// A type's domain is the target's classification of it.
    #[test]
    fn a_types_domain_is_the_targets_classification() {
        assert_eq!(
            ConversionDomain::of::<MockTarget>(&MockType::F64),
            ConversionDomain::Float
        );
        assert_eq!(
            ConversionDomain::of::<MockTarget>(&MockType::I32),
            ConversionDomain::Int
        );
        assert_eq!(
            ConversionDomain::of::<MockTarget>(&MockType::V4I32),
            ConversionDomain::Vector
        );
        assert_eq!(
            ConversionDomain::of::<MockTarget>(&MockType::V2F64),
            ConversionDomain::Vector
        );
    }
}
