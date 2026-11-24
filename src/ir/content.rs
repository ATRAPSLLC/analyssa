//! Fallible canonical visitation of SSA operation content.
//!
//! An operation, and every constant, layout, call contract and access
//! qualifier it carries, is visited as a stream of bytes and borrowed
//! metadata ([`ContentSink`](crate::ir::content::ContentSink)), so a host
//! derives an identity of its complete semantic content. Each stream starts with its format version
//! ([`OPERATION_CONTENT_VERSION`](crate::ir::content::OPERATION_CONTENT_VERSION),
//! [`CONSTANT_CONTENT_VERSION`](crate::ir::content::CONSTANT_CONTENT_VERSION)),
//! which changes whenever an encoding does. Raise wrappers encode their
//! metadata before their nested operation, keeping traversal at constant
//! call-stack depth.

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{ops::SsaOp, value::ConstValue},
    metadata::MetadataRef,
    target::Target,
};

/// Receives canonical content without owning an SSA snapshot.
///
/// Byte callbacks carry the primitive and payload representation. Opaque
/// metadata stays borrowed; metadata callbacks must supply a stable,
/// domain-separated encoding for the host's references, since operation bytes
/// alone do not identify opaque target values. A consumer that cannot
/// represent a metadata domain must refuse it. Refusal propagates immediately
/// through visitation.
///
/// A refusal leaves the consumed prefix in the sink. Consumers that publish a
/// digest or persistent snapshot must commit it only after visitation succeeds,
/// or discard that prefix before retrying.
///
/// # Examples
///
/// This consumer collects the bytes and refuses opaque metadata without
/// inspecting it. A native host supplies its own reference encoding.
///
/// ```rust
/// use analyssa::{
///     ir::{
///         ConstValue, SsaOp, SsaVarId,
///         content::{ContentSink, OPERATION_CONTENT_VERSION},
///     },
///     metadata::MetadataRef,
///     testing::MockTarget,
/// };
///
/// struct PrimitiveContent {
///     encoded: Vec<u8>,
/// }
///
/// impl ContentSink<MockTarget> for PrimitiveContent {
///     type Error = &'static str;
///
///     fn bytes(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
///         self.encoded.extend_from_slice(bytes);
///         Ok(())
///     }
///
///     fn metadata(&mut self, _: MetadataRef<'_, MockTarget>) -> Result<(), Self::Error> {
///         Err("metadata requires host encoding")
///     }
/// }
///
/// let op = SsaOp::<MockTarget>::Const {
///     dest: SsaVarId::from_index(0),
///     value: ConstValue::I32(7),
/// };
/// let mut sink = PrimitiveContent { encoded: Vec::new() };
/// op.write_content(&mut sink)?;
/// assert!(sink.encoded.starts_with(&OPERATION_CONTENT_VERSION.to_le_bytes()));
/// # Ok::<(), &'static str>(())
/// ```
pub trait ContentSink<T: Target> {
    /// The caller's refusal or output error.
    type Error;

    /// Consumes the next encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns the consumer's output error; the caller does not roll back the
    /// bytes it has already accepted.
    fn bytes(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Visits a borrowed opaque value.
    ///
    /// Encodes its metadata domain and complete semantic identity without using
    /// a display string as an identity. Unsupported domains must be refused.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-domain or output error. A consumed prefix is
    /// allowed; later source fields are not visited.
    fn metadata(&mut self, value: MetadataRef<'_, T>) -> Result<(), Self::Error>;
}

/// Format version leading every operation's content stream.
pub const OPERATION_CONTENT_VERSION: u16 = 9;

/// Format version leading every standalone constant's content stream.
pub const CONSTANT_CONTENT_VERSION: u16 = 2;

impl<T: Target> ConstValue<T> {
    /// Writes a borrowed constant through the shared canonical content visitor.
    ///
    /// The stream starts with [`CONSTANT_CONTENT_VERSION`] and uses the same
    /// value encoding as SSA operations. Payload bytes remain borrowed and use batches of at most
    /// 128 bytes; opaque metadata uses the sink's own encoding.
    /// No temporary operation, clone or serialized value is constructed.
    ///
    /// # Errors
    ///
    /// Returns the first refusal from `sink` without visiting subsequent fields.
    /// The sink retains any consumed prefix and owns its publication or rollback.
    pub fn write_content<S: ContentSink<T> + ?Sized>(&self, sink: &mut S) -> Result<(), S::Error> {
        let mut writer = ContentWriter::new(sink);
        CONSTANT_CONTENT_VERSION.emit(&mut writer)?;
        self.emit(&mut writer)
    }
}

impl<T: Target> SsaOp<T> {
    /// Writes every operation field through a fallible canonical content visitor.
    ///
    /// The stream starts with [`OPERATION_CONTENT_VERSION`]. It encodes
    /// primitive numbers in little-endian order, indices
    /// and lengths as 64-bit unsigned values, and enum variants by explicit
    /// schema tags. Byte payloads use batches of at most 128 bytes. The target's
    /// opaque values remain borrowed. No standard hash, clone or serialization
    /// method of a host value is called.
    ///
    /// # Errors
    ///
    /// Returns the first refusal from `sink` without visiting subsequent fields.
    /// The sink retains any consumed prefix and owns its publication or rollback.
    pub fn write_content<S: ContentSink<T> + ?Sized>(&self, sink: &mut S) -> Result<(), S::Error> {
        let mut writer = ContentWriter::new(sink);
        OPERATION_CONTENT_VERSION.emit(&mut writer)?;
        self.emit(&mut writer)
    }
}

/// The content tests, and the probe sink other modules' tests encode through.
#[cfg(test)]
pub mod tests {
    use std::{
        cell::Cell,
        num::{NonZeroU8, NonZeroU16},
        rc::Rc,
    };

    use super::*;
    use crate::{
        ir::{
            float::{FloatExceptionLayout, FloatExceptions},
            ops::{
                AccessPermission, AccessQualifiers, CallData, CallEffects, CallInput, CallInputKey,
                CallMemory, CallOutput, CallTermination, Callee, Capture, Computation,
                FpuControlKind, Kernel, LaneControl, Locality, ReturnTransfer, VectorBinaryKind,
                VectorCryptoKind, VectorElement, VectorInactivePolicy, VectorPredicateLayout,
                VectorRounding, VectorStatus,
            },
            value::{ConstValue, DecryptedArrayData},
            variable::SsaVarId,
        },
        metadata::tests::{DefaultTarget, EqualityProbe},
        target::VectorTailPolicy,
        testing::{MockTarget, MockType},
    };

    /// A test consumer with an exact event limit and an independent metadata representation.
    pub struct Probe {
        /// Bytes consumed before refusal.
        pub data: Vec<u8>,
        /// Accepted sink events.
        pub events: usize,
        /// Calls incorrectly made after the first refusal.
        pub after_refusal: usize,
        /// Metadata domains visited in order.
        pub domains: Vec<u8>,
        /// Whether the consumer supports host metadata.
        metadata_supported: bool,
        /// Maximum number of accepted events.
        limit: usize,
        /// Whether any event has already been refused.
        refused: bool,
    }

    impl Probe {
        /// Creates a consumer with a fixed event allowance.
        pub fn new(limit: usize) -> Self {
            Self {
                data: Vec::new(),
                events: 0,
                after_refusal: 0,
                domains: Vec::new(),
                metadata_supported: true,
                limit,
                refused: false,
            }
        }

        /// Accepts one event, refusing without inspecting its value past the limit.
        fn accept(&mut self) -> Result<(), ()> {
            if self.refused {
                self.after_refusal += 1;
                return Err(());
            }
            if self.events == self.limit {
                self.refused = true;
                return Err(());
            }
            self.events += 1;
            Ok(())
        }
    }

    impl ContentSink<MockTarget> for Probe {
        type Error = ();

        fn bytes(&mut self, bytes: &[u8]) -> Result<(), ()> {
            self.accept()?;
            self.data.extend_from_slice(bytes);
            Ok(())
        }

        fn metadata(&mut self, value: MetadataRef<'_, MockTarget>) -> Result<(), ()> {
            self.accept()?;
            if !self.metadata_supported {
                self.refused = true;
                return Err(());
            }
            let (domain, text) = match value {
                MetadataRef::Type(value) => (0, format!("{value:?}")),
                MetadataRef::TypeRef(value) => (1, format!("{value:?}")),
                MetadataRef::Method(value) => (2, format!("{value:?}")),
                MetadataRef::Field(value) => (3, format!("{value:?}")),
                MetadataRef::Signature(value) => (4, format!("{value:?}")),
                MetadataRef::Symbol(value) => (5, format!("{value:?}")),
            };
            self.domains.push(domain);
            self.data.push(0xf0 + domain);
            self.data
                .extend_from_slice(&(text.len() as u64).to_le_bytes());
            self.data.extend_from_slice(text.as_bytes());
            Ok(())
        }
    }

    /// Rejects every opaque domain without inspecting host data.
    struct RejectOpaque;

    impl ContentSink<DefaultTarget> for RejectOpaque {
        type Error = ();

        fn bytes(&mut self, _: &[u8]) -> Result<(), ()> {
            Ok(())
        }
        fn metadata(&mut self, _: MetadataRef<'_, DefaultTarget>) -> Result<(), ()> {
            Err(())
        }
    }

    /// Refused metadata cannot invoke a host's unrestricted clone, equality or hash.
    #[test]
    fn content_refusal_never_calls_unrestricted_host_traits() {
        let raw_calls = Rc::new(Cell::new(0usize));
        let probe = || EqualityProbe::new(Rc::clone(&raw_calls));
        let id = SsaVarId::from_index(1);
        let ops = [
            SsaOp::<DefaultTarget>::IntConv {
                dest: id,
                operand: id,
                target: probe(),
                overflow_check: false,
                unsigned: false,
            },
            SsaOp::Const {
                dest: id,
                value: ConstValue::Type(probe()),
            },
            SsaOp::Const {
                dest: id,
                value: ConstValue::MethodHandle(probe()),
            },
            SsaOp::Const {
                dest: id,
                value: ConstValue::FieldHandle(probe()),
            },
            SsaOp::Call(Box::new(crate::ir::CallData {
                dest: None,
                callee: crate::ir::Callee::Indirect {
                    fptr: id,
                    signature: probe(),
                },
                args: vec![id; 1024],
                effects: None,
                tail: false,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: Vec::new(),
            })),
            SsaOp::Const {
                dest: id,
                value: ConstValue::Symbol(probe()),
            },
        ];
        for op in ops {
            assert_eq!(op.write_content(&mut RejectOpaque), Err(()));
            assert_eq!(raw_calls.get(), 0);
        }
    }

    /// Checks an independent wire oracle and every refusal boundary of an operation.
    fn assert_operation_content(op: &SsaOp<MockTarget>, expected: &[u8], domains: &[u8]) {
        let mut complete = Probe::new(usize::MAX);
        op.write_content(&mut complete).unwrap();
        assert_eq!(complete.data, expected);
        assert_eq!(complete.domains, domains);
        for limit in 0..complete.events {
            let mut refused = Probe::new(limit);
            assert!(op.write_content(&mut refused).is_err());
            assert_eq!(refused.after_refusal, 0);
            assert!(expected.starts_with(&refused.data));
        }
    }

    /// Dispatch, nullable results, arguments, effects and tail status have independent bytes.
    #[test]
    fn unified_call_content_matches_independent_dispatch_oracles() {
        let method = [vec![0xf2], 2u64.to_le_bytes().to_vec(), b"42".to_vec()].concat();
        let constraint = [vec![0xf1], 1u64.to_le_bytes().to_vec(), b"9".to_vec()].concat();
        let signature = [vec![0xf4], 1u64.to_le_bytes().to_vec(), b"7".to_vec()].concat();
        for (callee, encoded, domains) in [
            (
                Callee::Direct(42),
                [vec![0, 0], method.clone()].concat(),
                vec![2],
            ),
            (
                Callee::Virtual(42),
                [vec![1, 0], method.clone()].concat(),
                vec![2],
            ),
            (
                Callee::Constrained {
                    method: 42,
                    constraint_type: 9,
                    instance: false,
                },
                [vec![2, 0], method.clone(), constraint.clone(), vec![0]].concat(),
                vec![2, 1],
            ),
            (
                Callee::Constrained {
                    method: 42,
                    constraint_type: 9,
                    instance: true,
                },
                [vec![2, 0], method.clone(), constraint, vec![1]].concat(),
                vec![2, 1],
            ),
            (
                Callee::Indirect {
                    fptr: SsaVarId::from_index(17),
                    signature: 7,
                },
                [vec![3, 0], 17u64.to_le_bytes().to_vec(), signature].concat(),
                vec![4],
            ),
        ] {
            for dest in [None, Some(SsaVarId::from_index(3))] {
                for tail in [false, true] {
                    for effects in [
                        None,
                        Some(CallEffects::pure()),
                        Some(CallEffects::unknown()),
                        Some(CallEffects {
                            memory: CallMemory::Only(vec![]),
                            world: true,
                            calls_back: false,
                            captures: Capture::Any,
                            termination: CallTermination::NeverReturns,
                            returns_twice: true,
                            may_throw: false,
                        }),
                    ] {
                        let effect_bytes = match effects.as_ref() {
                            None => vec![0],
                            Some(effect) if effect == &CallEffects::pure() => [
                                vec![1, 1, 0],
                                0u64.to_le_bytes().to_vec(),
                                vec![0, 0, 0, 0, 0, 0, 0],
                            ]
                            .concat(),
                            Some(effect) if effect == &CallEffects::unknown() => {
                                vec![1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
                            }
                            Some(_) => [
                                vec![1, 1, 0],
                                0u64.to_le_bytes().to_vec(),
                                vec![1, 0, 1, 2, 0, 1, 0],
                            ]
                            .concat(),
                        };
                        let mut expected = vec![9, 0, 109, 0, u8::from(dest.is_some())];
                        if dest.is_some() {
                            expected.extend_from_slice(&3u64.to_le_bytes());
                        }
                        expected.extend_from_slice(&encoded);
                        expected.extend_from_slice(&2u64.to_le_bytes());
                        expected.extend_from_slice(&11u64.to_le_bytes());
                        expected.extend_from_slice(&13u64.to_le_bytes());
                        expected.extend_from_slice(&0u64.to_le_bytes());
                        expected.extend_from_slice(&effect_bytes);
                        expected.push(u8::from(tail));
                        expected.extend_from_slice(&0u64.to_le_bytes()); // no staged candidates
                        let op = SsaOp::Call(Box::new(CallData {
                            dest,
                            callee: callee.clone(),
                            args: vec![SsaVarId::from_index(11), SsaVarId::from_index(13)],
                            effects,
                            tail,
                            implicit_inputs: Default::default(),
                            results: Default::default(),
                            staged: Vec::new(),
                        }));
                        assert_operation_content(&op, &expected, &domains);
                    }
                }
            }
        }
    }

    /// Staged offsets and values have independent bytes and propagate every sink refusal.
    #[test]
    fn staged_call_content_retains_offsets_and_values() {
        let mut call = CallData::<MockTarget>::new(None, Callee::Direct(9), vec![]);
        call.staged = vec![
            crate::ir::StagedValue {
                offset: -8,
                value: SsaVarId::from_index(7),
            },
            crate::ir::StagedValue {
                offset: 16,
                value: SsaVarId::from_index(11),
            },
        ];
        let mut expected = vec![9, 0, 109, 0, 0, 0, 0, 0xf2];
        expected.extend_from_slice(&1u64.to_le_bytes());
        expected.push(b'9');
        expected.extend_from_slice(&0u64.to_le_bytes()); // ordinary arguments
        expected.extend_from_slice(&0u64.to_le_bytes()); // implicit inputs
        expected.extend_from_slice(&[0, 0]); // effects and tail control
        expected.extend_from_slice(&2u64.to_le_bytes());
        expected.extend_from_slice(&(-8i64).to_le_bytes());
        expected.extend_from_slice(&7u64.to_le_bytes());
        expected.extend_from_slice(&16i64.to_le_bytes());
        expected.extend_from_slice(&11u64.to_le_bytes());
        assert_operation_content(&SsaOp::from(call), &expected, &[2]);
    }

    /// Implicit keys are 32-bit identities; their SSA uses retain 64-bit wire indices.
    #[test]
    fn call_input_content_matches_independent_key_and_value_oracles() {
        for inputs in [
            vec![],
            vec![(0u32, 7u64)],
            vec![(u32::MAX, 7), (17, 11)],
            vec![(17, 7), (19, 11), (17, 13)],
        ] {
            let mut call = CallData::<MockTarget>::new(None, Callee::Direct(9), vec![]);
            for (key, value) in &inputs {
                call = call.with_implicit_input(
                    crate::ir::CallInputKey::new(*key),
                    SsaVarId::from_index(*value as usize),
                );
            }
            let mut expected = vec![9, 0, 109, 0, 0, 0, 0, 0xf2];
            expected.extend_from_slice(&1u64.to_le_bytes());
            expected.push(b'9');
            expected.extend_from_slice(&0u64.to_le_bytes());
            expected.extend_from_slice(&(inputs.len() as u64).to_le_bytes());
            for (key, value) in &inputs {
                expected.extend_from_slice(&key.to_le_bytes());
                expected.extend_from_slice(&value.to_le_bytes());
            }
            expected.extend_from_slice(&[0, 0]); // no stated effects, ordinary return
            expected.extend_from_slice(&0u64.to_le_bytes()); // no staged candidates
            assert_operation_content(&SsaOp::from(call), &expected, &[2]);
        }
    }

    /// Every access qualifier contributes its full value to operation content.
    #[test]
    fn access_content_matches_independent_qualifier_oracles() {
        for address_space in [None, NonZeroU16::new(1), NonZeroU16::new(u16::MAX)] {
            for (permission, tag) in [
                (AccessPermission::Current, 0u16),
                (AccessPermission::Unprivileged, 1),
                (AccessPermission::Execute, 2),
                (AccessPermission::ShadowStack, 3),
                (AccessPermission::UnprivilegedShadowStack, 4),
            ] {
                for (locality, locality_tag) in
                    [(Locality::Temporal, 0u16), (Locality::NonTemporal, 1)]
                {
                    for volatile in [false, true] {
                        for (alignment, required) in [
                            (None, None),
                            (NonZeroU8::new(1), None),
                            (NonZeroU8::new(2), None),
                            (NonZeroU8::new(4), None),
                            (None, NonZeroU8::new(16)),
                            (NonZeroU8::new(4), NonZeroU8::new(4)),
                        ] {
                            let op = SsaOp::LoadStaticField {
                                dest: SsaVarId::from_index(3),
                                field: 42,
                                access: AccessQualifiers {
                                    address_space,
                                    permission,
                                    locality,
                                    volatile,
                                    alignment,
                                    required,
                                },
                            };
                            let mut expected = vec![9, 0, 86, 0];
                            expected.extend_from_slice(&3u64.to_le_bytes());
                            expected.push(0xf3);
                            expected.extend_from_slice(&2u64.to_le_bytes());
                            expected.extend_from_slice(b"42");
                            expected.push(u8::from(address_space.is_some()));
                            if let Some(space) = address_space {
                                expected.extend_from_slice(&space.get().to_le_bytes());
                            }
                            expected.extend_from_slice(&tag.to_le_bytes());
                            expected.extend_from_slice(&locality_tag.to_le_bytes());
                            expected.push(u8::from(volatile));
                            expected.push(u8::from(alignment.is_some()));
                            if let Some(bytes) = alignment {
                                expected.push(bytes.get());
                            }
                            expected.push(u8::from(required.is_some()));
                            if let Some(bytes) = required {
                                expected.push(bytes.get());
                            }
                            assert_operation_content(&op, &expected, &[3]);
                        }
                    }
                }
            }
        }
    }

    /// Nullable pointer constraints and readonly element addresses cannot share snapshots.
    #[test]
    fn pointer_content_matches_independent_constraint_and_readonly_oracles() {
        for constraint in [None, Some(0), Some(9)] {
            let op = SsaOp::LoadFunctionPtr {
                dest: SsaVarId::from_index(3),
                method: 42,
                constraint,
            };
            let mut expected = vec![9, 0, 112, 0];
            expected.extend_from_slice(&3u64.to_le_bytes());
            expected.push(0xf2);
            expected.extend_from_slice(&2u64.to_le_bytes());
            expected.extend_from_slice(b"42");
            expected.push(u8::from(constraint.is_some()));
            if let Some(value) = constraint {
                expected.push(0xf1);
                expected.extend_from_slice(&1u64.to_le_bytes());
                expected.push(if value == 0 { b'0' } else { b'9' });
            }
            assert_operation_content(
                &op,
                &expected,
                if constraint.is_some() { &[2, 1] } else { &[2] },
            );
        }
        for readonly in [false, true] {
            let op = SsaOp::LoadElementAddr {
                dest: SsaVarId::from_index(3),
                array: SsaVarId::from_index(11),
                index: SsaVarId::from_index(13),
                elem_type: 9,
                readonly,
            };
            let expected = [
                vec![9, 0, 92, 0],
                3u64.to_le_bytes().to_vec(),
                11u64.to_le_bytes().to_vec(),
                13u64.to_le_bytes().to_vec(),
                vec![0xf1],
                1u64.to_le_bytes().to_vec(),
                vec![b'9', u8::from(readonly)],
            ]
            .concat();
            assert_operation_content(&op, &expected, &[1]);
        }
    }

    /// Vector passthrough policy changes content even when every other field agrees.
    #[test]
    fn vector_content_records_absent_passthrough_policy() {
        let kind = Computation {
            kernel: Kernel::Binary(VectorBinaryKind::Add),
            element: VectorElement::integer(32, None),
            predicate: VectorPredicateLayout::Packed,
            rounding: VectorRounding::None,
            status: VectorStatus::None,
            canonical_nan: false,
            lanes: Some(LaneControl {
                mask_tail: false,
                tail: VectorTailPolicy::Undisturbed,
                inactive: Some(VectorInactivePolicy::Undisturbed),
                no_passthrough: false,
            }),
        };
        let mut original = Probe::new(usize::MAX);
        kind.emit(&mut ContentWriter::new(&mut original)).unwrap();
        let mut changed = Probe::new(usize::MAX);
        Computation {
            lanes: kind.lanes.map(|lanes| LaneControl {
                no_passthrough: true,
                ..lanes
            }),
            ..kind
        }
        .emit(&mut ContentWriter::new(&mut changed))
        .unwrap();
        assert_eq!(original.data.last(), Some(&0));
        assert_eq!(changed.data.last(), Some(&1));
        assert_eq!(original.data.len(), changed.data.len());
        assert_eq!(
            original.data[..original.data.len() - 1],
            changed.data[..changed.data.len() - 1]
        );
        for limit in 0..changed.events {
            let mut refused = Probe::new(limit);
            assert!(kind.emit(&mut ContentWriter::new(&mut refused)).is_err());
            assert_eq!(refused.after_refusal, 0);
            assert!(original.data.starts_with(&refused.data));
        }
    }

    /// Freeing an x87 register and freeing then popping have distinct fixed tags.
    #[test]
    fn fpu_free_and_pop_content_have_distinct_tags() {
        for (kind, expected) in [
            (FpuControlKind::FreeRegister, [12, 0]),
            (FpuControlKind::FreeRegisterPop, [25, 0]),
        ] {
            let mut probe = Probe::new(usize::MAX);
            kind.emit(&mut ContentWriter::new(&mut probe)).unwrap();
            assert_eq!(probe.data, expected);
        }
    }

    /// Single-block and eight-block key-locker steps retain distinct stable identities.
    #[test]
    fn key_locker_content_matches_independent_tags_and_refuses_cleanly() {
        for (kind, expected) in [
            (VectorCryptoKind::AesEncryptKeyLocker, [27, 0]),
            (VectorCryptoKind::AesDecryptKeyLocker, [28, 0]),
            (VectorCryptoKind::AesEncryptKeyLockerWide, [63, 0]),
            (VectorCryptoKind::AesDecryptKeyLockerWide, [64, 0]),
        ] {
            let mut complete = Probe::new(usize::MAX);
            assert_eq!(kind.emit(&mut ContentWriter::new(&mut complete)), Ok(()));
            assert_eq!(complete.data, expected);
            for limit in 0..complete.events {
                let mut refused = Probe::new(limit);
                assert!(kind.emit(&mut ContentWriter::new(&mut refused)).is_err());
                assert!(expected.starts_with(&refused.data));
                assert_eq!(refused.after_refusal, 0);
            }
        }
    }

    /// Ordered-comparison floating selects have separate stable tags.
    #[test]
    fn vector_select_content_matches_independent_tags() {
        for (kind, expected) in [
            (VectorBinaryKind::MinLess, [40, 0]),
            (VectorBinaryKind::MaxGreater, [41, 0]),
        ] {
            let mut probe = Probe::new(usize::MAX);
            kind.emit(&mut ContentWriter::new(&mut probe)).unwrap();
            assert_eq!(probe.data, expected);
        }
    }

    /// A return through a named target and a call's keyed outputs have tags of
    /// their own and keep every operand.
    #[test]
    fn return_targets_and_call_register_keys_are_content() {
        let returned = SsaOp::<MockTarget>::Return {
            value: Some(SsaVarId::from_index(11)),
            state: Box::default(),
            results: Default::default(),
            transfer: Some(ReturnTransfer::through(SsaVarId::from_index(7))),
        };
        let expected = [
            OPERATION_CONTENT_VERSION.to_le_bytes().as_slice(),
            214u16.to_le_bytes().as_slice(),
            &[1],
            11u64.to_le_bytes().as_slice(),
            0u64.to_le_bytes().as_slice(),
            7u64.to_le_bytes().as_slice(),
        ]
        .concat();
        assert_operation_content(&returned, &expected, &[]);
        let clobber = SsaOp::<MockTarget>::CallClobber {
            outputs: vec![SsaVarId::from_index(3)],
            machine: Some(
                vec![CallOutput {
                    key: CallInputKey::new(17),
                    index: 0,
                }]
                .into_boxed_slice(),
            ),
        };
        let expected = [
            OPERATION_CONTENT_VERSION.to_le_bytes().as_slice(),
            213u16.to_le_bytes().as_slice(),
            1u64.to_le_bytes().as_slice(),
            3u64.to_le_bytes().as_slice(),
            1u64.to_le_bytes().as_slice(),
            17u32.to_le_bytes().as_slice(),
            0u32.to_le_bytes().as_slice(),
        ]
        .concat();
        assert_operation_content(&clobber, &expected, &[]);
    }

    /// The byte oracle pins protocol version, operation tag, 64-bit indices and optional flags.
    #[test]
    fn add_content_matches_independent_little_endian_oracle() {
        let op = SsaOp::<MockTarget>::Add {
            dest: SsaVarId::from_index(7),
            left: SsaVarId::from_index(11),
            right: SsaVarId::from_index(13),
            flags: Some(SsaVarId::from_index(17)),
        };
        let mut probe = Probe::new(usize::MAX);
        assert_eq!(op.write_content(&mut probe), Ok(()));
        let expected = [
            OPERATION_CONTENT_VERSION.to_le_bytes().as_slice(),
            1u16.to_le_bytes().as_slice(),
            7u64.to_le_bytes().as_slice(),
            11u64.to_le_bytes().as_slice(),
            13u64.to_le_bytes().as_slice(),
            &[1],
            17u64.to_le_bytes().as_slice(),
        ]
        .concat();
        assert_eq!(probe.data, expected);
    }

    /// Standalone values share the pinned schema and preserve every refusal prefix.
    #[test]
    fn constant_content_matches_independent_bytes_and_stops_at_first_refusal() {
        let value = ConstValue::<MockTarget>::U32(0x01020304);
        let mut complete = Probe::new(usize::MAX);
        assert_eq!(value.write_content(&mut complete), Ok(()));
        assert_eq!(complete.data, [2, 0, 6, 0, 4, 3, 2, 1]);
        for limit in 0..complete.events {
            let mut refused = Probe::new(limit);
            assert!(value.write_content(&mut refused).is_err());
            assert_eq!(refused.after_refusal, 0);
            assert!(complete.data.starts_with(&refused.data));
            assert_eq!(refused.after_refusal, 0);
        }
    }

    /// Signed literal encodings retain their complete two's-complement pattern.
    #[test]
    fn integer_content_matches_independent_literal_oracle() {
        let op = SsaOp::<MockTarget>::Const {
            dest: SsaVarId::from_index(3),
            value: ConstValue::I32(-7),
        };
        let mut probe = Probe::new(usize::MAX);
        assert_eq!(op.write_content(&mut probe), Ok(()));
        let expected = [
            OPERATION_CONTENT_VERSION.to_le_bytes().as_slice(),
            0u16.to_le_bytes().as_slice(),
            3u64.to_le_bytes().as_slice(),
            2u16.to_le_bytes().as_slice(),
            &[0xf9, 0xff, 0xff, 0xff],
        ]
        .concat();
        assert_eq!(probe.data, expected);
    }

    /// Phi content preserves both predecessor indices and their paired value identities.
    #[test]
    fn phi_content_matches_independent_pair_oracle() {
        let op = SsaOp::<MockTarget>::Phi {
            dest: SsaVarId::from_index(7),
            operands: vec![(3, SsaVarId::from_index(11)), (5, SsaVarId::from_index(13))],
        };
        let mut probe = Probe::new(usize::MAX);
        assert_eq!(op.write_content(&mut probe), Ok(()));
        let expected = [
            OPERATION_CONTENT_VERSION.to_le_bytes().as_slice(),
            203u16.to_le_bytes().as_slice(),
            7u64.to_le_bytes().as_slice(),
            2u64.to_le_bytes().as_slice(),
            3u64.to_le_bytes().as_slice(),
            11u64.to_le_bytes().as_slice(),
            5u64.to_le_bytes().as_slice(),
            13u64.to_le_bytes().as_slice(),
        ]
        .concat();
        assert_eq!(probe.data, expected);
    }

    /// Arrays stream their exact source bytes and stop at the first refusal.
    #[test]
    fn array_content_streams_exact_bytes_without_finishing_after_refusal() {
        let op = SsaOp::<MockTarget>::Const {
            dest: SsaVarId::from_index(3),
            value: ConstValue::DecryptedArray(Box::new(DecryptedArrayData {
                data: vec![0x5a; 4096],
                element_type_ref: 9,
                element_size: 1,
            })),
        };
        let mut complete = Probe::new(usize::MAX);
        assert_eq!(op.write_content(&mut complete), Ok(()));
        // Version, opcode, destination, constant tag and array length occupy 22 bytes.
        assert_eq!(
            complete.data.get(22..4118),
            Some(vec![0x5a; 4096].as_slice())
        );
        assert_eq!(complete.domains, [1]);
        let mut partial = Probe::new(complete.events / 2);
        assert_eq!(op.write_content(&mut partial), Err(()));
        assert_eq!(partial.after_refusal, 0);
        assert!(complete.data.starts_with(&partial.data));
        assert!(partial.data.len() < complete.data.len());
        assert!(partial.domains.is_empty());
    }

    /// All opaque domains are delegated, and refusal prevents subsequent fields from being read.
    #[test]
    fn content_visits_each_borrowed_metadata_domain_and_propagates_refusal() {
        let id = SsaVarId::from_index(1);
        let ops = [
            SsaOp::<MockTarget>::IntConv {
                dest: id,
                operand: id,
                target: MockType::I32,
                overflow_check: false,
                unsigned: false,
            },
            SsaOp::Const {
                dest: id,
                value: ConstValue::Type(42),
            },
            SsaOp::Const {
                dest: id,
                value: ConstValue::MethodHandle(42),
            },
            SsaOp::Const {
                dest: id,
                value: ConstValue::FieldHandle(42),
            },
            SsaOp::Call(Box::new(crate::ir::CallData {
                dest: None,
                callee: crate::ir::Callee::Indirect {
                    fptr: id,
                    signature: 42,
                },
                args: vec![id; 1024],
                effects: None,
                tail: false,
                implicit_inputs: Default::default(),
                results: Default::default(),
                staged: Vec::new(),
            })),
            SsaOp::Const {
                dest: id,
                value: ConstValue::Symbol(42),
            },
        ];
        for (domain, op) in ops.iter().enumerate() {
            let mut probe = Probe::new(usize::MAX);
            assert_eq!(op.write_content(&mut probe), Ok(()));
            assert_eq!(probe.domains, [domain as u8]);
            let mut refused = Probe::new(usize::MAX);
            refused.metadata_supported = false;
            assert_eq!(op.write_content(&mut refused), Err(()));
            assert_eq!(refused.after_refusal, 0);
            assert!(refused.domains.is_empty());
            assert!(refused.data.len() < probe.data.len());
        }
    }

    /// A nested raise chain remains linear and uses no recursive Rust call stack.
    #[test]
    fn deeply_nested_raise_content_is_iterative_and_stops_on_refusal() {
        let mut op = SsaOp::<MockTarget>::returns(None);
        for _ in 0..40_000 {
            op = SsaOp::FloatRaise {
                dest: SsaVarId::from_index(0),
                op: Box::new(op),
                lanes: Some(SsaVarId::from_index(1)),
                raises: FloatExceptions::INEXACT,
                layout: FloatExceptionLayout::IEEE,
            };
        }
        let mut complete = Probe::new(usize::MAX);
        let result = op.write_content(&mut complete);
        let mut refused = Probe::new(1024);
        let refusal = op.write_content(&mut refused);
        // Teardown must remain iterative even when an assertion would fail.
        loop {
            op = match op {
                SsaOp::FloatRaise { op: inner, .. } => *inner,
                _ => break,
            };
        }
        assert_eq!(result, Ok(()));
        assert!(complete.events > 40_000);
        assert_eq!(refusal, Err(()));
        assert_eq!(refused.events, 1024);
        assert_eq!(refused.after_refusal, 0);
        assert!(complete.data.starts_with(&refused.data));
        assert_eq!(refused.after_refusal, 0);
    }

    /// New scalar signaling, exception layout and locality fields retain their identities.
    #[test]
    fn canonical_content_preserves_signaling_rounded_up_and_memory_locality() {
        let encode = |op: &SsaOp<MockTarget>| {
            let mut sink = Probe::new(usize::MAX);
            assert_eq!(op.write_content(&mut sink), Ok(()));
            sink.data
        };
        let equal = |signaling| SsaOp::Ceq {
            dest: SsaVarId::from_index(0),
            left: SsaVarId::from_index(1),
            right: SsaVarId::from_index(2),
            signaling,
        };
        assert_ne!(encode(&equal(false)), encode(&equal(true)));
        let raise = |layout| SsaOp::FloatRaise {
            dest: SsaVarId::from_index(3),
            op: Box::new(equal(false)),
            lanes: None,
            raises: FloatExceptions::INEXACT.union(FloatExceptions::ROUNDED_UP),
            layout,
        };
        assert_ne!(
            encode(&raise(FloatExceptionLayout::IEEE)),
            encode(&raise(FloatExceptionLayout::IEEE.rounded_up_at(9)))
        );
        let mut plain = Probe::new(usize::MAX);
        let mut hinted = Probe::new(usize::MAX);
        assert_eq!(
            AccessQualifiers::default().emit(&mut ContentWriter::new(&mut plain)),
            Ok(())
        );
        assert_eq!(
            AccessQualifiers::default()
                .with_locality(Locality::NonTemporal)
                .emit(&mut ContentWriter::new(&mut hinted)),
            Ok(())
        );
        assert_ne!(plain.data, hinted.data);
    }

    /// The independent prefix fixes the streamed-copy width and direction in the new operation schema.
    #[test]
    fn streamed_block_content_preserves_element_width_and_direction() {
        for width in [1u32, 2, 4, 8, 16] {
            for backward in [None, Some(SsaVarId::from_index(17))] {
                let op = SsaOp::<MockTarget>::CopyBlk {
                    dest_addr: SsaVarId::from_index(7),
                    src_addr: SsaVarId::from_index(11),
                    size: SsaVarId::from_index(13),
                    element_bytes: width,
                    backward,
                    src_access: crate::ir::ops::AccessQualifiers::FLAT,
                    dest_access: crate::ir::ops::AccessQualifiers::FLAT,
                };
                let flag = backward.map_or_else(Vec::new, |_| 17u64.to_le_bytes().to_vec());
                let prefix = [
                    OPERATION_CONTENT_VERSION.to_le_bytes().as_slice(),
                    130u16.to_le_bytes().as_slice(),
                    7u64.to_le_bytes().as_slice(),
                    11u64.to_le_bytes().as_slice(),
                    13u64.to_le_bytes().as_slice(),
                    width.to_le_bytes().as_slice(),
                    &[u8::from(backward.is_some())],
                    flag.as_slice(),
                ]
                .concat();
                let mut complete = Probe::new(usize::MAX);
                assert_eq!(op.write_content(&mut complete), Ok(()));
                assert!(complete.data.starts_with(&prefix));
                for limit in 0..complete.events {
                    let mut refused = Probe::new(limit);
                    assert!(op.write_content(&mut refused).is_err());
                    assert!(complete.data.starts_with(&refused.data));
                    assert_eq!(refused.after_refusal, 0);
                }
            }
        }
    }

    /// **What a call or a return hands back beside its value is content.** A
    /// call returning a second value, and a return handing one back, encode
    /// apart from the same operation returning one, and from each other by
    /// key and value.
    #[test]
    fn values_returned_beside_a_result_are_content() {
        let content = |op: &SsaOp<MockTarget>| {
            let mut probe = Probe::new(usize::MAX);
            assert_eq!(op.write_content(&mut probe), Ok(()));
            probe.data
        };
        let call = |results: &[(u32, usize)]| {
            let mut call = CallData::<MockTarget>::new(
                Some(SsaVarId::from_index(1)),
                Callee::Direct(9),
                vec![SsaVarId::from_index(3)],
            );
            for (key, value) in results {
                call = call.with_result(CallInputKey::new(*key), SsaVarId::from_index(*value));
            }
            SsaOp::from(call)
        };
        let returned = |results: &[(u32, usize)]| {
            SsaOp::<MockTarget>::returns_results(
                Some(SsaVarId::from_index(1)),
                results
                    .iter()
                    .map(|(key, value)| CallInput {
                        key: CallInputKey::new(*key),
                        value: SsaVarId::from_index(*value),
                    })
                    .collect(),
            )
        };
        for build in [call, returned] {
            let encodings = [
                build(&[]),
                build(&[(2, 4)]),
                build(&[(3, 4)]),
                build(&[(2, 5)]),
            ]
            .iter()
            .map(content)
            .collect::<Vec<_>>();
            for (index, encoding) in encodings.iter().enumerate() {
                assert!(
                    encodings
                        .iter()
                        .skip(index + 1)
                        .all(|other| other != encoding),
                    "{index}"
                );
            }
        }
        // A return handing back nothing beside its value is the plain return.
        assert_eq!(
            content(&returned(&[])),
            content(&SsaOp::returns(Some(SsaVarId::from_index(1))))
        );
    }

    /// Store status contracts survive canonical content and cannot share an identity.
    #[test]
    fn atomic_status_encodings_have_distinct_canonical_content() {
        let mut encodings = Vec::new();
        for encoding in [
            crate::ir::ops::AtomicStatus::Unspecified,
            crate::ir::ops::AtomicStatus::ZeroNonzero,
            crate::ir::ops::AtomicStatus::ZeroOne,
            crate::ir::ops::AtomicStatus::OneZero,
        ] {
            let op = SsaOp::<MockTarget>::AtomicStoreConditional {
                status: SsaVarId::from_index(0),
                status_encoding: encoding,
                consumes_reservation: false,
                addr: SsaVarId::from_index(1),
                value: SsaVarId::from_index(2),
                value_type: MockType::U32,
                success_ordering: crate::ir::ops::AtomicOrdering::Release,
                failure_ordering: crate::ir::ops::AtomicOrdering::Relaxed,
                width: crate::ir::ops::AtomicAccessWidth::Bits32,
                access: AccessQualifiers::default(),
            };
            let mut probe = Probe::new(usize::MAX);
            assert_eq!(op.write_content(&mut probe), Ok(()));
            let mut consuming = op.clone();
            if let SsaOp::AtomicStoreConditional {
                consumes_reservation,
                ..
            } = &mut consuming
            {
                *consumes_reservation = true;
            }
            let mut consuming_probe = Probe::new(usize::MAX);
            assert_eq!(consuming.write_content(&mut consuming_probe), Ok(()));
            assert_ne!(probe.data, consuming_probe.data);
            assert!(encodings.iter().all(|prior| prior != &probe.data));
            encodings.push(probe.data);
            #[cfg(feature = "serde")]
            {
                let value = serde_json::to_value(&op).expect("encode");
                let decoded: SsaOp<MockTarget> =
                    serde_json::from_value(value.clone()).expect("decode");
                assert_eq!(decoded, op);
                let mut legacy = value;
                legacy
                    .get_mut("AtomicStoreConditional")
                    .and_then(serde_json::Value::as_object_mut)
                    .expect("variant fields")
                    .remove("status_encoding");
                legacy
                    .get_mut("AtomicStoreConditional")
                    .and_then(serde_json::Value::as_object_mut)
                    .expect("variant fields")
                    .remove("consumes_reservation");
                let decoded: SsaOp<MockTarget> =
                    serde_json::from_value(legacy).expect("legacy contract");
                assert!(matches!(
                    decoded,
                    SsaOp::AtomicStoreConditional {
                        status_encoding: crate::ir::ops::AtomicStatus::Unspecified,
                        consumes_reservation: false,
                        ..
                    }
                ));
            }
        }
    }
    /// Returned processor state and allocation initialization retain exact wire fields.
    #[test]
    fn current_state_and_allocation_contracts_have_exact_content() {
        let v = SsaVarId::from_index;
        for state in [vec![], vec![v(1)], vec![v(1), v(2)]] {
            let operation = SsaOp::<MockTarget>::returns_with(None, state.clone());
            let mut expected = vec![9, 0, 83, 0, 0];
            expected.extend_from_slice(&(state.len() as u64).to_le_bytes());
            for value in state {
                expected.extend_from_slice(&(value.index() as u64).to_le_bytes());
            }
            let mut complete = Probe::new(usize::MAX);
            assert_eq!(operation.write_content(&mut complete), Ok(()));
            assert_eq!(complete.data, expected);
            for limit in 0..complete.events {
                let mut refused = Probe::new(limit);
                assert!(operation.write_content(&mut refused).is_err());
                assert!(expected.starts_with(&refused.data));
                assert_eq!(refused.after_refusal, 0);
            }
        }
        for zeroed in [false, true] {
            let operation = SsaOp::<MockTarget>::LocalAlloc {
                dest: v(3),
                size: v(4),
                zeroed,
            };
            let mut expected = vec![9, 0, 197, 0];
            expected.extend_from_slice(&3u64.to_le_bytes());
            expected.extend_from_slice(&4u64.to_le_bytes());
            expected.push(u8::from(zeroed));
            let mut complete = Probe::new(usize::MAX);
            assert_eq!(operation.write_content(&mut complete), Ok(()));
            assert_eq!(complete.data, expected);
        }
    }
}
