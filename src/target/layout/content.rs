//! Canonical content for fixed-size semantic type layouts.

use crate::{
    encoder::{ContentWriter, Emit},
    ir::content::ContentSink,
    target::{
        ScalableVectorMaskShape, ScalableVectorShape, Target, TypeLayout, VectorDescriptor,
        VectorLaneKind, VectorLengthMultiplier, VectorMaskDescriptor, VectorMaskPolicy,
        VectorMaskShape, VectorShape,
    },
};

impl<T: Target> Emit<T> for TypeLayout {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            bits,
            integer,
            floating,
            boolean,
            boolean_storage,
            signed,
            pointer,
            reference,
            unknown,
            float,
            vector,
            mask,
            conversion,
        } = self;
        bits.emit(writer)?;
        integer.emit(writer)?;
        floating.emit(writer)?;
        boolean.emit(writer)?;
        boolean_storage
            .map(|storage| storage.bytes())
            .emit(writer)?;
        signed.emit(writer)?;
        pointer.emit(writer)?;
        reference.emit(writer)?;
        unknown.emit(writer)?;
        float.emit(writer)?;
        vector.emit(writer)?;
        mask.emit(writer)?;
        conversion.emit(writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            lane_count,
            lane_kind,
            lane_bits,
            total_bits,
        } = self;
        lane_count.emit(writer)?;
        lane_kind.emit(writer)?;
        lane_bits.emit(writer)?;
        total_bits.emit(writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorMaskShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            lane_count,
            lane_bits,
        } = self;
        lane_count.emit(writer)?;
        lane_bits.emit(writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorLengthMultiplier {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            numerator,
            denominator,
        } = self;
        numerator.emit(writer)?;
        denominator.emit(writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for ScalableVectorShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            min_lane_count,
            lane_kind,
            lane_bits,
            length_multiplier,
            tail_policy,
            mask_policy,
        } = self;
        min_lane_count.emit(writer)?;
        lane_kind.emit(writer)?;
        lane_bits.emit(writer)?;
        length_multiplier.emit(writer)?;
        tail_policy.emit(writer)?;
        mask_policy.emit(writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for ScalableVectorMaskShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            min_lane_count,
            lane_bits,
            length_multiplier,
        } = self;
        min_lane_count.emit(writer)?;
        lane_bits.emit(writer)?;
        length_multiplier.emit(writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorLaneKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Integer => 0u8.emit(writer),
            Self::Float(format) => {
                1u8.emit(writer)?;
                format.emit(writer)
            }
            Self::NativeInteger => 2u8.emit(writer),
        }
    }
}

impl<T: Target> Emit<T> for VectorMaskPolicy {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Agnostic => 0u8.emit(writer),
            Self::Undisturbed => 1u8.emit(writer),
        }
    }
}

impl<T: Target> Emit<T> for VectorDescriptor {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Fixed(shape) => {
                0u8.emit(writer)?;
                shape.emit(writer)
            }
            Self::Scalable(shape) => {
                1u8.emit(writer)?;
                shape.emit(writer)
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorMaskDescriptor {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Fixed(shape) => {
                0u8.emit(writer)?;
                shape.emit(writer)
            }
            Self::Scalable(shape) => {
                1u8.emit(writer)?;
                shape.emit(writer)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::{content::ContentSink, float::FloatFormat, integer::IntegerType},
        metadata::MetadataRef,
        target::{
            BooleanStorage, ScalableVectorMaskShape, ScalableVectorShape, TypeLayout,
            VectorDescriptor, VectorLaneKind, VectorLengthMultiplier, VectorMaskDescriptor,
            VectorMaskPolicy, VectorMaskShape, VectorShape, VectorTailPolicy,
        },
        testing::MockTarget,
    };

    /// Records bounded sink events and detects any visitation after refusal.
    struct Recording {
        /// Canonical bytes accepted before refusal.
        bytes: Vec<u8>,
        /// Number of accepted byte events.
        events: usize,
        /// Maximum events accepted by this sink.
        limit: usize,
        /// Whether the sink has refused its first event.
        refused: bool,
        /// Invalid callbacks following the first refusal.
        after_refusal: usize,
    }

    impl Recording {
        /// Creates a sink that accepts exactly `limit` events.
        fn new(limit: usize) -> Self {
            Self {
                bytes: Vec::new(),
                events: 0,
                limit,
                refused: false,
                after_refusal: 0,
            }
        }

        /// Accepts an event before inspecting its payload.
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

    impl ContentSink<MockTarget> for Recording {
        type Error = ();

        fn bytes(&mut self, bytes: &[u8]) -> Result<(), ()> {
            self.accept()?;
            assert!(bytes.len() <= 128);
            self.bytes.extend_from_slice(bytes);
            Ok(())
        }

        fn metadata(&mut self, _: MetadataRef<'_, MockTarget>) -> Result<(), ()> {
            panic!("a flat type layout must not query host metadata")
        }
    }

    /// Serializes one layout without using host metadata.
    fn encoded(layout: TypeLayout) -> Vec<u8> {
        let mut sink = Recording::new(usize::MAX);
        assert_eq!(layout.write_content(&mut sink), Ok(()));
        sink.bytes
    }

    /// Primitive wire bytes are independent of native endianness and standard hashing.
    #[test]
    fn layout_content_has_explicit_primitive_encoding() {
        let layout = TypeLayout::integer(IntegerType::new(32, false).expect("u32"));
        assert_eq!(
            encoded(layout),
            [
                1, 0, 1, 32, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 32, 0, 0, 0, 0, 0
            ]
        );
    }

    /// Every flat semantic field and the native constant carrier affects identity.
    #[test]
    fn layout_content_distinguishes_every_field() {
        let base = TypeLayout::integer(IntegerType::new(32, false).expect("u32"));
        for change in 0..15 {
            let mut layout = base;
            match change {
                0 => layout.bits = None,
                1 => layout.integer = false,
                2 => layout.floating = true,
                3 => layout.boolean = true,
                4 => layout.boolean_storage = Some(BooleanStorage::BYTE),
                5 => layout.signed = true,
                6 => layout.pointer = true,
                7 => layout.reference = true,
                8 => layout.unknown = true,
                9 => layout.float = Some(FloatFormat::Half),
                10 => {
                    layout.vector = Some(VectorDescriptor::Fixed(VectorShape {
                        lane_count: 1,
                        lane_kind: VectorLaneKind::Integer,
                        lane_bits: 32,
                        total_bits: 32,
                    }))
                }
                11 => {
                    layout.mask = Some(VectorMaskDescriptor::Fixed(VectorMaskShape {
                        lane_count: 4,
                        lane_bits: 8,
                    }))
                }
                12 => layout.conversion = None,
                13 => layout.conversion = Some(IntegerType::new(32, false).expect("u32").native()),
                14 => layout.conversion = IntegerType::new(32, true),
                _ => unreachable!(),
            }
            assert_ne!(encoded(base), encoded(layout), "field {change}");
        }
        let mut half = base;
        half.float = Some(FloatFormat::Half);
        let mut brain = half;
        brain.float = Some(FloatFormat::BFloat16);
        assert_ne!(encoded(half), encoded(brain));
    }

    /// Vector shape, mask extent and inactive-lane policy keep their full fields.
    #[test]
    fn layout_content_distinguishes_geometry_and_policy() {
        let mut base = TypeLayout::integer(IntegerType::new(32, false).expect("u32"));
        let vector = ScalableVectorShape {
            min_lane_count: 4,
            lane_kind: VectorLaneKind::Integer,
            lane_bits: 32,
            length_multiplier: VectorLengthMultiplier {
                numerator: 1,
                denominator: 1,
            },
            tail_policy: VectorTailPolicy::Agnostic,
            mask_policy: VectorMaskPolicy::Agnostic,
        };
        base.vector = Some(VectorDescriptor::Scalable(vector));
        for change in 0..8 {
            let mut shape = vector;
            match change {
                0 => shape.min_lane_count += 1,
                1 => shape.lane_kind = VectorLaneKind::Float(FloatFormat::Single),
                2 => shape.lane_kind = VectorLaneKind::NativeInteger,
                3 => shape.lane_bits += 1,
                4 => shape.length_multiplier.numerator += 1,
                5 => shape.length_multiplier.denominator += 1,
                6 => shape.tail_policy = VectorTailPolicy::Undisturbed,
                7 => shape.mask_policy = VectorMaskPolicy::Undisturbed,
                _ => unreachable!(),
            }
            let mut changed = base;
            changed.vector = Some(VectorDescriptor::Scalable(shape));
            assert_ne!(encoded(base), encoded(changed), "scalable field {change}");
        }
        let fixed = VectorShape {
            lane_count: 4,
            lane_kind: VectorLaneKind::Integer,
            lane_bits: 32,
            total_bits: 128,
        };
        base.vector = Some(VectorDescriptor::Fixed(fixed));
        for change in 0..4 {
            let mut shape = fixed;
            match change {
                0 => shape.lane_count += 1,
                1 => shape.lane_kind = VectorLaneKind::NativeInteger,
                2 => shape.lane_bits += 1,
                3 => shape.total_bits += 1,
                _ => unreachable!(),
            }
            let mut changed = base;
            changed.vector = Some(VectorDescriptor::Fixed(shape));
            assert_ne!(encoded(base), encoded(changed), "fixed field {change}");
        }
        let mask = ScalableVectorMaskShape {
            min_lane_count: 4,
            lane_bits: 1,
            length_multiplier: vector.length_multiplier,
        };
        base.mask = Some(VectorMaskDescriptor::Scalable(mask));
        for change in 0..5 {
            let mut shape = mask;
            let mut changed = base;
            match change {
                0 => shape.min_lane_count += 1,
                1 => shape.lane_bits += 1,
                2 => shape.length_multiplier.numerator += 1,
                3 => shape.length_multiplier.denominator += 1,
                4 => {
                    changed.mask = Some(VectorMaskDescriptor::Fixed(VectorMaskShape {
                        lane_count: 4,
                        lane_bits: 1,
                    }))
                }
                _ => unreachable!(),
            }
            if change != 4 {
                changed.mask = Some(VectorMaskDescriptor::Scalable(shape));
            }
            assert_ne!(encoded(base), encoded(changed), "mask field {change}");
        }
        let fixed_mask = VectorMaskShape {
            lane_count: 4,
            lane_bits: 1,
        };
        base.mask = Some(VectorMaskDescriptor::Fixed(fixed_mask));
        for shape in [
            VectorMaskShape {
                lane_count: 5,
                ..fixed_mask
            },
            VectorMaskShape {
                lane_bits: 2,
                ..fixed_mask
            },
        ] {
            let mut changed = base;
            changed.mask = Some(VectorMaskDescriptor::Fixed(shape));
            assert_ne!(encoded(base), encoded(changed));
        }
    }

    /// Every refusal point stops the stream before any later field or callback.
    #[test]
    fn layout_content_stops_at_each_refusal() {
        let mut layout = TypeLayout::integer(IntegerType::new(64, true).expect("i64").native());
        layout.boolean_storage = Some(BooleanStorage::BYTE);
        layout.float = Some(FloatFormat::Double);
        layout.vector = Some(VectorDescriptor::Scalable(ScalableVectorShape {
            min_lane_count: 4,
            lane_kind: VectorLaneKind::Float(FloatFormat::Single),
            lane_bits: 32,
            length_multiplier: VectorLengthMultiplier {
                numerator: 2,
                denominator: 1,
            },
            tail_policy: VectorTailPolicy::Undisturbed,
            mask_policy: VectorMaskPolicy::Undisturbed,
        }));
        layout.mask = Some(VectorMaskDescriptor::Scalable(ScalableVectorMaskShape {
            min_lane_count: 4,
            lane_bits: 1,
            length_multiplier: VectorLengthMultiplier {
                numerator: 1,
                denominator: 2,
            },
        }));
        let mut full = Recording::new(usize::MAX);
        assert_eq!(layout.write_content(&mut full), Ok(()));
        for limit in 0..full.events {
            let mut partial = Recording::new(limit);
            assert_eq!(layout.write_content(&mut partial), Err(()), "event {limit}");
            assert_eq!(partial.events, limit);
            assert_eq!(partial.after_refusal, 0);
            assert!(full.bytes.starts_with(&partial.bytes));
        }
    }
}
