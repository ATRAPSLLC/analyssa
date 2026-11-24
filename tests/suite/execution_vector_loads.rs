//! Lane-wise vector loads: each family reads its active lanes from known
//! bytes, as an ordinary load reads them, and a load whose every read is
//! proven is replaced by its constants with the read it made.

use std::process;

use analyssa::{
    Error, Result,
    analysis::{
        context::AnalysisContext,
        execution::{ReplacementProof, StaticExecution, StaticFacts},
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        function::{SsaBlockBuilder, SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        lanes::VectorConst,
        ops::{
            AccessQualifiers, SsaOp, VectorFaultMode, VectorMaskMode, VectorPackKind,
            VectorSegmentLayout, VectorStructLoadReplicateData,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    passes::residual::fold_constants,
    target::Endianness,
    testing::{FactfulTarget, MACHINE, MockType},
};

/// Where the encoded bytes lie.
const AT: u64 = 0x1000;

/// Sixteen read-only bytes: `0x10, 0x11, ..., 0x1f`.
const ENCODED: [u8; 16] = [
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

/// [`ENCODED`] read-only at [`AT`], and nothing else.
struct Encoded;

impl ImageData for Encoded {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    fn region(&self, address: u64) -> Option<ImageRegion> {
        let end = AT.checked_add(ENCODED.len() as u64)?;
        (AT..end).contains(&address).then_some(ImageRegion {
            start: AT,
            end,
            mutability: Mutability::ReadOnly,
        })
    }

    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(address.checked_sub(AT)?).ok()?;
        ENCODED.get(start..start.checked_add(len)?)
    }

    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
}

/// Ends the test: a fixture the builder rejects is a broken test.
fn abort<E, R>(_: E) -> R {
    process::abort()
}

/// The little-endian element of `bytes` bytes at offset `at` of [`ENCODED`].
fn element(at: usize, bytes: usize) -> u128 {
    ENCODED
        .get(at..at.saturating_add(bytes))
        .unwrap_or_default()
        .iter()
        .rev()
        .fold(0, |word, byte| (word << 8) | u128::from(*byte))
}

/// A vector of `bits`-bit lanes.
fn lanes(bits: u32, values: &[u128]) -> ConstValue<FactfulTarget> {
    VectorConst::from_lanes(bits, values.iter().copied())
        .map(ConstValue::Vector)
        .unwrap_or_else(|| abort(()))
}

/// A load built by `load` from the address `from`, run over [`Encoded`].
struct Loaded {
    ssa: SsaFunction<FactfulTarget>,
    facts: StaticFacts<FactfulTarget>,
    results: Vec<SsaVarId>,
}

impl Loaded {
    /// Builds and runs the load, whose results `load` returns.
    fn of(
        from: u64,
        load: impl FnOnce(&mut SsaBlockBuilder<'_, FactfulTarget>, SsaVarId) -> Result<Vec<SsaVarId>>,
    ) -> Result<Self> {
        let mut builder = SsaFunctionBuilder::<FactfulTarget>::new(0, 0);
        let mut results = Vec::new();
        builder.in_block(0, |block| {
            let source =
                block.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::U64(from))?;
            results = load(block, source)?;
            block.ret(None)
        })?;
        let ssa = builder.finish()?;
        let context = AnalysisContext::new(MACHINE).with_image(&Encoded);
        let facts = StaticExecution::new(&ssa, &context).run();
        if !facts.is_complete() {
            return Err(Error::new(format!("refused: {:?}", facts.completion())));
        }
        Ok(Self {
            ssa,
            facts,
            results,
        })
    }

    /// What each result holds on every run.
    fn constants(&self) -> Vec<Option<ConstValue<FactfulTarget>>> {
        self.results
            .iter()
            .map(|result| self.facts.constant(*result).cloned())
            .collect()
    }

    /// Replaces what the run proves, returning how many operations went,
    /// whether every result is now its constant, and the image reads kept.
    fn folded(mut self) -> (usize, bool, Vec<(u64, u64)>) {
        let folded = fold_constants(&mut self.ssa, &self.facts).unwrap_or_else(abort);
        let constant = self
            .results
            .iter()
            .all(|result| self.ssa.get_var_constant(*result).is_some());
        (folded, constant, self.ssa.facts().image_reads.clone())
    }
}

/// A mask constant: one bit per lane.
fn mask(b: &mut SsaBlockBuilder<'_, FactfulTarget>, bits: u8) -> Result<SsaVarId> {
    b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(bits))
}

/// **A masked load reads its active lanes, and fills the rest as its mode
/// says**: zero, or the passthrough's lanes. Its reads are proven, so it is
/// replaced by its constant and keeps the extent it read.
#[test]
fn a_masked_load_reads_its_active_lanes() -> Result<()> {
    for (mode, inactive) in [(VectorMaskMode::Zero, 0), (VectorMaskMode::Merge, 9)] {
        let loaded = Loaded::of(AT, |b, addr| {
            let active = mask(b, 0b0101)?;
            let through = b.const_value(SsaDefSpec::tmp(MockType::V4I32), lanes(32, &[9; 4]))?;
            Ok(vec![b.emit_def(
                SsaDefSpec::tmp(MockType::V4I32),
                |dest| SsaOp::VectorMaskedLoad {
                    dest,
                    addr,
                    mask: active,
                    passthrough: Some(through),
                    vector_type: MockType::V4I32,
                    mode,
                    access: AccessQualifiers::FLAT,
                },
            )?])
        })?;
        assert_eq!(
            loaded.constants(),
            vec![Some(lanes(
                32,
                &[element(0, 4), inactive, element(8, 4), inactive]
            ))],
            "{mode:?}"
        );
        assert_eq!(
            loaded
                .facts
                .replacement_proof(loaded.results.first().copied().unwrap_or_else(|| abort(()))),
            Some(ReplacementProof::ReadOnlyImage {
                address: AT,
                len: 12
            })
        );
        assert_eq!(loaded.folded(), (1, true, vec![(AT, 12)]), "{mode:?}");
    }
    Ok(())
}

/// **A gather reads each active lane at its own offset.**
#[test]
fn a_gather_reads_each_lane_at_its_offset() -> Result<()> {
    let loaded = Loaded::of(AT, |b, base| {
        let active = mask(b, 0b1111)?;
        let indices = b.const_value(SsaDefSpec::tmp(MockType::V4I32), lanes(32, &[12, 0, 4, 8]))?;
        Ok(vec![b.emit_def(
            SsaDefSpec::tmp(MockType::V4I32),
            |dest| SsaOp::VectorGather {
                dest,
                base,
                indices,
                mask: active,
                passthrough: None,
                vector_type: MockType::V4I32,
                mode: VectorMaskMode::Zero,
                access: AccessQualifiers::FLAT,
            },
        )?])
    })?;
    assert_eq!(
        loaded.constants(),
        vec![Some(lanes(
            32,
            &[element(12, 4), element(0, 4), element(4, 4), element(8, 4)]
        ))]
    );
    assert_eq!(loaded.folded(), (1, true, vec![(AT, 16)]));
    Ok(())
}

/// **A first-faulting load reads every lane it can prove readable, and says
/// which.** Wholly in the image, it reads all four and its fault mask names
/// them; with lanes past the image, whether they fault is not known, so
/// neither its value nor its mask is, and it stays.
#[test]
fn a_first_faulting_load_names_the_lanes_it_read() -> Result<()> {
    let faulting = |from| {
        Loaded::of(from, |b, addr| {
            let [dest, fault] = [
                SsaDefSpec::tmp(MockType::V4I32),
                SsaDefSpec::tmp(MockType::U8),
            ];
            let defs = b.emit_many_defs(&[dest, fault], |ids| SsaOp::VectorFaultingLoad {
                dest: ids.first().copied().unwrap_or_else(|| abort(())),
                fault: ids.get(1).copied(),
                addr,
                mask: None,
                passthrough: None,
                vector_type: MockType::V4I32,
                fault_mode: VectorFaultMode::FirstFault,
                mask_mode: VectorMaskMode::Zero,
                access: AccessQualifiers::FLAT,
            })?;
            Ok(defs)
        })
    };
    let whole = faulting(AT)?;
    assert_eq!(
        whole.constants(),
        vec![
            Some(lanes(
                32,
                &[element(0, 4), element(4, 4), element(8, 4), element(12, 4)]
            )),
            Some(ConstValue::U8(0b1111)),
        ]
    );
    assert_eq!(whole.folded(), (1, true, vec![(AT, 16)]));
    let past = faulting(AT.saturating_add(8))?;
    assert_eq!(past.constants(), vec![None, None]);
    assert_eq!(past.folded(), (0, false, vec![]));
    Ok(())
}

/// **A segment load reads interleaved structures into one vector per
/// field.**
#[test]
fn a_segment_load_reads_one_vector_per_field() -> Result<()> {
    let loaded = Loaded::of(AT, |b, base| {
        b.emit_many_defs(
            &[
                SsaDefSpec::tmp(MockType::V4I16),
                SsaDefSpec::tmp(MockType::V4I16),
            ],
            |ids| SsaOp::VectorSegmentLoad {
                dests: ids.to_vec(),
                base,
                mask: None,
                vector_type: MockType::V4I16,
                segments: 2,
                layout: VectorSegmentLayout::Interleaved,
                access: AccessQualifiers::FLAT,
            },
        )
    })?;
    let field = |first: usize| {
        lanes(
            16,
            &[0, 1, 2, 3].map(|structure: usize| element(structure * 4 + first, 2)),
        )
    };
    assert_eq!(loaded.constants(), vec![Some(field(0)), Some(field(2))]);
    assert_eq!(loaded.folded(), (1, true, vec![(AT, 16)]));
    Ok(())
}

/// **An expanding load fills its active lanes with consecutive elements.**
#[test]
fn an_expanding_load_fills_active_lanes_in_order() -> Result<()> {
    let loaded = Loaded::of(AT, |b, addr| {
        let active = mask(b, 0b1010)?;
        Ok(vec![b.emit_def(
            SsaDefSpec::tmp(MockType::V4I32),
            |dest| SsaOp::VectorPackLoad {
                dest,
                addr,
                mask: active,
                passthrough: None,
                vector_type: MockType::V4I32,
                element_bits: 32,
                kind: VectorPackKind::Expand,
                mode: VectorMaskMode::Zero,
                access: AccessQualifiers::FLAT,
            },
        )?])
    })?;
    assert_eq!(
        loaded.constants(),
        vec![Some(lanes(32, &[0, element(0, 4), 0, element(4, 4)]))]
    );
    assert_eq!(loaded.folded(), (1, true, vec![(AT, 8)]));
    Ok(())
}

/// **A replicating structure load fills each output with one element.**
#[test]
fn a_replicating_structure_load_fills_each_output() -> Result<()> {
    let loaded = Loaded::of(AT, |b, base| {
        b.emit_many_defs(
            &[
                SsaDefSpec::tmp(MockType::V4I32),
                SsaDefSpec::tmp(MockType::V4I32),
            ],
            |ids| {
                SsaOp::VectorStructLoadReplicate(Box::new(VectorStructLoadReplicateData {
                    count: 2,
                    element_bits: 32,
                    outputs: ids.to_vec(),
                    inputs: vec![base],
                }))
            },
        )
    })?;
    assert_eq!(
        loaded.constants(),
        vec![
            Some(lanes(32, &[element(0, 4); 4])),
            Some(lanes(32, &[element(4, 4); 4])),
        ]
    );
    assert_eq!(loaded.folded(), (1, true, vec![(AT, 8)]));
    Ok(())
}
