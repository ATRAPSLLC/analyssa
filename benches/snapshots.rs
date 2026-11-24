//! Snapshot and join scaling after an exact byte decode.
//!
//! Independent unknown input bits select a branching tail with a shared phi
//! and return. Every path sees the same decoded buffer. This exercises actual
//! frame/memory snapshots and joins while retaining an independent byte oracle.

mod common;

use std::{error::Error, hint::black_box, io, time::Instant};

use analyssa::{
    analysis::{context::AnalysisContext, execution::StaticExecution},
    testing::MACHINE,
};

use crate::common::{DecoderFrame, Image, decoder};

/// Checks exact bytes and nonzero snapshots, then times complete executions.
fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "aligned,bytes,fork_levels,median_ns,charged_steps,state_snapshots,allocated_units,charged_bytes"
    );
    for aligned in [false, true] {
        for fork_levels in [1, 4] {
            for count in [64usize, 512, 2048, 8192] {
                let ssa = decoder(count, aligned, fork_levels, None)?;
                let frame = DecoderFrame::new(count, None);
                let image = Image((0..count).map(|index| index as u8).collect());
                let context = AnalysisContext::new(MACHINE).with_image(&image);
                let facts = StaticExecution::new(&ssa, &context).with_host(&frame).run();
                if !facts.is_complete() {
                    return Err(io::Error::other(format!(
                        "snapshot execution refused: {:?}",
                        facts.completion()
                    ))
                    .into());
                }
                if facts.usage().state_snapshots == 0 {
                    return Err(io::Error::other("workload did not create state snapshots").into());
                }
                let bytes: Vec<_> = facts
                    .loop_effect(1)
                    .ok_or_else(|| io::Error::other("missing decoder effect"))?
                    .writes
                    .iter()
                    .flat_map(|(_, bytes)| bytes.iter().copied())
                    .collect();
                let expected: Vec<_> = image.0.iter().map(|byte| byte ^ 0x5a).collect();
                if bytes != expected {
                    return Err(io::Error::other("snapshot output differs from XOR oracle").into());
                }
                let mut times = Vec::with_capacity(5);
                for _ in 0..5 {
                    let started = Instant::now();
                    black_box(
                        StaticExecution::new(black_box(&ssa), black_box(&context))
                            .with_host(&frame)
                            .run(),
                    );
                    times.push(started.elapsed().as_nanos());
                }
                times.sort_unstable();
                let median = times
                    .get(2)
                    .copied()
                    .ok_or_else(|| io::Error::other("missing median"))?;
                println!(
                    "{aligned},{count},{fork_levels},{median},{},{}",
                    facts.usage().steps,
                    facts.usage().state_snapshots
                );
            }
        }
    }
    Ok(())
}
