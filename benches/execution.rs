//! Scaling of exact byte decoding through the shared SSA executor.
//!
//! Every size runs the same load/XOR/store loop over independent image bytes.
//! The table reports the median of five complete executions, including fact
//! construction and cleanup. It checks exact output bytes before timing.

mod common;

use std::{error::Error, hint::black_box, io, time::Instant};

use analyssa::{
    analysis::{context::AnalysisContext, execution::StaticExecution},
    testing::MACHINE,
};

use crate::common::{DecoderFrame, Image, decoder};

/// Checks exact decoded bytes, then reports execution cost per byte.
fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "aligned,bytes,median_ns,ns_per_byte,charged_steps,state_snapshots,allocated_units,charged_bytes"
    );
    for aligned in [false, true] {
        for count in [64usize, 512, 2048, 8192] {
            let ssa = decoder(count, aligned, 0, None)?;
            let frame = DecoderFrame::new(count, None);
            let image = Image((0..count).map(|index| index as u8).collect());
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            let facts = StaticExecution::new(&ssa, &context).with_host(&frame).run();
            if !facts.is_complete() {
                return Err(io::Error::other(format!(
                    "execution refused: {:?}",
                    facts.completion()
                ))
                .into());
            }
            let bytes: Vec<u8> = facts
                .loop_effect(1)
                .ok_or_else(|| io::Error::other("missing decoder effect"))?
                .writes
                .iter()
                .flat_map(|(_, bytes)| bytes.iter().copied())
                .collect();
            let expected: Vec<_> = image.0.iter().map(|byte| byte ^ 0x5a).collect();
            if bytes != expected {
                return Err(
                    io::Error::other("decoded output differs from independent XOR oracle").into(),
                );
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
                "{aligned},{count},{median},{:.1},{},{}",
                median as f64 / count as f64,
                facts.usage().steps,
                facts.usage().state_snapshots
            );
        }
    }
    Ok(())
}
