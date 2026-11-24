//! Copy-on-write scaling for dense and sparse decoded buffers.
//!
//! Independent unknown input bits select a branching tail with a shared phi
//! and return. Every path sees the same decoded buffer. This exercises actual
//! frame/memory snapshots and joins while retaining an independent byte oracle.

mod common;

use std::{env, error::Error, hint::black_box, io, time::Instant};

use analyssa::{
    analysis::{
        context::AnalysisContext,
        execution::{StaticExecution, Value},
    },
    ir::value::ConstValue,
    testing::MACHINE,
};

use crate::common::{DecoderFrame, Image, decoder};

/// Checks exact bytes and nonzero snapshots, then times complete executions.
fn main() -> Result<(), Box<dyn Error>> {
    let disagree = env::args().any(|argument| argument == "--disagree");
    let selected = env::args()
        .find_map(|argument| argument.strip_prefix("--stride=").map(str::to_owned))
        .map(|value| value.parse::<u32>())
        .transpose()?;
    if selected.is_some_and(|stride| !matches!(stride, 1 | 4096)) {
        return Err(io::Error::other("benchmark stride must be 1 or 4096").into());
    }
    println!("aligned,bytes,stride,disagree,median_ns,charged_steps,state_snapshots");
    for aligned in [false, true] {
        for stride in [1, 4096] {
            if selected.is_some_and(|selected| selected != stride) {
                continue;
            }
            for count in [64usize, 512, 2048, 8192] {
                let ssa = decoder(count, aligned, 4, Some((stride, disagree)))?;
                let frame = DecoderFrame::new(count, Some((stride, disagree)));
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
                let expected_return = if disagree {
                    Value::Unknown
                } else {
                    Value::Const(ConstValue::U8(0xa5))
                };
                if facts.returned() != Some(&expected_return) {
                    return Err(io::Error::other(
                        "post-fork memory write did not survive the join",
                    )
                    .into());
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
                    "{aligned},{count},{stride},{disagree},{median},{},{}",
                    facts.usage().steps,
                    facts.usage().state_snapshots
                );
            }
        }
    }
    Ok(())
}
