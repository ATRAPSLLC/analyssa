//! Exact byte-decoder fixtures shared by execution and snapshot benchmarks.

use std::{error::Error, io, ops::Range};

use analyssa::{
    analysis::{
        execution::{ExecutionHost, RealignedExtent},
        image::{ImageData, ImageRegion, Mutability},
    },
    ir::{
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{CmpKind, SsaOp},
        value::ConstValue,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    target::Endianness,
    testing::{MockTarget, MockType},
};

/// An immutable source whose byte at index `i` is `i mod 256`.
pub struct Image(
    /// The immutable source contents, indexed from module address `0x1000`.
    pub Vec<u8>,
);

impl ImageData for Image {
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }
    fn number_offset(&self) -> Option<i64> {
        Some(0)
    }
    fn region(&self, address: u64) -> Option<ImageRegion> {
        let end = 0x1000u64.saturating_add(self.0.len() as u64);
        (0x1000..end).contains(&address).then_some(ImageRegion {
            start: 0x1000,
            end,
            mutability: Mutability::ReadOnly,
        })
    }
    fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
        let start = usize::try_from(address.checked_sub(0x1000)?).ok()?;
        self.0.get(start..start.checked_add(len)?)
    }
}

/// How far below the stack the decoder's output begins: room for every
/// byte it writes, `stride` apart, and 64 bytes of gap above them.
fn output_depth(count: usize, stride: i64) -> i64 {
    (count as i64).saturating_mul(stride).saturating_add(64)
}

/// The host of a decoder: it attests the frame the decoder writes its output
/// into, as a compiled function's host attests the frame its prologue
/// reserved. The analysis proves no stack write without one.
pub struct DecoderFrame {
    /// The bytes the frame spans below the entry stack, or below the base
    /// the aligned decoder realigns it to.
    depth: i64,
}

impl DecoderFrame {
    /// The frame of [`decoder`] over `count` bytes written `write_stride`
    /// apart: from the output's start up to the (realigned) stack.
    pub fn new(count: usize, write_stride: Option<(u32, bool)>) -> Self {
        let stride = i64::from(write_stride.map_or(1, |(stride, _)| stride));
        Self {
            depth: output_depth(count, stride),
        }
    }
}

impl ExecutionHost<MockTarget> for DecoderFrame {
    fn stack_extent(
        &self,
        _body: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<Range<i64>> {
        Some(self.depth.saturating_neg()..0)
    }

    fn realigned_stack_extent(
        &self,
        _body: &SsaFunction<MockTarget>,
        _method: Option<&u32>,
    ) -> Option<RealignedExtent> {
        Some(RealignedExtent {
            at: 0,
            range: self.depth.saturating_neg()..0,
        })
    }
}

/// Builds byte decoding followed by optional independent-bit branches and a join.
pub fn decoder(
    count: usize,
    aligned: bool,
    fork_levels: u32,
    write_stride: Option<(u32, bool)>,
) -> Result<SsaFunction<MockTarget>, Box<dyn Error>> {
    if fork_levels > 8 {
        return Err(io::Error::other("benchmark fork depth exceeds eight").into());
    }
    let stride = i64::from(write_stride.map_or(1, |(stride, _)| stride));
    if stride == 0 {
        return Err(io::Error::other("zero benchmark stride").into());
    }
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    // An aligned decoder masks its stack pointer as lifted machine code does:
    // as an integer register, which is how a realignment's mask is read.
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        if aligned {
            MockType::U64
        } else {
            MockType::Ptr
        },
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .ok_or_else(|| io::Error::other("missing stack value"))?
        .set_role(VariableRole::EntryStackPointer);
    let selector = (fork_levels != 0).then(|| {
        builder.function_mut().create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I64,
        )
    });
    let (source, target, zero, one, limit, key) = builder.in_block(0, |b| {
        let stack = if aligned {
            let mask = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(-16))?;
            b.and(SsaDefSpec::tmp(MockType::U64), stack, mask)?
        } else {
            stack
        };
        let source = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
        let target = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 1,
            offset: output_depth(count, stride).saturating_neg(),
            result_type: MockType::Ptr,
        })?;
        let zero = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(0))?;
        let one = b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(1))?;
        let limit = b.const_value(
            SsaDefSpec::tmp(MockType::I64),
            ConstValue::I64(count as i64),
        )?;
        let key = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x5a))?;
        b.jump(1)?;
        Ok((source, target, zero, one, limit, key))
    })?;
    let index = builder.in_block(1, |b| b.empty_phi(SsaDefSpec::tmp(MockType::I64)))?;
    let next = builder.in_block(1, |b| {
        let from = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: source,
            index: Some(index),
            stride: 1,
            offset: 0,
            result_type: MockType::Ptr,
        })?;
        let byte = b.load_indirect(SsaDefSpec::tmp(MockType::U8), from, MockType::U8)?;
        let decoded = b.xor(SsaDefSpec::tmp(MockType::U8), byte, key)?;
        let to = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: target,
            index: Some(index),
            stride: stride.cast_unsigned(),
            offset: 0,
            result_type: MockType::Ptr,
        })?;
        b.store_indirect(to, decoded, MockType::U8)?;
        let next = b.add(SsaDefSpec::tmp(MockType::I64), index, one)?;
        b.branch_cmp(next, limit, CmpKind::Lt, false, 1, 2)?;
        Ok(next)
    })?;
    builder.add_phi_operand(1, index, 0, zero)?;
    builder.add_phi_operand(1, index, 1, next)?;
    if fork_levels == 0 {
        builder.in_block(2, |b| b.ret(None))?;
    } else {
        let selector = selector.ok_or_else(|| io::Error::other("missing branch input"))?;
        let first_leaf = 1usize
            .checked_shl(fork_levels)
            .ok_or_else(|| io::Error::other("fork depth overflow"))?;
        let last_node = first_leaf.saturating_mul(2).saturating_sub(1);
        let join = last_node.saturating_add(2);
        let exit = join.saturating_add(1);
        let mut leaves = Vec::with_capacity(first_leaf);
        for node in 1..=last_node {
            let block = node.saturating_add(1);
            if node < first_leaf {
                let mask = 1i64
                    .checked_shl(node.ilog2())
                    .ok_or_else(|| io::Error::other("selector mask overflow"))?;
                builder.in_block(block, |b| {
                    let mask =
                        b.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(mask))?;
                    let bit = b.and(SsaDefSpec::tmp(MockType::I64), selector, mask)?;
                    b.branch_cmp(
                        bit,
                        zero,
                        CmpKind::Eq,
                        false,
                        node.saturating_mul(2).saturating_add(1),
                        node.saturating_mul(2).saturating_add(2),
                    )
                })?;
            } else {
                let value = builder.in_block(block, |b| {
                    let value = b.const_value(
                        SsaDefSpec::tmp(MockType::U8),
                        ConstValue::U8((node & 1) as u8),
                    )?;
                    if write_stride.is_some() {
                        let marker = b.const_value(
                            SsaDefSpec::tmp(MockType::U8),
                            ConstValue::U8(if write_stride.is_some_and(|(_, disagree)| disagree) {
                                (node & 1) as u8
                            } else {
                                0xa5
                            }),
                        )?;
                        b.store_indirect(target, marker, MockType::U8)?;
                    }
                    b.jump(join)?;
                    Ok(value)
                })?;
                leaves.push((block, value));
            }
        }
        let merged = builder.in_block(join, |b| {
            let merged = b.empty_phi(SsaDefSpec::tmp(MockType::U8))?;
            b.jump(exit)?;
            Ok(merged)
        })?;
        for (from, value) in leaves {
            builder.add_phi_operand(join, merged, from, value)?;
        }
        builder.in_block(exit, |b| {
            if write_stride.is_some() {
                let byte = b.load_indirect(SsaDefSpec::tmp(MockType::U8), target, MockType::U8)?;
                b.ret(Some(byte))
            } else {
                b.ret(Some(merged))
            }
        })?;
    }
    Ok(builder.finish()?)
}
