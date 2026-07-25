//! Shared IR generator for the fuzz targets.
//!
//! Builds a `SsaFunction<MockTarget>` from arbitrary bytes *without* going
//! through the checked builder, so the result is deliberately allowed to be
//! malformed: dangling branch targets, phi operands naming non-existent
//! predecessors, uses of variables nothing defines, out-of-range definition
//! sites. Those are exactly the shapes a hostile or misread binary produces, and
//! the shapes the verifier exists to report rather than crash on.

use analyssa::{
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MockTarget, MockType},
};
use arbitrary::{Arbitrary, Unstructured};

/// A compact, fuzzer-friendly description of a function.
#[derive(Debug, Arbitrary)]
pub struct FuzzFunction {
    pub num_args: u8,
    pub num_locals: u8,
    pub blocks: Vec<FuzzBlock>,
}

#[derive(Debug, Arbitrary)]
pub struct FuzzBlock {
    pub phis: Vec<FuzzPhi>,
    pub instructions: Vec<FuzzInstr>,
    pub terminator: FuzzTerminator,
}

#[derive(Debug, Arbitrary)]
pub struct FuzzPhi {
    pub result: u8,
    pub operands: Vec<(u8, u8)>,
}

#[derive(Debug, Arbitrary)]
pub enum FuzzInstr {
    Const { dest: u8, value: i32 },
    Add { dest: u8, left: u8, right: u8 },
    Sub { dest: u8, left: u8, right: u8 },
    Copy { dest: u8, src: u8 },
    Nop,
}

#[derive(Debug, Arbitrary)]
pub enum FuzzTerminator {
    Return { value: Option<u8> },
    Jump { target: u8 },
    Branch { condition: u8, t: u8, f: u8 },
}

fn var(index: u8) -> SsaVarId {
    SsaVarId::from_index(usize::from(index))
}

/// Builds a (possibly malformed) function from the description.
pub fn build(spec: &FuzzFunction) -> SsaFunction<MockTarget> {
    let block_count = spec.blocks.len().max(1);
    let mut ssa: SsaFunction<MockTarget> =
        SsaFunction::new(usize::from(spec.num_args), usize::from(spec.num_locals));

    // Register a bounded pool of variables so most ids resolve, while leaving
    // room for ids that do not — the interesting case.
    for idx in 0..64u16 {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
    }

    for (block_idx, block_spec) in spec.blocks.iter().enumerate() {
        let mut block = SsaBlock::new(block_idx);

        for phi_spec in &block_spec.phis {
            let mut phi = PhiNode::new(var(phi_spec.result), VariableOrigin::Phi);
            for &(value, pred) in &phi_spec.operands {
                phi.add_operand(PhiOperand::new(var(value), usize::from(pred)));
            }
            block.add_phi(phi);
        }

        for instr in &block_spec.instructions {
            let op = match *instr {
                FuzzInstr::Const { dest, value } => SsaOp::Const {
                    dest: var(dest),
                    value: ConstValue::I32(value),
                },
                FuzzInstr::Add { dest, left, right } => SsaOp::Add {
                    dest: var(dest),
                    left: var(left),
                    right: var(right),
                    flags: None,
                },
                FuzzInstr::Sub { dest, left, right } => SsaOp::Sub {
                    dest: var(dest),
                    left: var(left),
                    right: var(right),
                    flags: None,
                },
                FuzzInstr::Copy { dest, src } => SsaOp::Copy {
                    dest: var(dest),
                    src: var(src),
                },
                FuzzInstr::Nop => SsaOp::Nop,
            };
            block.add_instruction(SsaInstruction::synthetic(op));
        }

        let terminator = match block_spec.terminator {
            FuzzTerminator::Return { value } => SsaOp::Return {
                value: value.map(var),
            },
            FuzzTerminator::Jump { target } => SsaOp::Jump {
                target: usize::from(target),
            },
            FuzzTerminator::Branch { condition, t, f } => SsaOp::Branch {
                condition: var(condition),
                true_target: usize::from(t),
                false_target: usize::from(f),
            },
        };
        block.add_instruction(SsaInstruction::synthetic(terminator));
        ssa.add_block(block);
    }

    if block_count > 0 && ssa.block_count() == 0 {
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
        ssa.add_block(block);
    }

    ssa.recompute_uses();
    ssa
}

/// Generates a function from raw fuzzer bytes.
pub fn from_bytes(data: &[u8]) -> Option<SsaFunction<MockTarget>> {
    let mut u = Unstructured::new(data);
    let spec = FuzzFunction::arbitrary(&mut u).ok()?;
    // Keep inputs small enough that a campaign explores shapes rather than size.
    if spec.blocks.len() > 32 {
        return None;
    }
    Some(build(&spec))
}
