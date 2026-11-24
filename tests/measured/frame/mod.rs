//! A function whose frame write the memory pass removes.
//!
//! The frame-exit analysis indexes every use in the function, so a function with
//! one removable frame write runs it however the rest of the function reads.

use analyssa::{
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{AccessQualifiers, SsaOp},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
    },
    testing::{MockTarget, MockType},
};

/// `*(sp - 8) = 7; _ = copy read; return`, where `sp` is the declared entry
/// stack pointer.
///
/// The store is dead, and `read` is named only by the copy.
pub fn frame_write_reading(read: SsaVarId) -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 8);
    let sp = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    if let Some(variable) = ssa.variable_mut(sp) {
        variable.set_role(VariableRole::EntryStackPointer);
    }
    let value = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let slot = ssa.create_variable(
        VariableOrigin::Local(2),
        0,
        DefSite::instruction(0, 1),
        MockType::Ptr,
    );
    let copied = ssa.create_variable(
        VariableOrigin::Local(3),
        0,
        DefSite::instruction(0, 3),
        MockType::I32,
    );
    let mut block = SsaBlock::new(0);
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(7),
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::PtrAdd {
        dest: slot,
        base: sp,
        index: None,
        stride: 0,
        offset: -8,
        result_type: MockType::Ptr,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::StoreIndirect {
        addr: slot,
        value,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
        dest: copied,
        src: read,
    }));
    block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa
}

/// The number of stores `ssa` holds.
pub fn stores(ssa: &SsaFunction<MockTarget>) -> usize {
    ssa.blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter(|instruction| matches!(instruction.op(), SsaOp::StoreIndirect { .. }))
        .count()
}
