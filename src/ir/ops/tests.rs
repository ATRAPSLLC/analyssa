//! Unit tests for the [`SsaOp`] surface.

use std::collections::HashMap;

use super::*;
use crate::{
    ir::{value::ConstValue, variable::SsaVarId},
    target::VectorShuffleMask,
    testing::{MockTarget, MockType},
};

/// The coarse token deliberately merges variants a similarity fingerprint
/// should treat alike, folds signedness, and returns a bare `"call"` for a
/// call — leaving classification to a target-binding consumer.
#[test]
fn coarse_token_coarsens_merges_and_folds() {
    let v = SsaVarId::from_index(0);

    // `Add` and its overflow-checked sibling collapse to one token, while
    // `opcode_name` keeps them distinct — the coarsening is the point.
    let add: SsaOp<MockTarget> = SsaOp::Add {
        dest: v,
        left: v,
        right: v,
        flags: None,
    };
    let add_ovf: SsaOp<MockTarget> = SsaOp::AddOvf {
        dest: v,
        left: v,
        right: v,
        unsigned: false,
        flags: None,
    };
    assert_eq!(add.coarse_token(), "add");
    assert_eq!(add_ovf.coarse_token(), "add");
    assert_ne!(
        add.opcode_name(),
        add_ovf.opcode_name(),
        "opcode_name must stay fine-grained even though coarse_token merges"
    );

    // Shift folds signedness into the token.
    let shr = |unsigned| SsaOp::<MockTarget>::Shr {
        dest: v,
        value: v,
        amount: v,
        unsigned,
        flags: None,
    };
    assert_eq!(shr(true).coarse_token(), "shru");
    assert_eq!(shr(false).coarse_token(), "shrs");

    // A call is a bare token here; the callee is classified downstream.
    let call: SsaOp<MockTarget> = SsaOp::Call {
        dest: Some(v),
        method: 0,
        args: vec![],
    };
    assert_eq!(call.coarse_token(), "call");

    // Two genuinely different operations must not collide on a token.
    let sub: SsaOp<MockTarget> = SsaOp::Sub {
        dest: v,
        left: v,
        right: v,
        flags: None,
    };
    assert_ne!(add.coarse_token(), sub.coarse_token());
}

/// Pure ops are hoistable; impure ones (calls) are not.
#[test]
fn is_pure_classifies_calls_and_arith() {
    let add: SsaOp<MockTarget> = SsaOp::Add {
        dest: SsaVarId::from_index(0),
        left: SsaVarId::from_index(1),
        right: SsaVarId::from_index(2),
        flags: None,
    };
    assert!(add.is_pure());

    let const_op: SsaOp<MockTarget> = SsaOp::Const {
        dest: SsaVarId::from_index(3),
        value: ConstValue::I32(42),
    };
    assert!(const_op.is_pure());

    let call: SsaOp<MockTarget> = SsaOp::Call {
        dest: Some(SsaVarId::from_index(4)),
        method: 0xAB,
        args: vec![],
    };
    assert!(!call.is_pure());
}

/// `Add` reports both operands; `Const` reports none.
#[test]
fn uses_lists_operands() {
    let v1 = SsaVarId::from_index(0);
    let v2 = SsaVarId::from_index(1);
    let dest = SsaVarId::from_index(2);

    let op: SsaOp<MockTarget> = SsaOp::Add {
        dest,
        left: v1,
        right: v2,
        flags: None,
    };
    let uses = op.uses();
    assert_eq!(uses.len(), 2);
    assert!(uses.contains(&v1));
    assert!(uses.contains(&v2));

    let const_op: SsaOp<MockTarget> = SsaOp::Const {
        dest,
        value: ConstValue::I32(42),
    };
    assert!(const_op.uses().is_empty());
}

/// New rotate ops report dest and correct uses.
#[test]
fn rotate_ops_dest_and_uses() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);
    let a = SsaVarId::from_index(2);

    let rol: SsaOp<MockTarget> = SsaOp::Rol {
        dest: d,
        value: v,
        amount: a,
    };
    assert_eq!(rol.dest(), Some(d));
    let uses = rol.uses();
    assert_eq!(uses.len(), 2);
    assert!(uses.contains(&v));
    assert!(uses.contains(&a));

    let ror: SsaOp<MockTarget> = SsaOp::Ror {
        dest: d,
        value: v,
        amount: a,
    };
    assert_eq!(ror.dest(), Some(d));
    assert!(ror.uses().contains(&v));
}

/// New bit-manip unary ops report dest and correct uses.
#[test]
fn bit_manip_ops_dest_and_uses() {
    let d = SsaVarId::from_index(0);
    let s = SsaVarId::from_index(1);

    let bswap: SsaOp<MockTarget> = SsaOp::BSwap { dest: d, src: s };
    assert_eq!(bswap.dest(), Some(d));
    assert_eq!(bswap.uses(), vec![s]);

    let brev: SsaOp<MockTarget> = SsaOp::BRev { dest: d, src: s };
    assert_eq!(brev.dest(), Some(d));
    assert_eq!(brev.uses(), vec![s]);

    let bsf: SsaOp<MockTarget> = SsaOp::BitScanForward { dest: d, src: s };
    assert_eq!(bsf.dest(), Some(d));
    assert_eq!(bsf.uses(), vec![s]);

    let bsr: SsaOp<MockTarget> = SsaOp::BitScanReverse { dest: d, src: s };
    assert_eq!(bsr.dest(), Some(d));
    assert_eq!(bsr.uses(), vec![s]);

    let popcnt: SsaOp<MockTarget> = SsaOp::Popcount { dest: d, src: s };
    assert_eq!(popcnt.dest(), Some(d));
    assert_eq!(popcnt.uses(), vec![s]);

    let parity: SsaOp<MockTarget> = SsaOp::Parity { dest: d, src: s };
    assert_eq!(parity.dest(), Some(d));
    assert_eq!(parity.uses(), vec![s]);
}

/// Select reports dest and three uses.
#[test]
fn select_dest_and_uses() {
    let d = SsaVarId::from_index(0);
    let c = SsaVarId::from_index(1);
    let t = SsaVarId::from_index(2);
    let f = SsaVarId::from_index(3);

    let op: SsaOp<MockTarget> = SsaOp::Select {
        dest: d,
        condition: c,
        true_val: t,
        false_val: f,
    };
    assert_eq!(op.dest(), Some(d));
    assert_eq!(op.uses().len(), 3);
    assert!(op.uses().contains(&c));
    assert!(op.uses().contains(&t));
    assert!(op.uses().contains(&f));
}

/// Atomic ops report dest and correct uses.
#[test]
fn atomic_ops_dest_and_uses() {
    let d = SsaVarId::from_index(0);
    let a = SsaVarId::from_index(1);
    let e = SsaVarId::from_index(2);
    let v = SsaVarId::from_index(3);

    let op: SsaOp<MockTarget> = SsaOp::CmpXchg {
        dest: d,
        addr: a,
        expected: e,
        desired: v,
    };
    assert_eq!(op.dest(), Some(d));
    assert_eq!(op.uses().len(), 3);

    let op2: SsaOp<MockTarget> = SsaOp::AtomicRmw {
        dest: d,
        addr: a,
        value: v,
        op: AtomicRmwOp::Xchg,
    };
    assert_eq!(op2.dest(), Some(d));
    assert_eq!(op2.uses().len(), 2);
}

#[test]
fn native_atomic_ops_report_defs_uses_and_effects() {
    let old = SsaVarId::from_index(0);
    let success = SsaVarId::from_index(1);
    let addr = SsaVarId::from_index(2);
    let expected = SsaVarId::from_index(3);
    let desired = SsaVarId::from_index(4);

    let cmpxchg: SsaOp<MockTarget> = SsaOp::AtomicCmpXchg {
        old,
        success: Some(success),
        addr,
        expected,
        desired,
        success_ordering: AtomicOrdering::SeqCst,
        failure_ordering: AtomicOrdering::Acquire,
        width: AtomicAccessWidth::Bits32,
        weak: false,
        volatile: false,
    };

    assert_eq!(cmpxchg.dest(), Some(old));
    assert_eq!(cmpxchg.defs().collect::<Vec<_>>(), vec![old, success]);
    assert_eq!(cmpxchg.uses(), vec![addr, expected, desired]);
    assert_eq!(cmpxchg.stack_effect(), (3, 2));
    let cmpxchg_effects = cmpxchg.effects();
    assert_eq!(cmpxchg_effects.kind, SsaEffectKind::Atomic);
    assert_eq!(
        cmpxchg_effects.memory_semantics,
        MemoryAccessSemantics::Atomic
    );
    assert_eq!(cmpxchg_effects.ordering, Some(AtomicOrdering::SeqCst));
    assert_eq!(cmpxchg_effects.trap, TrapClass::MemoryFault);
    assert!(!cmpxchg.is_pure());

    let xchg: SsaOp<MockTarget> = SsaOp::AtomicExchange {
        dest: old,
        addr,
        value: desired,
        ordering: AtomicOrdering::AcqRel,
        width: AtomicAccessWidth::Bits32,
        volatile: true,
    };
    assert_eq!(xchg.defs().collect::<Vec<_>>(), vec![old]);
    assert_eq!(xchg.uses(), vec![addr, desired]);
    assert_eq!(xchg.stack_effect(), (2, 1));
    assert_eq!(
        xchg.effects().memory_semantics,
        MemoryAccessSemantics::Atomic
    );
    assert!(xchg.effects().volatile);
    assert_eq!(
        format!("{xchg}"),
        "v0 = atomicxchg.volatile.acqrel.i32 v2, v4"
    );
}

#[test]
fn boolean_ops_are_pure_and_remappable() {
    let dest = SsaVarId::from_index(0);
    let left = SsaVarId::from_index(1);
    let right = SsaVarId::from_index(2);
    let replacement = SsaVarId::from_index(9);
    let op: SsaOp<MockTarget> = SsaOp::BoolAnd { dest, left, right };

    assert_eq!(op.dest(), Some(dest));
    assert_eq!(op.uses(), vec![left, right]);
    assert_eq!(op.stack_effect(), (2, 1));
    assert!(op.is_pure());
    assert_eq!(format!("{op}"), "v0 = bool.and v1, v2");

    let remapped = op.remap_variables(|var| (var == right).then_some(replacement));
    assert_eq!(remapped.uses(), vec![left, replacement]);

    let not: SsaOp<MockTarget> = SsaOp::BoolNot { dest, value: left };
    assert_eq!(not.uses(), vec![left]);
    assert_eq!(not.stack_effect(), (1, 1));
    assert_eq!(format!("{not}"), "v0 = bool.not v1");
}

#[test]
fn wide_arithmetic_ops_report_secondary_defs() {
    let low = SsaVarId::from_index(0);
    let high = SsaVarId::from_index(1);
    let left = SsaVarId::from_index(2);
    let right = SsaVarId::from_index(3);
    let mul: SsaOp<MockTarget> = SsaOp::WideMul {
        low,
        high,
        left,
        right,
        unsigned: true,
    };

    assert_eq!(mul.dest(), Some(low));
    assert_eq!(mul.defs().collect::<Vec<_>>(), vec![low, high]);
    assert_eq!(mul.uses(), vec![left, right]);
    assert_eq!(mul.stack_effect(), (2, 2));
    assert_eq!(format!("{mul}"), "v0, v1 = widemul.un v2, v3");

    let quotient = SsaVarId::from_index(4);
    let remainder = SsaVarId::from_index(5);
    let divisor = SsaVarId::from_index(6);
    let div: SsaOp<MockTarget> = SsaOp::WideDiv {
        quotient,
        remainder,
        high,
        low,
        divisor,
        unsigned: false,
    };

    assert_eq!(div.defs().collect::<Vec<_>>(), vec![quotient, remainder]);
    assert_eq!(div.uses(), vec![high, low, divisor]);
    assert_eq!(div.stack_effect(), (3, 2));
    assert!(div.may_throw());
    assert_eq!(format!("{div}"), "v4, v5 = widediv v1:v0, v6");
}

#[test]
fn expanded_vector_ops_report_uses_effects_and_stack_shape() {
    let dest = SsaVarId::from_index(0);
    let addr = SsaVarId::from_index(1);
    let mask = SsaVarId::from_index(2);
    let passthrough = SsaVarId::from_index(3);
    let indices = SsaVarId::from_index(4);

    let masked_load: SsaOp<MockTarget> = SsaOp::VectorMaskedLoad {
        dest,
        addr,
        mask,
        passthrough: Some(passthrough),
        vector_type: MockType::V4I32,
        mode: VectorMaskMode::Merge,
    };
    assert_eq!(masked_load.uses(), vec![addr, mask, passthrough]);
    assert_eq!(masked_load.stack_effect(), (3, 1));
    assert_eq!(masked_load.effects().kind, SsaEffectKind::Read);

    let scatter: SsaOp<MockTarget> = SsaOp::VectorScatter {
        base: addr,
        indices,
        value: dest,
        mask,
        vector_type: MockType::V4I32,
    };
    assert_eq!(scatter.uses(), vec![addr, indices, dest, mask]);
    assert_eq!(scatter.stack_effect(), (4, 0));
    assert_eq!(scatter.effects().kind, SsaEffectKind::Write);

    let fault = SsaVarId::from_index(5);
    let faulting_load: SsaOp<MockTarget> = SsaOp::VectorFaultingLoad {
        dest,
        fault: Some(fault),
        addr,
        mask: Some(mask),
        passthrough: Some(passthrough),
        vector_type: MockType::V4I32,
        fault_mode: VectorFaultMode::FaultOnlyFirst,
        mask_mode: VectorMaskMode::Merge,
    };
    assert_eq!(faulting_load.defs().collect::<Vec<_>>(), vec![dest, fault]);
    assert_eq!(faulting_load.uses(), vec![addr, mask, passthrough]);
    assert_eq!(faulting_load.stack_effect(), (3, 1));
    assert_eq!(faulting_load.effects().kind, SsaEffectKind::Read);

    let second_dest = SsaVarId::from_index(6);
    let segment_load: SsaOp<MockTarget> = SsaOp::VectorSegmentLoad {
        dests: vec![dest, second_dest],
        base: addr,
        mask: Some(mask),
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Interleaved,
    };
    assert_eq!(segment_load.dest(), Some(dest));
    assert_eq!(
        segment_load.defs().collect::<Vec<_>>(),
        vec![dest, second_dest]
    );
    assert_eq!(segment_load.uses(), vec![addr, mask]);
    assert_eq!(segment_load.stack_effect(), (2, 2));
    assert_eq!(segment_load.effects().kind, SsaEffectKind::Read);

    let segment_store: SsaOp<MockTarget> = SsaOp::VectorSegmentStore {
        base: addr,
        values: vec![dest, second_dest],
        mask: Some(mask),
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Interleaved,
    };
    assert_eq!(segment_store.dest(), None);
    assert_eq!(segment_store.uses(), vec![addr, dest, second_dest, mask]);
    assert_eq!(segment_store.stack_effect(), (4, 0));
    assert_eq!(segment_store.effects().kind, SsaEffectKind::Write);

    let bitmask: SsaOp<MockTarget> = SsaOp::VectorBitmask {
        dest,
        value: passthrough,
        kind: VectorBitmaskKind::LaneMostSignificantBits,
    };
    assert_eq!(bitmask.uses(), vec![passthrough]);
    assert_eq!(bitmask.stack_effect(), (1, 1));
    assert!(bitmask.is_pure());
}

/// Fence and InterruptReturn have no dest and no uses.
#[test]
fn fence_and_iret_no_dest_no_uses() {
    let fence: SsaOp<MockTarget> = SsaOp::Fence {
        kind: FenceKind::Full,
    };
    assert_eq!(fence.dest(), None);
    assert!(fence.uses().is_empty());

    let iret: SsaOp<MockTarget> = SsaOp::InterruptReturn;
    assert_eq!(iret.dest(), None);
    assert!(iret.uses().is_empty());
}

/// New pure ops are classified as pure.
#[test]
fn new_pure_ops_classification() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);
    let a = SsaVarId::from_index(2);

    assert!(
        SsaOp::<MockTarget>::Rol {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure()
    );
    assert!(
        SsaOp::<MockTarget>::Ror {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure()
    );
    // `Rcl`/`Rcr` rotate *through carry*: the carry flag is an implicit
    // input and output with no SSA operand, so they are NOT pure and must
    // never be value-numbered or eliminated as if they were.
    assert!(
        !SsaOp::<MockTarget>::Rcl {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure()
    );
    assert!(
        !SsaOp::<MockTarget>::Rcr {
            dest: d,
            value: v,
            amount: a
        }
        .is_pure()
    );
    assert!(SsaOp::<MockTarget>::BSwap { dest: d, src: v }.is_pure());
    assert!(SsaOp::<MockTarget>::BRev { dest: d, src: v }.is_pure());
    assert!(SsaOp::<MockTarget>::BitScanForward { dest: d, src: v }.is_pure());
    assert!(SsaOp::<MockTarget>::BitScanReverse { dest: d, src: v }.is_pure());
    assert!(SsaOp::<MockTarget>::Popcount { dest: d, src: v }.is_pure());
    assert!(SsaOp::<MockTarget>::Parity { dest: d, src: v }.is_pure());
    assert!(
        SsaOp::<MockTarget>::Select {
            dest: d,
            condition: v,
            true_val: a,
            false_val: d,
        }
        .is_pure()
    );
}

/// Side-effecting new ops are not pure.
#[test]
fn new_impure_ops_classification() {
    assert!(
        !SsaOp::<MockTarget>::Fence {
            kind: FenceKind::Full,
        }
        .is_pure()
    );
    assert!(!SsaOp::<MockTarget>::InterruptReturn.is_pure());
    assert!(
        !SsaOp::<MockTarget>::CmpXchg {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            expected: SsaVarId::from_index(2),
            desired: SsaVarId::from_index(3),
        }
        .is_pure()
    );
    assert!(
        !SsaOp::<MockTarget>::AtomicRmw {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            value: SsaVarId::from_index(2),
            op: AtomicRmwOp::Add,
        }
        .is_pure()
    );
}

/// InterruptReturn is a terminator.
#[test]
fn interrupt_return_is_terminator() {
    assert!(SsaOp::<MockTarget>::InterruptReturn.is_terminator());
}

/// CmpXchg and AtomicRmw may throw.
#[test]
fn atomic_ops_may_throw() {
    assert!(
        SsaOp::<MockTarget>::CmpXchg {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            expected: SsaVarId::from_index(2),
            desired: SsaVarId::from_index(3),
        }
        .may_throw()
    );
    assert!(
        SsaOp::<MockTarget>::AtomicRmw {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            value: SsaVarId::from_index(2),
            op: AtomicRmwOp::Xchg,
        }
        .may_throw()
    );
}

/// New pure ops do not throw.
#[test]
fn new_pure_ops_no_throw() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);
    assert!(
        !SsaOp::<MockTarget>::Rol {
            dest: d,
            value: v,
            amount: v
        }
        .may_throw()
    );
    assert!(!SsaOp::<MockTarget>::BSwap { dest: d, src: v }.may_throw());
    assert!(
        !SsaOp::<MockTarget>::Select {
            dest: d,
            condition: v,
            true_val: v,
            false_val: v,
        }
        .may_throw()
    );
}

/// as_binary_op returns info for rotate ops.
#[test]
fn as_binary_op_rotations() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);
    let a = SsaVarId::from_index(2);

    let rol = SsaOp::<MockTarget>::Rol {
        dest: d,
        value: v,
        amount: a,
    };
    let info = rol.as_binary_op().unwrap();
    assert_eq!(info.kind, BinaryOpKind::Rol);
    assert_eq!(info.dest, d);
    assert_eq!(info.left, v);
    assert_eq!(info.right, a);

    let ror = SsaOp::<MockTarget>::Ror {
        dest: d,
        value: v,
        amount: a,
    };
    let info = ror.as_binary_op().unwrap();
    assert_eq!(info.kind, BinaryOpKind::Ror);
}

/// as_unary_op returns info for bit-manip ops.
#[test]
fn as_unary_op_bit_manip() {
    let d = SsaVarId::from_index(0);
    let s = SsaVarId::from_index(1);

    let bswap = SsaOp::<MockTarget>::BSwap { dest: d, src: s };
    let info = bswap.as_unary_op().unwrap();
    assert_eq!(info.kind, UnaryOpKind::BSwap);
    assert_eq!(info.dest, d);
    assert_eq!(info.operand, s);

    let brev = SsaOp::<MockTarget>::BRev { dest: d, src: s };
    assert_eq!(brev.as_unary_op().unwrap().kind, UnaryOpKind::BRev);

    let bsf = SsaOp::<MockTarget>::BitScanForward { dest: d, src: s };
    assert_eq!(bsf.as_unary_op().unwrap().kind, UnaryOpKind::BitScanForward);

    let bsr = SsaOp::<MockTarget>::BitScanReverse { dest: d, src: s };
    assert_eq!(bsr.as_unary_op().unwrap().kind, UnaryOpKind::BitScanReverse);

    let popcnt = SsaOp::<MockTarget>::Popcount { dest: d, src: s };
    assert_eq!(popcnt.as_unary_op().unwrap().kind, UnaryOpKind::Popcount);

    let parity = SsaOp::<MockTarget>::Parity { dest: d, src: s };
    assert_eq!(parity.as_unary_op().unwrap().kind, UnaryOpKind::Parity);
}

/// stack_effect for new ops.
#[test]
fn stack_effect_new_ops() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);
    let a = SsaVarId::from_index(2);

    // Rotates: pop 2, push 1
    assert_eq!(
        SsaOp::<MockTarget>::Rol {
            dest: d,
            value: v,
            amount: a
        }
        .stack_effect(),
        (2, 1)
    );
    assert_eq!(
        SsaOp::<MockTarget>::Ror {
            dest: d,
            value: v,
            amount: a
        }
        .stack_effect(),
        (2, 1)
    );

    // Bit-manip: pop 1, push 1
    assert_eq!(
        SsaOp::<MockTarget>::BSwap { dest: d, src: v }.stack_effect(),
        (1, 1)
    );
    assert_eq!(
        SsaOp::<MockTarget>::Popcount { dest: d, src: v }.stack_effect(),
        (1, 1)
    );

    // Select: pop 3, push 1
    assert_eq!(
        SsaOp::<MockTarget>::Select {
            dest: d,
            condition: v,
            true_val: a,
            false_val: d,
        }
        .stack_effect(),
        (3, 1)
    );

    // CmpXchg: pop 3, push 1
    assert_eq!(
        SsaOp::<MockTarget>::CmpXchg {
            dest: d,
            addr: v,
            expected: a,
            desired: d,
        }
        .stack_effect(),
        (3, 1)
    );

    // AtomicRmw: pop 2, push 1
    assert_eq!(
        SsaOp::<MockTarget>::AtomicRmw {
            dest: d,
            addr: v,
            value: a,
            op: AtomicRmwOp::Add,
        }
        .stack_effect(),
        (2, 1)
    );

    // Fence: pop 0, push 0
    assert_eq!(
        SsaOp::<MockTarget>::Fence {
            kind: FenceKind::SeqCst
        }
        .stack_effect(),
        (0, 0)
    );

    // InterruptReturn: pop 0, push 0
    assert_eq!(SsaOp::<MockTarget>::InterruptReturn.stack_effect(), (0, 0));
}

/// replace_uses works for new ops.
#[test]
fn replace_uses_new_ops() {
    let d = SsaVarId::from_index(0);
    let old = SsaVarId::from_index(1);
    let new = SsaVarId::from_index(99);
    let other = SsaVarId::from_index(2);

    // Test rotation
    let mut op: SsaOp<MockTarget> = SsaOp::Rol {
        dest: d,
        value: old,
        amount: other,
    };
    assert_eq!(op.replace_uses(old, new), 1);
    assert_eq!(op.uses(), vec![new, other]);

    // Test bit-manip
    let mut op2: SsaOp<MockTarget> = SsaOp::BSwap { dest: d, src: old };
    assert_eq!(op2.replace_uses(old, new), 1);
    assert_eq!(op2.uses(), vec![new]);

    // Test Select
    let mut op3: SsaOp<MockTarget> = SsaOp::Select {
        dest: d,
        condition: old,
        true_val: other,
        false_val: old,
    };
    assert_eq!(op3.replace_uses(old, new), 2);
    assert_eq!(op3.uses(), vec![new, other, new]);

    // Test CmpXchg
    let mut op4: SsaOp<MockTarget> = SsaOp::CmpXchg {
        dest: d,
        addr: old,
        expected: other,
        desired: old,
    };
    assert_eq!(op4.replace_uses(old, new), 2);

    // Test AtomicRmw
    let mut op5: SsaOp<MockTarget> = SsaOp::AtomicRmw {
        dest: d,
        addr: old,
        value: other,
        op: AtomicRmwOp::Xor,
    };
    assert_eq!(op5.replace_uses(old, new), 1);
}

/// set_dest works for new dest-bearing ops.
#[test]
fn set_dest_new_ops() {
    let d = SsaVarId::from_index(0);
    let new_d = SsaVarId::from_index(99);
    let v = SsaVarId::from_index(1);
    let a = SsaVarId::from_index(2);

    let mut rol: SsaOp<MockTarget> = SsaOp::Rol {
        dest: d,
        value: v,
        amount: a,
    };
    assert!(rol.set_dest(new_d));
    assert_eq!(rol.dest(), Some(new_d));

    let mut bswap: SsaOp<MockTarget> = SsaOp::BSwap { dest: d, src: v };
    assert!(bswap.set_dest(new_d));
    assert_eq!(bswap.dest(), Some(new_d));

    let mut select: SsaOp<MockTarget> = SsaOp::Select {
        dest: d,
        condition: v,
        true_val: a,
        false_val: d,
    };
    assert!(select.set_dest(new_d));
    assert_eq!(select.dest(), Some(new_d));
}

/// set_dest returns false for ops without dest.
#[test]
fn set_dest_fails_for_no_dest_ops() {
    assert!(
        !SsaOp::<MockTarget>::Fence {
            kind: FenceKind::Full
        }
        .set_dest(SsaVarId::from_index(0))
    );
    assert!(!SsaOp::<MockTarget>::InterruptReturn.set_dest(SsaVarId::from_index(0)));
}

/// remap_variables works for new ops.
#[test]
fn remap_variables_new_ops() {
    let d0 = SsaVarId::from_index(0);
    let d99 = SsaVarId::from_index(99);
    let v1 = SsaVarId::from_index(1);
    let v55 = SsaVarId::from_index(55);
    let a2 = SsaVarId::from_index(2);

    let mut map = HashMap::new();
    map.insert(d0, d99);
    map.insert(v1, v55);
    let remap = |v: SsaVarId| map.get(&v).copied();

    let rol = SsaOp::<MockTarget>::Rol {
        dest: d0,
        value: v1,
        amount: a2,
    };
    let remapped = rol.remap_variables(remap);
    assert_eq!(remapped.dest(), Some(d99));
    assert!(remapped.uses().contains(&v55));
    assert!(remapped.uses().contains(&a2));
}

/// FenceKind display.
#[test]
fn fence_kind_display() {
    assert_eq!(format!("{}", FenceKind::Full), "full");
    assert_eq!(format!("{}", FenceKind::Acquire), "acquire");
    assert_eq!(format!("{}", FenceKind::Release), "release");
    assert_eq!(format!("{}", FenceKind::AcqRel), "acqrel");
    assert_eq!(format!("{}", FenceKind::SeqCst), "seqcst");
}

/// AtomicRmwOp display.
#[test]
fn atomic_rmw_op_display() {
    assert_eq!(format!("{}", AtomicRmwOp::Xchg), "xchg");
    assert_eq!(format!("{}", AtomicRmwOp::Add), "add");
    assert_eq!(format!("{}", AtomicRmwOp::Sub), "sub");
    assert_eq!(format!("{}", AtomicRmwOp::And), "and");
    assert_eq!(format!("{}", AtomicRmwOp::Or), "or");
    assert_eq!(format!("{}", AtomicRmwOp::Xor), "xor");
    assert_eq!(format!("{}", AtomicRmwOp::Min), "min");
    assert_eq!(format!("{}", AtomicRmwOp::Max), "max");
}

// -----------------------------------------------------------------------
// FlagsMask tests
// -----------------------------------------------------------------------

#[test]
fn flags_mask_constants() {
    assert_ne!(FlagsMask::CARRY, FlagsMask::ZERO);
    assert_ne!(FlagsMask::CARRY, FlagsMask::OVERFLOW);
    assert_eq!(FlagsMask::CARRY.bits(), 1 << 0);
    assert_eq!(FlagsMask::ZERO.bits(), 1 << 3);
    assert_eq!(FlagsMask::OVERFLOW.bits(), 1 << 5);
    assert!(FlagsMask::from_bits(0).is_empty());
    assert!(!FlagsMask::CARRY.is_empty());
    assert!(FlagsMask::x86_status().contains(FlagsMask::ADJUST));
    assert!(FlagsMask::x86_status().contains(FlagsMask::OVERFLOW));
    assert_eq!(
        FlagsMask::from_flag_bit(NativeFlagBit::Carry),
        FlagsMask::CARRY
    );
    assert_eq!(FlagsMask::CARRY.union(FlagsMask::ZERO).bits(), 0b1001);
}

#[test]
fn flags_mask_display() {
    assert_eq!(format!("{}", FlagsMask::CARRY), "CF");
    assert_eq!(
        format!(
            "{}",
            FlagsMask::from_bits(FlagsMask::CARRY.bits() | FlagsMask::ZERO.bits())
        ),
        "CF,ZF"
    );
    assert_eq!(format!("{}", FlagsMask::from_bits(0)), "none");
}

// -----------------------------------------------------------------------
// FlagCondition tests
// -----------------------------------------------------------------------

#[test]
fn flag_condition_display() {
    assert_eq!(format!("{}", FlagCondition::Carry), "carry");
    assert_eq!(format!("{}", FlagCondition::NotCarry), "not_carry");
    assert_eq!(format!("{}", FlagCondition::Zero), "zero");
    assert_eq!(format!("{}", FlagCondition::NotZero), "not_zero");
    assert_eq!(format!("{}", FlagCondition::Overflow), "overflow");
    assert_eq!(format!("{}", FlagCondition::NotOverflow), "not_overflow");
    assert_eq!(format!("{}", FlagCondition::Negative), "negative");
    assert_eq!(format!("{}", FlagCondition::Positive), "positive");
    assert_eq!(format!("{}", FlagCondition::ParityEven), "parity_even");
    assert_eq!(format!("{}", FlagCondition::ParityOdd), "parity_odd");
}

#[test]
fn flag_condition_variants_are_distinct() {
    assert_ne!(FlagCondition::Carry, FlagCondition::Zero);
    assert_ne!(FlagCondition::Overflow, FlagCondition::NotOverflow);
    assert_ne!(FlagCondition::Negative, FlagCondition::Positive);
    assert_ne!(FlagCondition::ParityEven, FlagCondition::ParityOdd);
}

#[test]
fn flag_condition_required_flags() {
    assert_eq!(FlagCondition::Carry.required_flags(), FlagsMask::CARRY);
    assert_eq!(FlagCondition::NotZero.required_flags(), FlagsMask::ZERO);
    assert_eq!(
        FlagCondition::NotOverflow.required_flags(),
        FlagsMask::OVERFLOW
    );
    assert_eq!(FlagCondition::Positive.required_flags(), FlagsMask::SIGN);
    assert_eq!(FlagCondition::ParityOdd.required_flags(), FlagsMask::PARITY);
}

#[test]
fn flag_producer_semantics_classify_defined_and_undefined_flags() {
    assert!(
        FlagProducerSemantics::X86Arithmetic
            .defined_mask()
            .contains(FlagsMask::x86_status())
    );
    assert!(
        FlagProducerSemantics::X86Logical
            .defined_mask()
            .contains(FlagsMask::CARRY.union(FlagsMask::OVERFLOW))
    );
    assert!(
        !FlagProducerSemantics::X86Multiply
            .defined_mask()
            .contains(FlagsMask::ZERO)
    );
    assert!(
        FlagProducerSemantics::AArch64Arithmetic
            .defined_mask()
            .contains(FlagsMask::SIGN.union(FlagsMask::ZERO))
    );

    let logical_writes = FlagProducerSemantics::X86Logical.writes();
    assert!(logical_writes.contains(&FlagWrite::undefined(NativeFlagBit::Adjust)));
    assert!(logical_writes.contains(&FlagWrite::cleared(NativeFlagBit::Carry)));
}

// -----------------------------------------------------------------------
// flags_dest tests
// -----------------------------------------------------------------------

#[test]
fn flags_dest_returns_flags_on_flag_setting_ops() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);
    let flags_var = SsaVarId::from_index(99);

    let add: SsaOp<MockTarget> = SsaOp::Add {
        dest: d,
        left: v,
        right: v,
        flags: Some(flags_var),
    };
    assert_eq!(add.flags_dest(), Some(flags_var));

    let sub: SsaOp<MockTarget> = SsaOp::Sub {
        dest: d,
        left: v,
        right: v,
        flags: Some(flags_var),
    };
    assert_eq!(sub.flags_dest(), Some(flags_var));

    let _and: SsaOp<MockTarget> = SsaOp::And {
        dest: d,
        left: v,
        right: v,
        flags: Some(flags_var),
    };
}

#[test]
fn flags_dest_is_none_when_no_flags_set() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);

    let add: SsaOp<MockTarget> = SsaOp::Add {
        dest: d,
        left: v,
        right: v,
        flags: None,
    };
    assert_eq!(add.flags_dest(), None);

    let mul: SsaOp<MockTarget> = SsaOp::Mul {
        dest: d,
        left: v,
        right: v,
        flags: None,
    };
    assert_eq!(mul.flags_dest(), None);
}

#[test]
fn flags_dest_is_none_for_non_flag_ops() {
    let d = SsaVarId::from_index(0);
    let v = SsaVarId::from_index(1);

    let select: SsaOp<MockTarget> = SsaOp::Select {
        dest: d,
        condition: v,
        true_val: v,
        false_val: v,
    };
    assert_eq!(select.flags_dest(), None);

    let call: SsaOp<MockTarget> = SsaOp::Call {
        dest: Some(d),
        method: 0,
        args: vec![],
    };
    assert_eq!(call.flags_dest(), None);
}

// -----------------------------------------------------------------------
// ReadFlags tests
// -----------------------------------------------------------------------

#[test]
fn read_flags_dest_and_uses() {
    let d = SsaVarId::from_index(0);
    let flags_var = SsaVarId::from_index(1);

    let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
        dest: d,
        flags: flags_var,
        mask: FlagsMask::ZERO,
    };
    assert_eq!(op.dest(), Some(d));
    assert_eq!(op.uses(), vec![flags_var]);
    assert!(op.is_pure());
    assert!(!op.is_terminator());
    assert!(!op.may_throw());
}

#[test]
fn read_flags_stack_effect() {
    let d = SsaVarId::from_index(0);
    let f = SsaVarId::from_index(1);

    let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
        dest: d,
        flags: f,
        mask: FlagsMask::CARRY,
    };
    assert_eq!(op.stack_effect(), (1, 1));
}

#[test]
fn read_flags_replace_uses() {
    let d = SsaVarId::from_index(0);
    let old = SsaVarId::from_index(1);
    let new = SsaVarId::from_index(99);

    let mut op: SsaOp<MockTarget> = SsaOp::ReadFlags {
        dest: d,
        flags: old,
        mask: FlagsMask::SIGN,
    };
    assert_eq!(op.replace_uses(old, new), 1);
    assert_eq!(op.uses(), vec![new]);
}

#[test]
fn read_flags_remap_variables() {
    let d0 = SsaVarId::from_index(0);
    let d99 = SsaVarId::from_index(99);
    let f1 = SsaVarId::from_index(1);
    let f55 = SsaVarId::from_index(55);

    let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
        dest: d0,
        flags: f1,
        mask: FlagsMask::OVERFLOW,
    };
    let remapped = op.remap_variables(|v| {
        if v == d0 {
            Some(d99)
        } else if v == f1 {
            Some(f55)
        } else {
            None
        }
    });
    assert_eq!(remapped.dest(), Some(d99));
    assert_eq!(remapped.uses(), vec![f55]);
}

#[test]
fn read_flags_display() {
    let d = SsaVarId::from_index(0);
    let f = SsaVarId::from_index(1);
    let op: SsaOp<MockTarget> = SsaOp::ReadFlags {
        dest: d,
        flags: f,
        mask: FlagsMask::ZERO,
    };
    assert_eq!(format!("{op}"), "v0 = readflags v1, ZF");
}

// -----------------------------------------------------------------------
// BranchFlags tests
// -----------------------------------------------------------------------

#[test]
fn branch_flags_is_terminator_with_successors() {
    let f = SsaVarId::from_index(0);

    let op: SsaOp<MockTarget> = SsaOp::BranchFlags {
        flags: f,
        condition: FlagCondition::Zero,
        true_target: 1,
        false_target: 2,
    };
    assert!(op.is_terminator());
    assert!(!op.is_pure());
    assert!(!op.may_throw());
    assert_eq!(op.dest(), None);
    assert_eq!(op.uses(), vec![f]);
    assert_eq!(op.successors(), vec![1, 2]);
}

#[test]
fn branch_flags_stack_effect() {
    let f = SsaVarId::from_index(0);
    let op: SsaOp<MockTarget> = SsaOp::BranchFlags {
        flags: f,
        condition: FlagCondition::Carry,
        true_target: 1,
        false_target: 2,
    };
    assert_eq!(op.stack_effect(), (1, 0));
}

#[test]
fn branch_flags_redirect_target() {
    let f = SsaVarId::from_index(0);
    let mut op: SsaOp<MockTarget> = SsaOp::BranchFlags {
        flags: f,
        condition: FlagCondition::NotZero,
        true_target: 3,
        false_target: 5,
    };
    assert!(op.redirect_target(3, 7));
    assert_eq!(op.successors(), vec![7, 5]);
    assert!(!op.redirect_target(99, 42)); // no-op
}

#[test]
fn branch_flags_remap_targets() {
    let f = SsaVarId::from_index(0);
    let mut op: SsaOp<MockTarget> = SsaOp::BranchFlags {
        flags: f,
        condition: FlagCondition::Overflow,
        true_target: 2,
        false_target: 4,
    };
    op.remap_branch_targets(|t| {
        if t == 2 {
            Some(10)
        } else if t == 4 {
            Some(20)
        } else {
            None
        }
    });
    assert_eq!(op.successors(), vec![10, 20]);
}

#[test]
fn branch_flags_replace_uses() {
    let old = SsaVarId::from_index(1);
    let new = SsaVarId::from_index(99);

    let mut op: SsaOp<MockTarget> = SsaOp::BranchFlags {
        flags: old,
        condition: FlagCondition::Positive,
        true_target: 1,
        false_target: 2,
    };
    assert_eq!(op.replace_uses(old, new), 1);
    assert_eq!(op.uses(), vec![new]);
}

#[test]
fn branch_flags_display() {
    let f = SsaVarId::from_index(0);
    let op: SsaOp<MockTarget> = SsaOp::BranchFlags {
        flags: f,
        condition: FlagCondition::Carry,
        true_target: 3,
        false_target: 7,
    };
    // The flags operand is rendered: two branches on different flags variables
    // would otherwise print identically.
    assert_eq!(format!("{op}"), "branchflags v0 carry B3, B7");
}

// -----------------------------------------------------------------------
// Unreachable tests
// -----------------------------------------------------------------------

#[test]
fn unreachable_is_terminator_with_no_successors() {
    assert!(SsaOp::<MockTarget>::Unreachable.is_terminator());
    assert!(!SsaOp::<MockTarget>::Unreachable.is_pure());
    assert!(!SsaOp::<MockTarget>::Unreachable.may_throw());
    assert_eq!(SsaOp::<MockTarget>::Unreachable.dest(), None);
    assert!(SsaOp::<MockTarget>::Unreachable.uses().is_empty());
    assert!(SsaOp::<MockTarget>::Unreachable.successors().is_empty());
}

#[test]
fn unreachable_stack_effect() {
    assert_eq!(SsaOp::<MockTarget>::Unreachable.stack_effect(), (0, 0));
}

#[test]
fn unreachable_display() {
    assert_eq!(
        format!("{}", SsaOp::<MockTarget>::Unreachable),
        "unreachable"
    );
}

#[test]
fn unreachable_no_variable_remap() {
    let op = SsaOp::<MockTarget>::Unreachable;
    let remapped = op.remap_variables(|_| unreachable!());
    assert_eq!(remapped, SsaOp::Unreachable);
}

// -----------------------------------------------------------------------
// Flag-setting op with flags integrated tests
// -----------------------------------------------------------------------

#[test]
fn flag_setting_op_has_two_defs() {
    let d = SsaVarId::from_index(0);
    let f = SsaVarId::from_index(99);
    let v = SsaVarId::from_index(1);

    let op: SsaOp<MockTarget> = SsaOp::Add {
        dest: d,
        left: v,
        right: v,
        flags: Some(f),
    };
    assert_eq!(op.dest(), Some(d));
    assert_eq!(op.flags_dest(), Some(f));
    assert!(op.is_pure());
}

#[test]
fn flag_setting_op_remap_remaps_flags() {
    let d0 = SsaVarId::from_index(0);
    let d99 = SsaVarId::from_index(99);
    let f1 = SsaVarId::from_index(1);
    let f55 = SsaVarId::from_index(55);

    let op: SsaOp<MockTarget> = SsaOp::Add {
        dest: d0,
        left: d0,
        right: d0,
        flags: Some(f1),
    };
    let remapped = op.remap_variables(|v| {
        if v == d0 {
            Some(d99)
        } else if v == f1 {
            Some(f55)
        } else {
            None
        }
    });
    assert_eq!(remapped.dest(), Some(d99));
    assert_eq!(remapped.flags_dest(), Some(f55));
}

#[test]
fn flag_setting_op_display_shows_flags() {
    let d = SsaVarId::from_index(0);
    let f = SsaVarId::from_index(99);
    let v = SsaVarId::from_index(1);

    let with_flags: SsaOp<MockTarget> = SsaOp::Add {
        dest: d,
        left: v,
        right: v,
        flags: Some(f),
    };
    assert_eq!(format!("{with_flags}"), "v0 = add v1, v1 flags=v99");

    let without_flags: SsaOp<MockTarget> = SsaOp::Add {
        dest: d,
        left: v,
        right: v,
        flags: None,
    };
    assert_eq!(format!("{without_flags}"), "v0 = add v1, v1");
}

#[test]
fn effect_summaries_classify_pure_memory_atomic_and_call_ops() {
    let d = SsaVarId::from_index(0);
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);

    assert!(
        SsaOp::<MockTarget>::Add {
            dest: d,
            left: a,
            right: b,
            flags: None,
        }
        .effects()
        .is_pure()
    );

    let load = SsaOp::<MockTarget>::LoadIndirect {
        dest: d,
        addr: a,
        value_type: MockType::I32,
        address_space: None,
    }
    .effects();
    assert_eq!(load.kind, SsaEffectKind::Read);
    assert_eq!(load.trap, TrapClass::MemoryFault);
    assert!(load.reads_memory());
    assert!(!load.writes_memory());

    let store = SsaOp::<MockTarget>::StoreIndirect {
        addr: a,
        value: b,
        value_type: MockType::I32,
        address_space: None,
    }
    .effects();
    assert_eq!(store.kind, SsaEffectKind::Write);
    assert!(store.writes_memory());

    let atomic = SsaOp::<MockTarget>::AtomicRmw {
        dest: d,
        addr: a,
        value: b,
        op: AtomicRmwOp::Add,
    }
    .effects();
    assert_eq!(atomic.kind, SsaEffectKind::Atomic);
    assert_eq!(atomic.memory_semantics, MemoryAccessSemantics::Atomic);
    assert_eq!(atomic.ordering, Some(AtomicOrdering::SeqCst));
    assert!(atomic.reads_memory());
    assert!(atomic.writes_memory());
    assert!(!atomic.removable_when_unused());

    // AArch64 `ld2r`/`ld3r`/`ld4r` reads memory through its address base, so
    // it must classify exactly like its sibling vector loads. Treating it as
    // pure would let GVN CSE it across an intervening store and let DCE
    // delete it outright.
    let struct_load_replicate =
        SsaOp::<MockTarget>::VectorStructLoadReplicate(Box::new(VectorStructLoadReplicateData {
            count: 2,
            element_bits: 32,
            outputs: vec![d],
            inputs: vec![a],
        }))
        .effects();
    assert_eq!(struct_load_replicate.kind, SsaEffectKind::Read);
    assert_eq!(struct_load_replicate.trap, TrapClass::MemoryFault);
    assert!(struct_load_replicate.reads_memory());
    assert!(!struct_load_replicate.writes_memory());
    assert!(
        !struct_load_replicate.is_pure(),
        "ld2r/ld3r/ld4r is a memory load and must never be pure"
    );
    assert!(
        !struct_load_replicate.removable_when_unused(),
        "a faulting load must not be removable just because its dests are unused"
    );

    // `rdffr` *reads* the same first-fault register `setffr`/`wrffr` write, and
    // the FFR is not an SSA operand — so `rdffr` is not a function of its SSA
    // operands, and its `inputs` list is typically empty. Pure would let GVN
    // collapse every `rdffr` in a function onto the first (they all normalize to
    // one key) and let LICM hoist it out of the loop containing the
    // first-faulting load whose result it reports.
    let rdffr = SsaOp::<MockTarget>::VectorPredicateGen(Box::new(KindedVecData {
        kind: PredicateGenKind::ReadFfr,
        outputs: vec![d],
        inputs: vec![],
    }))
    .effects();
    assert!(
        !rdffr.is_pure(),
        "rdffr reads hidden machine state and must not be pure"
    );

    // The rest of the predicate-generation family really are functions of their
    // operands and must stay pure — the fix above must not blanket the family.
    for kind in [
        PredicateGenKind::True,
        PredicateGenKind::False,
        PredicateGenKind::Next,
        PredicateGenKind::First,
        PredicateGenKind::UnpackHi,
        PredicateGenKind::UnpackLo,
        PredicateGenKind::Select,
        PredicateGenKind::HazardRw,
        PredicateGenKind::HazardWr,
    ] {
        let effects = SsaOp::<MockTarget>::VectorPredicateGen(Box::new(KindedVecData {
            kind,
            outputs: vec![d],
            inputs: vec![a],
        }))
        .effects();
        assert!(effects.is_pure(), "{kind:?} computes a value and is pure");
    }

    // `setffr`/`wrffr` write the FFR, which is not an SSA operand. With no
    // outputs, a pure classification lets DCE delete them.
    let setffr = SsaOp::<MockTarget>::VectorPredicateOp(Box::new(VectorPredicateOpData {
        op: PredicateOpKind::SetFirstFault,
        element_bits: 32,
        outputs: vec![],
        inputs: vec![],
    }))
    .effects();
    assert!(
        !setffr.removable_when_unused(),
        "setffr writes the first-fault register and must not be DCE'd"
    );

    // A predicate op that only computes a value stays pure.
    let count_active = SsaOp::<MockTarget>::VectorPredicateOp(Box::new(VectorPredicateOpData {
        op: PredicateOpKind::CountActive,
        element_bits: 32,
        outputs: vec![d],
        inputs: vec![a],
    }))
    .effects();
    assert!(count_active.is_pure(), "cntp computes a value and is pure");

    // `dsb`/`dmb`/`isb` order memory: they are fences, not plain writes.
    let barrier = SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::Barrier,
        mnemonic: "dmb".into(),
        metadata: None,
        clobbers: vec![],
        outputs: vec![],
        inputs: vec![],
    }))
    .effects();
    assert_eq!(
        barrier.kind,
        SsaEffectKind::Fence,
        "a memory barrier must classify as a fence, not a write"
    );
    assert_eq!(barrier.ordering, Some(AtomicOrdering::SeqCst));

    let branch = SsaOp::<MockTarget>::Branch {
        condition: a,
        true_target: 1,
        false_target: 2,
    }
    .effects();
    assert_eq!(branch.control, ControlEffect::Terminator);

    let fence = SsaOp::<MockTarget>::Fence {
        kind: FenceKind::Acquire,
    }
    .effects();
    assert_eq!(fence.memory_semantics, MemoryAccessSemantics::Fence);
    assert_eq!(fence.ordering, Some(AtomicOrdering::Acquire));

    assert_eq!(
        SsaOp::<MockTarget>::Call {
            dest: Some(d),
            method: 1,
            args: vec![a],
        }
        .effects()
        .kind,
        SsaEffectKind::Call
    );
}

/// A [`SystemOpKind`] is always carried by [`SsaOp::SystemOp`], which is
/// **not** a structural terminator ([`SsaOp::is_terminator`] excludes it).
/// The `check_native_effects` verifier invariant rejects any op whose
/// `effects().control` is block-ending (`Terminator`/`Return`/`Throw`)
/// unless it is a terminator — so **no** system-op kind may declare a
/// block-ending control effect. Regression guard for `iret`/`eret`
/// (`SystemOpKind::InterruptReturn`), which formerly declared
/// `ControlEffect::Return` and made every mid-block `iret` fail lift SSA
/// verification.
#[test]
fn system_op_kinds_never_declare_a_block_ending_control_effect() {
    let all = [
        SystemOpKind::CpuId,
        SystemOpKind::Timestamp { aux: false },
        SystemOpKind::Timestamp { aux: true },
        SystemOpKind::ReadSysReg {
            namespace: SysRegNamespace::X86Msr,
        },
        SystemOpKind::WriteSysReg {
            namespace: SysRegNamespace::Arm64System,
        },
        SystemOpKind::ReadPerfCounter,
        SystemOpKind::SystemCall,
        SystemOpKind::SystemReturn,
        SystemOpKind::Trap { vector: None },
        SystemOpKind::Trap { vector: Some(0x80) },
        SystemOpKind::InterruptReturn,
        SystemOpKind::CacheMaintenance,
        SystemOpKind::TlbMaintenance,
        SystemOpKind::Barrier,
        SystemOpKind::Privileged,
        SystemOpKind::Hypervisor,
        SystemOpKind::HardwareEngine,
        SystemOpKind::Transaction(SystemTransactionKind::Start),
        SystemOpKind::Transaction(SystemTransactionKind::Commit),
        SystemOpKind::Transaction(SystemTransactionKind::Cancel),
        SystemOpKind::Transaction(SystemTransactionKind::Test),
    ];
    for kind in all {
        let control = kind.effects().control;
        assert!(
            !matches!(
                control,
                ControlEffect::Terminator | ControlEffect::Return | ControlEffect::Throw
            ),
            "SystemOpKind {kind:?} declares block-ending control {control:?} but \
             SsaOp::SystemOp is not a terminator",
        );
    }
    // `iret`/`eret` specifically transfers control externally like `sysret`.
    assert_eq!(
        SystemOpKind::InterruptReturn.effects().control,
        ControlEffect::Call,
    );
    assert!(
        !SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::InterruptReturn,
            mnemonic: String::from("iret"),
            metadata: None,
            outputs: Vec::new(),
            inputs: Vec::new(),
            clobbers: Vec::new(),
        }))
        .is_terminator()
    );
}

#[test]
fn op_class_groups_native_scalar_vector_memory_and_control_ops() {
    let d = SsaVarId::from_index(0);
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);

    assert_eq!(
        SsaOp::<MockTarget>::Add {
            dest: d,
            left: a,
            right: b,
            flags: None,
        }
        .class(),
        SsaOpClass::Scalar
    );
    assert_eq!(
        SsaOp::<MockTarget>::Add {
            dest: d,
            left: a,
            right: b,
            flags: Some(SsaVarId::from_index(3)),
        }
        .class(),
        SsaOpClass::Flags
    );
    assert_eq!(
        SsaOp::<MockTarget>::VectorBinary {
            dest: d,
            left: a,
            right: b,
            kind: VectorBinaryKind::Add,
            element: VectorElement::default(),
        }
        .class(),
        SsaOpClass::Vector
    );
    assert_eq!(
        SsaOp::<MockTarget>::AtomicRmw {
            dest: d,
            addr: a,
            value: b,
            op: AtomicRmwOp::Add,
        }
        .class(),
        SsaOpClass::Atomic
    );
    assert_eq!(
        SsaOp::<MockTarget>::Branch {
            condition: a,
            true_target: 1,
            false_target: 2,
        }
        .class(),
        SsaOpClass::Control
    );
    assert_eq!(
        SsaOp::<MockTarget>::NativeOpaque(Box::new(NativeOpaqueData {
            mnemonic: "ud2".to_string(),
            metadata: None,
            outputs: Vec::new(),
            inputs: Vec::new(),
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, true),
        }))
        .class(),
        SsaOpClass::NativeOpaque
    );
    assert_eq!(
        SsaOp::<MockTarget>::NativeIntrinsic(Box::new(NativeIntrinsicData {
            id: NativeIntrinsicId::Rdtsc,
            mnemonic: "rdtsc".to_string(),
            metadata: None,
            outputs: vec![d],
            inputs: Vec::new(),
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, false),
        }))
        .class(),
        SsaOpClass::NativeIntrinsic
    );
}

/// Builds a battery of representative ops covering every visitor arm shape.
fn visitor_battery() -> Vec<SsaOp<MockTarget>> {
    let v: Vec<SsaVarId> = (0..8).map(SsaVarId::from_index).collect();
    vec![
        SsaOp::Const {
            dest: v[0],
            value: ConstValue::I32(7),
        },
        SsaOp::Add {
            dest: v[0],
            left: v[1],
            right: v[2],
            flags: Some(v[3]),
        },
        SsaOp::Neg {
            dest: v[0],
            operand: v[1],
            flags: None,
        },
        SsaOp::Shr {
            dest: v[0],
            value: v[1],
            amount: v[2],
            unsigned: true,
            flags: Some(v[3]),
        },
        SsaOp::WideMul {
            low: v[0],
            high: v[1],
            left: v[2],
            right: v[3],
            unsigned: false,
        },
        SsaOp::WideDiv {
            quotient: v[0],
            remainder: v[1],
            high: v[2],
            low: v[3],
            divisor: v[4],
            unsigned: false,
        },
        SsaOp::FloatCompareFlags {
            flags: v[0],
            left: v[1],
            right: v[2],
            signaling: false,
        },
        SsaOp::Select {
            dest: v[0],
            condition: v[1],
            true_val: v[2],
            false_val: v[3],
        },
        SsaOp::StoreIndirect {
            addr: v[0],
            value: v[1],
            value_type: MockType::I32,
            address_space: None,
        },
        SsaOp::AtomicCmpXchg {
            old: v[0],
            success: Some(v[1]),
            addr: v[2],
            expected: v[3],
            desired: v[4],
            success_ordering: AtomicOrdering::SeqCst,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits64,
            weak: false,
            volatile: false,
        },
        SsaOp::AtomicPairLoad {
            first: v[0],
            second: v[1],
            addr: v[2],
            first_type: MockType::I64,
            second_type: MockType::I64,
            ordering: AtomicOrdering::Acquire,
            width: AtomicAccessWidth::Bits128,
            volatile: false,
        },
        SsaOp::Call {
            dest: Some(v[0]),
            method: 3,
            args: vec![v[1], v[2], v[3]],
        },
        SsaOp::CallIndirect {
            dest: None,
            fptr: v[0],
            signature: 0,
            args: vec![v[1]],
        },
        SsaOp::Return { value: Some(v[0]) },
        SsaOp::Return { value: None },
        SsaOp::Phi {
            dest: v[0],
            operands: vec![(0, v[1]), (1, v[2])],
        },
        SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
            mnemonic: "ud2".to_string(),
            metadata: None,
            outputs: vec![v[0], v[1]],
            inputs: vec![v[2], v[3]],
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, true),
        })),
        SsaOp::NativeIntrinsic(Box::new(NativeIntrinsicData {
            id: NativeIntrinsicId::Cpuid,
            mnemonic: "cpuid".to_string(),
            metadata: None,
            outputs: vec![v[0], v[1]],
            inputs: vec![v[2]],
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, false),
        })),
        SsaOp::BcdAdjust(Box::new(BcdAdjustData {
            kind: BcdAdjustKind::AsciiMulAdjust,
            base: 10,
            mnemonic: "aam".to_string(),
            metadata: None,
            outputs: vec![v[0], v[1]],
            inputs: vec![v[2]],
            clobbers: Vec::new(),
        })),
        SsaOp::VectorDotProduct(Box::new(VectorDotProductData {
            imm8: 0xff,
            element_bits: 32,
            outputs: vec![v[0]],
            inputs: vec![v[1], v[2]],
        })),
        SsaOp::VectorMultiSad(Box::new(VecImm8Data {
            imm8: 0x05,
            outputs: vec![v[0]],
            inputs: vec![v[1], v[2]],
        })),
        SsaOp::VectorStringCompare(Box::new(VectorStringCompareData {
            imm8: 0x0c,
            explicit_length: true,
            result_index: true,
            outputs: vec![v[0], v[1]],
            inputs: vec![v[2], v[3]],
        })),
        SsaOp::VectorHorizontalMinPos(Box::new(VectorHorizontalMinPosData {
            outputs: vec![v[0]],
            inputs: vec![v[1]],
        })),
        SsaOp::VectorConditionalMove(Box::new(VectorConditionalMoveData {
            condition: ByteMoveCondition::Negative,
            outputs: vec![v[0]],
            inputs: vec![v[1], v[2]],
        })),
        SsaOp::VectorIntersect(Box::new(VectorIntersectData {
            outputs: vec![v[0], v[1]],
            inputs: vec![v[2], v[3]],
        })),
        SsaOp::VectorShuffleBits(Box::new(VectorShuffleBitsData {
            outputs: vec![v[0]],
            inputs: vec![v[1], v[2]],
        })),
        SsaOp::VectorBitfield(Box::new(VectorBitfieldData {
            insert: false,
            index: 4,
            length: 8,
            outputs: vec![v[0]],
            inputs: vec![v[1]],
        })),
        SsaOp::Jump { target: 4 },
        SsaOp::Nop,
    ]
}

/// A typed `VectorBitfield` reports its output as a def / inputs as uses, is
/// pure, and survives a variable remap with its fields intact.
#[test]
fn vector_bitfield_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);
    let c = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::VectorBitfield(Box::new(VectorBitfieldData {
        insert: true,
        index: 16,
        length: 8,
        outputs: vec![a],
        inputs: vec![b, c],
    }));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert_eq!(op.uses(), vec![b, c]);
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);
    let remapped = op.remap_variables(|x| Some(SsaVarId::from_index(x.index() + 10)));
    let SsaOp::VectorBitfield(data) = &remapped else {
        unreachable!("remap must preserve the VectorBitfield variant")
    };
    assert!(data.insert);
    assert_eq!(data.index, 16);
    assert_eq!(data.length, 8);
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(
        data.inputs,
        vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
    );
}

/// A typed `VectorHorizontalMinPos` reports its output as a def / input as a
/// use, is pure, and survives a variable remap.
#[test]
fn vector_horizontal_minpos_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);
    let op: SsaOp<MockTarget> =
        SsaOp::VectorHorizontalMinPos(Box::new(VectorHorizontalMinPosData {
            outputs: vec![a],
            inputs: vec![b],
        }));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert_eq!(op.uses(), vec![b]);
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);
    let remapped = op.remap_variables(|x| Some(SsaVarId::from_index(x.index() + 10)));
    let SsaOp::VectorHorizontalMinPos(data) = &remapped else {
        unreachable!("remap must preserve the VectorHorizontalMinPos variant")
    };
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(data.inputs, vec![SsaVarId::from_index(11)]);
}

/// A typed `VectorStringCompare` reports outputs as defs / inputs as uses,
/// is pure, and survives a variable remap with its imm8 and flags intact.
#[test]
fn vector_string_compare_defs_uses_effects_and_remap() {
    let v: Vec<SsaVarId> = (0..4).map(SsaVarId::from_index).collect();
    let op: SsaOp<MockTarget> = SsaOp::VectorStringCompare(Box::new(VectorStringCompareData {
        imm8: 0x0c,
        explicit_length: false,
        result_index: false,
        outputs: vec![v[0], v[1]],
        inputs: vec![v[2], v[3]],
    }));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![v[0], v[1]]);
    assert_eq!(op.uses(), vec![v[2], v[3]]);
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);
    let remapped = op.remap_variables(|x| Some(SsaVarId::from_index(x.index() + 10)));
    let SsaOp::VectorStringCompare(data) = &remapped else {
        unreachable!("remap must preserve the VectorStringCompare variant")
    };
    assert_eq!(data.imm8, 0x0c);
    assert!(!data.explicit_length);
    assert!(!data.result_index);
    assert_eq!(
        data.outputs,
        vec![SsaVarId::from_index(10), SsaVarId::from_index(11)]
    );
    assert_eq!(
        data.inputs,
        vec![SsaVarId::from_index(12), SsaVarId::from_index(13)]
    );
}

/// A typed `VectorMultiSad` reports its output as a def / inputs as uses, is
/// pure, and survives a variable remap with its imm8 intact.
#[test]
fn vector_multi_sad_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);
    let c = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::VectorMultiSad(Box::new(VecImm8Data {
        imm8: 0x05,
        outputs: vec![a],
        inputs: vec![b, c],
    }));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert_eq!(op.uses(), vec![b, c]);
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);
    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(11)),
        v if v == c => Some(SsaVarId::from_index(12)),
        _ => None,
    });
    let SsaOp::VectorMultiSad(data) = &remapped else {
        unreachable!("remap must preserve the VectorMultiSad variant")
    };
    assert_eq!(data.imm8, 0x05);
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(
        data.inputs,
        vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
    );
}

/// A typed `VectorDotProduct` reports its output as a def / inputs as uses,
/// is pure, and survives a variable remap with its imm8 and width intact.
#[test]
fn vector_dot_product_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);
    let c = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::VectorDotProduct(Box::new(VectorDotProductData {
        imm8: 0x31,
        element_bits: 64,
        outputs: vec![a],
        inputs: vec![b, c],
    }));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert_eq!(op.uses(), vec![b, c]);
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);
    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(11)),
        v if v == c => Some(SsaVarId::from_index(12)),
        _ => None,
    });
    let SsaOp::VectorDotProduct(data) = &remapped else {
        unreachable!("remap must preserve the VectorDotProduct variant")
    };
    assert_eq!(data.imm8, 0x31);
    assert_eq!(data.element_bits, 64);
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(
        data.inputs,
        vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
    );
}

/// A typed `BcdAdjust` reports outputs as defs / inputs as uses, is pure,
/// and survives a variable remap with its kind and radix intact.
#[test]
fn bcd_adjust_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);
    let c = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::BcdAdjust(Box::new(BcdAdjustData {
        kind: BcdAdjustKind::AsciiDivAdjust,
        base: 16,
        mnemonic: "aad".to_string(),
        metadata: None,
        outputs: vec![a, b],
        inputs: vec![c],
        clobbers: Vec::new(),
    }));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a, b]);
    assert_eq!(op.uses(), vec![c]);
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);
    assert!(!op.effects().may_throw);
    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(11)),
        v if v == c => Some(SsaVarId::from_index(12)),
        _ => None,
    });
    let SsaOp::BcdAdjust(data) = &remapped else {
        unreachable!("remap must preserve the BcdAdjust variant")
    };
    assert_eq!(data.kind, BcdAdjustKind::AsciiDivAdjust);
    assert_eq!(data.base, 16);
    assert_eq!(
        data.outputs,
        vec![SsaVarId::from_index(10), SsaVarId::from_index(11)]
    );
    assert_eq!(data.inputs, vec![SsaVarId::from_index(12)]);
}

/// Compile-time completeness guard. This exhaustive match (deliberately no
/// `_` arm) names every `SsaOp` variant, so adding a new variant breaks the
/// build here until it is consciously handled. That is the signal to (1)
/// classify it in the exhaustive `effects()` match, (2) cover its operands
/// in `visit_operands` / `visit_operands_mut`, and (3) add a sample to
/// `visitor_battery` so the invariant tests exercise it. Never executed — it
/// exists purely for the exhaustiveness check.
#[allow(dead_code)]
fn op_variant_exhaustiveness_sentinel(op: SsaOp<MockTarget>) {
    match op {
        SsaOp::Const { .. } => {}
        SsaOp::Add { .. } => {}
        SsaOp::AddOvf { .. } => {}
        SsaOp::Sub { .. } => {}
        SsaOp::SubOvf { .. } => {}
        SsaOp::Mul { .. } => {}
        SsaOp::MulOvf { .. } => {}
        SsaOp::WideMul { .. } => {}
        SsaOp::Div { .. } => {}
        SsaOp::Rem { .. } => {}
        SsaOp::FloatCompareFlags { .. } => {}
        SsaOp::WideDiv { .. } => {}
        SsaOp::Neg { .. } => {}
        SsaOp::And { .. } => {}
        SsaOp::Or { .. } => {}
        SsaOp::Xor { .. } => {}
        SsaOp::Not { .. } => {}
        SsaOp::Shl { .. } => {}
        SsaOp::Shr { .. } => {}
        SsaOp::Rol { .. } => {}
        SsaOp::Ror { .. } => {}
        SsaOp::Rcl { .. } => {}
        SsaOp::Rcr { .. } => {}
        SsaOp::BSwap { .. } => {}
        SsaOp::BRev { .. } => {}
        SsaOp::BitScanForward { .. } => {}
        SsaOp::BitScanReverse { .. } => {}
        SsaOp::Popcount { .. } => {}
        SsaOp::Parity { .. } => {}
        SsaOp::Ceq { .. } => {}
        SsaOp::Clt { .. } => {}
        SsaOp::Cgt { .. } => {}
        SsaOp::BoolAnd { .. } => {}
        SsaOp::BoolOr { .. } => {}
        SsaOp::BoolXor { .. } => {}
        SsaOp::BoolNot { .. } => {}
        SsaOp::IntConv { .. } => {}
        SsaOp::IntToPtr { .. } => {}
        SsaOp::PtrToInt { .. } => {}
        SsaOp::IntToFloat { .. } => {}
        SsaOp::FloatToInt { .. } => {}
        SsaOp::FloatConv { .. } => {}
        SsaOp::Bitcast { .. } => {}
        SsaOp::Select { .. } => {}
        SsaOp::ReadFlags { .. } => {}
        SsaOp::VectorUnary { .. } => {}
        SsaOp::VectorBinary { .. } => {}
        SsaOp::VectorTernary { .. } => {}
        SsaOp::VectorPredicatedUnary { .. } => {}
        SsaOp::VectorPredicatedBinary { .. } => {}
        SsaOp::VectorPredicatedTernary { .. } => {}
        SsaOp::VectorCompare { .. } => {}
        SsaOp::VectorLoad { .. } => {}
        SsaOp::VectorStore { .. } => {}
        SsaOp::VectorMaskedLoad { .. } => {}
        SsaOp::VectorMaskedStore { .. } => {}
        SsaOp::VectorBroadcastLoad { .. } => {}
        SsaOp::VectorGather { .. } => {}
        SsaOp::VectorFaultingLoad { .. } => {}
        SsaOp::VectorSegmentLoad { .. } => {}
        SsaOp::VectorScatter { .. } => {}
        SsaOp::VectorSegmentStore { .. } => {}
        SsaOp::VectorExtract { .. } => {}
        SsaOp::VectorInsert { .. } => {}
        SsaOp::VectorSplat { .. } => {}
        SsaOp::VectorShuffle { .. } => {}
        SsaOp::VectorCast { .. } => {}
        SsaOp::VectorReinterpret { .. } => {}
        SsaOp::VectorPack { .. } => {}
        SsaOp::VectorPackLoad { .. } => {}
        SsaOp::VectorPackStore { .. } => {}
        SsaOp::VectorZeroUpper { .. } => {}
        SsaOp::VectorMaskUnary { .. } => {}
        SsaOp::VectorMaskBinary { .. } => {}
        SsaOp::VectorReduce { .. } => {}
        SsaOp::VectorBitmask { .. } => {}
        SsaOp::Jump { .. } => {}
        SsaOp::Branch { .. } => {}
        SsaOp::BranchCmp { .. } => {}
        SsaOp::BranchFlags { .. } => {}
        SsaOp::Switch { .. } => {}
        SsaOp::IndirectBranch { .. } => {}
        SsaOp::Return { .. } => {}
        SsaOp::LoadField { .. } => {}
        SsaOp::StoreField { .. } => {}
        SsaOp::LoadStaticField { .. } => {}
        SsaOp::StoreStaticField { .. } => {}
        SsaOp::LoadFieldAddr { .. } => {}
        SsaOp::LoadStaticFieldAddr { .. } => {}
        SsaOp::LoadElement { .. } => {}
        SsaOp::StoreElement { .. } => {}
        SsaOp::LoadElementAddr { .. } => {}
        SsaOp::PtrAdd { .. } => {}
        SsaOp::ArrayLength { .. } => {}
        SsaOp::LoadIndirect { .. } => {}
        SsaOp::StoreIndirect { .. } => {}
        SsaOp::NewObj { .. } => {}
        SsaOp::NewArr { .. } => {}
        SsaOp::CastClass { .. } => {}
        SsaOp::IsInst { .. } => {}
        SsaOp::Box { .. } => {}
        SsaOp::Unbox { .. } => {}
        SsaOp::UnboxAny { .. } => {}
        SsaOp::SizeOf { .. } => {}
        SsaOp::LoadToken { .. } => {}
        SsaOp::Call { .. } => {}
        SsaOp::CallVirt { .. } => {}
        SsaOp::CallIndirect { .. } => {}
        SsaOp::LoadFunctionPtr { .. } => {}
        SsaOp::LoadVirtFunctionPtr { .. } => {}
        SsaOp::LoadArg { .. } => {}
        SsaOp::LoadLocal { .. } => {}
        SsaOp::LoadArgAddr { .. } => {}
        SsaOp::LoadLocalAddr { .. } => {}
        SsaOp::Copy { .. } => {}
        SsaOp::Pop { .. } => {}
        SsaOp::Throw { .. } => {}
        SsaOp::Rethrow => {}
        SsaOp::EndFinally => {}
        SsaOp::EndFilter { .. } => {}
        SsaOp::InterruptReturn => {}
        SsaOp::Unreachable => {}
        SsaOp::Leave { .. } => {}
        SsaOp::InitBlk { .. } => {}
        SsaOp::CopyBlk { .. } => {}
        SsaOp::Fence { .. } => {}
        SsaOp::NativeOpaque(_) => {}
        SsaOp::NativeIntrinsic(_) => {}
        SsaOp::SystemOp(_) => {}
        SsaOp::ComputeOp(_) => {}
        SsaOp::BcdAdjust(_) => {}
        SsaOp::VectorCrypto(_) => {}
        SsaOp::TileOp(_) => {}
        SsaOp::VectorPermute(_) => {}
        SsaOp::VectorMultiplyAdd(_) => {}
        SsaOp::VectorPackNarrow(_) => {}
        SsaOp::VectorNarrowSaturate(_) => {}
        SsaOp::VectorPredicateWhile(_) => {}
        SsaOp::VectorPredicateBreak(_) => {}
        SsaOp::VectorComplexAdd(_) => {}
        SsaOp::VectorCountAdjust(_) => {}
        SsaOp::VectorExtendInLane(_) => {}
        SsaOp::VectorElementCount(_) => {}
        SsaOp::VectorSveAddressGen(_) => {}
        SsaOp::FlagAdjust(_) => {}
        SsaOp::VectorStructLoadReplicate(_) => {}
        SsaOp::VectorSmeMisc(_) => {}
        SsaOp::VectorPredicateOp(_) => {}
        SsaOp::VectorSveCompute(_) => {}
        SsaOp::VectorReverseChunks(_) => {}
        SsaOp::VectorMatrixMulAcc(_) => {}
        SsaOp::VectorSmeOuterProduct(_) => {}
        SsaOp::VectorPredicateGen(_) => {}
        SsaOp::VectorFpHelper(_) => {}
        SsaOp::VectorSvePermute(_) => {}
        SsaOp::VectorTernaryLogic(_) => {}
        SsaOp::VectorDotProduct(_) => {}
        SsaOp::VectorMultiSad(_) => {}
        SsaOp::VectorIntDotProduct(_) => {}
        SsaOp::VectorStringCompare(_) => {}
        SsaOp::VectorBitfield(_) => {}
        SsaOp::VectorIntersect(_) => {}
        SsaOp::VectorShuffleBits(_) => {}
        SsaOp::VectorConditionalMove(_) => {}
        SsaOp::VectorHorizontalMinPos(_) => {}
        SsaOp::VectorComplexMul(_) => {}
        SsaOp::VectorClassify(_) => {}
        SsaOp::VectorHorizontalReduce(_) => {}
        SsaOp::BlockString(_) => {}
        SsaOp::WideCompareExchange(_) => {}
        SsaOp::ComputeFlags { .. } => {}
        SsaOp::CallClobber { .. } => {}
        SsaOp::CmpXchg { .. } => {}
        SsaOp::AtomicRmw { .. } => {}
        SsaOp::AtomicLoad { .. } => {}
        SsaOp::AtomicStore { .. } => {}
        SsaOp::AtomicStoreConditional { .. } => {}
        SsaOp::AtomicPairLoad { .. } => {}
        SsaOp::AtomicPairStoreConditional { .. } => {}
        SsaOp::AtomicExchange { .. } => {}
        SsaOp::AtomicLockRmw { .. } => {}
        SsaOp::AtomicCmpXchg { .. } => {}
        SsaOp::AtomicPairCmpXchg { .. } => {}
        SsaOp::InitObj { .. } => {}
        SsaOp::CopyObj { .. } => {}
        SsaOp::LoadObj { .. } => {}
        SsaOp::StoreObj { .. } => {}
        SsaOp::Nop => {}
        SsaOp::Break => {}
        SsaOp::Ckfinite { .. } => {}
        SsaOp::FpClassify { .. } => {}
        SsaOp::FpTranscendental(_) => {}
        SsaOp::FpuControl(_) => {}
        SsaOp::LocalAlloc { .. } => {}
        SsaOp::Constrained { .. } => {}
        SsaOp::Volatile => {}
        SsaOp::Unaligned { .. } => {}
        SsaOp::TailPrefix => {}
        SsaOp::Readonly => {}
        SsaOp::Phi { .. } => {}
    }
}

/// Constructs one sample of every `SsaOp` variant (all 200). Field values
/// are placeholders chosen only to be type-valid; the tests assert
/// structural invariants, not semantics. Kept complete by
/// `op_variant_exhaustiveness_sentinel`: adding a variant breaks the build
/// there until a sample is added here too.
fn all_sample_ops() -> Vec<SsaOp<MockTarget>> {
    let sv = SsaVarId::from_index(1);
    vec![
        SsaOp::Const {
            dest: sv,
            value: ConstValue::I32(0),
        },
        SsaOp::Add {
            dest: sv,
            left: sv,
            right: sv,
            flags: Some(sv),
        },
        SsaOp::AddOvf {
            dest: sv,
            left: sv,
            right: sv,
            unsigned: false,
            flags: Some(sv),
        },
        SsaOp::Sub {
            dest: sv,
            left: sv,
            right: sv,
            flags: Some(sv),
        },
        SsaOp::SubOvf {
            dest: sv,
            left: sv,
            right: sv,
            unsigned: false,
            flags: Some(sv),
        },
        SsaOp::Mul {
            dest: sv,
            left: sv,
            right: sv,
            flags: Some(sv),
        },
        SsaOp::MulOvf {
            dest: sv,
            left: sv,
            right: sv,
            unsigned: false,
            flags: Some(sv),
        },
        SsaOp::WideMul {
            low: sv,
            high: sv,
            left: sv,
            right: sv,
            unsigned: false,
        },
        SsaOp::Div {
            dest: sv,
            left: sv,
            right: sv,
            unsigned: false,
            flags: Some(sv),
        },
        SsaOp::Rem {
            dest: sv,
            left: sv,
            right: sv,
            unsigned: false,
            flags: Some(sv),
        },
        SsaOp::FloatCompareFlags {
            flags: sv,
            left: sv,
            right: sv,
            signaling: false,
        },
        SsaOp::WideDiv {
            quotient: sv,
            remainder: sv,
            high: sv,
            low: sv,
            divisor: sv,
            unsigned: false,
        },
        SsaOp::Neg {
            dest: sv,
            operand: sv,
            flags: Some(sv),
        },
        SsaOp::And {
            dest: sv,
            left: sv,
            right: sv,
            flags: Some(sv),
        },
        SsaOp::Or {
            dest: sv,
            left: sv,
            right: sv,
            flags: Some(sv),
        },
        SsaOp::Xor {
            dest: sv,
            left: sv,
            right: sv,
            flags: Some(sv),
        },
        SsaOp::Not {
            dest: sv,
            operand: sv,
            flags: Some(sv),
        },
        SsaOp::Shl {
            dest: sv,
            value: sv,
            amount: sv,
            flags: Some(sv),
        },
        SsaOp::Shr {
            dest: sv,
            value: sv,
            amount: sv,
            unsigned: false,
            flags: Some(sv),
        },
        SsaOp::Rol {
            dest: sv,
            value: sv,
            amount: sv,
        },
        SsaOp::Ror {
            dest: sv,
            value: sv,
            amount: sv,
        },
        SsaOp::Rcl {
            dest: sv,
            value: sv,
            amount: sv,
        },
        SsaOp::Rcr {
            dest: sv,
            value: sv,
            amount: sv,
        },
        SsaOp::BSwap { dest: sv, src: sv },
        SsaOp::BRev { dest: sv, src: sv },
        SsaOp::BitScanForward { dest: sv, src: sv },
        SsaOp::BitScanReverse { dest: sv, src: sv },
        SsaOp::Popcount { dest: sv, src: sv },
        SsaOp::Parity { dest: sv, src: sv },
        SsaOp::Ceq {
            dest: sv,
            left: sv,
            right: sv,
        },
        SsaOp::Clt {
            dest: sv,
            left: sv,
            right: sv,
            unsigned: false,
        },
        SsaOp::Cgt {
            dest: sv,
            left: sv,
            right: sv,
            unsigned: false,
        },
        SsaOp::BoolAnd {
            dest: sv,
            left: sv,
            right: sv,
        },
        SsaOp::BoolOr {
            dest: sv,
            left: sv,
            right: sv,
        },
        SsaOp::BoolXor {
            dest: sv,
            left: sv,
            right: sv,
        },
        SsaOp::BoolNot {
            dest: sv,
            value: sv,
        },
        SsaOp::IntConv {
            dest: sv,
            operand: sv,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        },
        SsaOp::IntToPtr {
            dest: sv,
            operand: sv,
            target: MockType::I32,
        },
        SsaOp::PtrToInt {
            dest: sv,
            operand: sv,
            target: MockType::I32,
        },
        SsaOp::IntToFloat {
            dest: sv,
            operand: sv,
            target: MockType::I32,
            unsigned: false,
        },
        SsaOp::FloatToInt {
            dest: sv,
            operand: sv,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        },
        SsaOp::FloatConv {
            dest: sv,
            operand: sv,
            target: MockType::I32,
        },
        SsaOp::Bitcast {
            dest: sv,
            operand: sv,
            target: MockType::I32,
        },
        SsaOp::Select {
            dest: sv,
            condition: sv,
            true_val: sv,
            false_val: sv,
        },
        SsaOp::ReadFlags {
            dest: sv,
            flags: sv,
            mask: FlagsMask::from_bits(0),
        },
        SsaOp::VectorUnary {
            dest: sv,
            value: sv,
            kind: VectorUnaryKind::Neg,
            element: VectorElement {
                kind: VectorElementKind::Integer,
                bits: 32,
                scalar: false,
            },
        },
        SsaOp::VectorBinary {
            dest: sv,
            left: sv,
            right: sv,
            kind: VectorBinaryKind::Add,
            element: VectorElement {
                kind: VectorElementKind::Integer,
                bits: 32,
                scalar: false,
            },
        },
        SsaOp::VectorTernary {
            dest: sv,
            first: sv,
            second: sv,
            third: sv,
            kind: VectorTernaryKind::Fma,
        },
        SsaOp::VectorPredicatedUnary {
            dest: sv,
            value: sv,
            mask: sv,
            passthrough: Some(sv),
            kind: VectorUnaryKind::Neg,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorPredicatedBinary {
            dest: sv,
            left: sv,
            right: sv,
            mask: sv,
            passthrough: Some(sv),
            kind: VectorBinaryKind::Add,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorPredicatedTernary {
            dest: sv,
            first: sv,
            second: sv,
            third: sv,
            mask: sv,
            passthrough: Some(sv),
            kind: VectorTernaryKind::Fma,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorCompare {
            dest: sv,
            left: sv,
            right: sv,
            kind: VectorCompareKind::Eq,
            unsigned: false,
        },
        SsaOp::VectorLoad {
            dest: sv,
            addr: sv,
            vector_type: MockType::I32,
        },
        SsaOp::VectorStore {
            addr: sv,
            value: sv,
            vector_type: MockType::I32,
        },
        SsaOp::VectorMaskedLoad {
            dest: sv,
            addr: sv,
            mask: sv,
            passthrough: Some(sv),
            vector_type: MockType::I32,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorMaskedStore {
            addr: sv,
            value: sv,
            mask: sv,
            vector_type: MockType::I32,
        },
        SsaOp::VectorBroadcastLoad {
            dest: sv,
            addr: sv,
            vector_type: MockType::I32,
        },
        SsaOp::VectorGather {
            dest: sv,
            base: sv,
            indices: sv,
            mask: sv,
            passthrough: Some(sv),
            vector_type: MockType::I32,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorFaultingLoad {
            dest: sv,
            fault: Some(sv),
            addr: sv,
            mask: Some(sv),
            passthrough: Some(sv),
            vector_type: MockType::I32,
            fault_mode: VectorFaultMode::Normal,
            mask_mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorSegmentLoad {
            dests: vec![sv],
            base: sv,
            mask: Some(sv),
            vector_type: MockType::I32,
            segments: 0,
            layout: VectorSegmentLayout::Interleaved,
        },
        SsaOp::VectorScatter {
            base: sv,
            indices: sv,
            value: sv,
            mask: sv,
            vector_type: MockType::I32,
        },
        SsaOp::VectorSegmentStore {
            base: sv,
            values: vec![sv],
            mask: Some(sv),
            vector_type: MockType::I32,
            segments: 0,
            layout: VectorSegmentLayout::Interleaved,
        },
        SsaOp::VectorExtract {
            dest: sv,
            vector: sv,
            lane: 0,
        },
        SsaOp::VectorInsert {
            dest: sv,
            vector: sv,
            lane: 0,
            value: sv,
        },
        SsaOp::VectorSplat {
            dest: sv,
            value: sv,
            vector_type: MockType::I32,
        },
        SsaOp::VectorShuffle {
            dest: sv,
            left: sv,
            right: Some(sv),
            mask: VectorShuffleMask::new(vec![crate::target::VectorShuffleLane::Zero]),
        },
        SsaOp::VectorCast {
            dest: sv,
            value: sv,
            target_type: MockType::I32,
            kind: VectorCastKind::Signed,
        },
        SsaOp::VectorReinterpret {
            dest: sv,
            value: sv,
            target_type: MockType::I32,
        },
        SsaOp::VectorPack {
            dest: sv,
            value: sv,
            mask: sv,
            passthrough: Some(sv),
            vector_type: MockType::I32,
            element_bits: 0,
            kind: VectorPackKind::Compress,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorPackLoad {
            dest: sv,
            addr: sv,
            mask: sv,
            passthrough: Some(sv),
            vector_type: MockType::I32,
            element_bits: 0,
            kind: VectorPackKind::Compress,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorPackStore {
            addr: sv,
            value: sv,
            mask: sv,
            vector_type: MockType::I32,
            element_bits: 0,
            kind: VectorPackKind::Compress,
        },
        SsaOp::VectorZeroUpper { all: false },
        SsaOp::VectorMaskUnary {
            dest: sv,
            mask: sv,
            kind: VectorMaskUnaryKind::Not,
        },
        SsaOp::VectorMaskBinary {
            dest: sv,
            left: sv,
            right: sv,
            kind: VectorMaskBinaryKind::And,
        },
        SsaOp::VectorReduce {
            dest: sv,
            value: sv,
            kind: VectorReduceKind::Add,
        },
        SsaOp::VectorBitmask {
            dest: sv,
            value: sv,
            kind: VectorBitmaskKind::LaneMostSignificantBits,
        },
        SsaOp::Jump { target: 0 },
        SsaOp::Branch {
            condition: sv,
            true_target: 0,
            false_target: 0,
        },
        SsaOp::BranchCmp {
            left: sv,
            right: sv,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 0,
            false_target: 0,
        },
        SsaOp::BranchFlags {
            flags: sv,
            condition: FlagCondition::Carry,
            true_target: 0,
            false_target: 0,
        },
        SsaOp::Switch {
            value: sv,
            targets: vec![0usize],
            default: 0,
        },
        SsaOp::IndirectBranch {
            target: sv,
            resolved_targets: vec![0usize],
        },
        SsaOp::Return { value: Some(sv) },
        SsaOp::LoadField {
            dest: sv,
            object: sv,
            field: 0u32,
        },
        SsaOp::StoreField {
            object: sv,
            field: 0u32,
            value: sv,
        },
        SsaOp::LoadStaticField {
            dest: sv,
            field: 0u32,
        },
        SsaOp::StoreStaticField {
            field: 0u32,
            value: sv,
        },
        SsaOp::LoadFieldAddr {
            dest: sv,
            object: sv,
            field: 0u32,
        },
        SsaOp::LoadStaticFieldAddr {
            dest: sv,
            field: 0u32,
        },
        SsaOp::LoadElement {
            dest: sv,
            array: sv,
            index: sv,
            elem_type: MockType::I32,
        },
        SsaOp::StoreElement {
            array: sv,
            index: sv,
            value: sv,
            elem_type: MockType::I32,
        },
        SsaOp::LoadElementAddr {
            dest: sv,
            array: sv,
            index: sv,
            elem_type: 0u32,
        },
        SsaOp::PtrAdd {
            dest: sv,
            base: sv,
            index: Some(sv),
            stride: 4,
            offset: 8,
            result_type: MockType::I64,
        },
        SsaOp::ArrayLength {
            dest: sv,
            array: sv,
        },
        SsaOp::LoadIndirect {
            dest: sv,
            addr: sv,
            value_type: MockType::I32,
            address_space: None,
        },
        SsaOp::StoreIndirect {
            addr: sv,
            value: sv,
            value_type: MockType::I32,
            address_space: None,
        },
        SsaOp::NewObj {
            dest: sv,
            ctor: 0u32,
            args: vec![sv],
        },
        SsaOp::NewArr {
            dest: sv,
            elem_type: 0u32,
            length: sv,
        },
        SsaOp::CastClass {
            dest: sv,
            object: sv,
            target_type: 0u32,
        },
        SsaOp::IsInst {
            dest: sv,
            object: sv,
            target_type: 0u32,
        },
        SsaOp::Box {
            dest: sv,
            value: sv,
            value_type: 0u32,
        },
        SsaOp::Unbox {
            dest: sv,
            object: sv,
            value_type: 0u32,
        },
        SsaOp::UnboxAny {
            dest: sv,
            object: sv,
            value_type: 0u32,
        },
        SsaOp::SizeOf {
            dest: sv,
            value_type: 0u32,
        },
        SsaOp::LoadToken {
            dest: sv,
            token: 0u32,
        },
        SsaOp::Call {
            dest: Some(sv),
            method: 0u32,
            args: vec![sv],
        },
        SsaOp::CallVirt {
            dest: Some(sv),
            method: 0u32,
            args: vec![sv],
        },
        SsaOp::CallIndirect {
            dest: Some(sv),
            fptr: sv,
            signature: 0u32,
            args: vec![sv],
        },
        SsaOp::LoadFunctionPtr {
            dest: sv,
            method: 0u32,
        },
        SsaOp::LoadVirtFunctionPtr {
            dest: sv,
            object: sv,
            method: 0u32,
        },
        SsaOp::LoadArg {
            dest: sv,
            arg_index: 0,
        },
        SsaOp::LoadLocal {
            dest: sv,
            local_index: 0,
        },
        SsaOp::LoadArgAddr {
            dest: sv,
            arg_index: 0,
        },
        SsaOp::LoadLocalAddr {
            dest: sv,
            local_index: 0,
        },
        SsaOp::Copy { dest: sv, src: sv },
        SsaOp::Pop { value: sv },
        SsaOp::Throw { exception: sv },
        SsaOp::Rethrow,
        SsaOp::EndFinally,
        SsaOp::EndFilter { result: sv },
        SsaOp::InterruptReturn,
        SsaOp::Unreachable,
        SsaOp::Leave { target: 0 },
        SsaOp::InitBlk {
            dest_addr: sv,
            value: sv,
            size: sv,
            reverse: false,
        },
        SsaOp::CopyBlk {
            dest_addr: sv,
            src_addr: sv,
            size: sv,
            reverse: false,
        },
        SsaOp::Fence {
            kind: FenceKind::Full,
        },
        SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![sv],
            inputs: vec![sv],
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, false),
        })),
        SsaOp::NativeIntrinsic(Box::new(NativeIntrinsicData {
            id: NativeIntrinsicId::Cpuid,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![sv],
            inputs: vec![sv],
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, false),
        })),
        SsaOp::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::CpuId,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![sv],
            inputs: vec![sv],
            clobbers: Vec::new(),
        })),
        SsaOp::ComputeOp(Box::new(NativeKindedData {
            kind: ComputeKind::BitDeposit,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![sv],
            inputs: vec![sv],
            clobbers: Vec::new(),
        })),
        SsaOp::BcdAdjust(Box::new(BcdAdjustData {
            kind: BcdAdjustKind::DecimalAddAdjust,
            base: 0,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![sv],
            inputs: vec![sv],
            clobbers: Vec::new(),
        })),
        SsaOp::VectorCrypto(Box::new(KindedVecData {
            kind: VectorCryptoKind::AesEncrypt,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::TileOp(Box::new(KindedVecData {
            kind: TileOpKind::Zero,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorPermute(Box::new(VectorPermuteData {
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorMultiplyAdd(Box::new(KindedVecData {
            kind: VectorMaddKind::MultiplyAddS16,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorPackNarrow(Box::new(VectorPackNarrowData {
            unsigned: false,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorNarrowSaturate(Box::new(VectorNarrowSaturateData {
            signed_src: false,
            unsigned_dst: false,
            rounding: false,
            shift: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorPredicateWhile(Box::new(VectorPredicateWhileData {
            kind: VectorCompareKind::Eq,
            unsigned: false,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorPredicateBreak(Box::new(VectorPredicateBreakData {
            after: false,
            pair: false,
            propagate: false,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorComplexAdd(Box::new(VectorComplexAddData {
            rotate_270: false,
            saturate: false,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorCountAdjust(Box::new(VectorCountAdjustData {
            decrement: false,
            saturate: false,
            signed: false,
            by_predicate: false,
            element_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorExtendInLane(Box::new(VectorExtendInLaneData {
            signed: false,
            source_bits: 0,
            element_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorElementCount(Box::new(VectorElementCountData {
            element_bits: 0,
            multiplier: 0,
            outputs: vec![sv],
        })),
        SsaOp::VectorSveAddressGen(Box::new(VectorSveAddressGenData {
            signed_extend: Some(false),
            shift: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::FlagAdjust(Box::new(KindedVecData {
            kind: FlagAdjustKind::InvertCarry,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorStructLoadReplicate(Box::new(VectorStructLoadReplicateData {
            count: 0,
            element_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorSmeMisc(Box::new(VectorSmeMiscData {
            op: SmeMiscKind::AddHorizontal,
            element_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorPredicateOp(Box::new(VectorPredicateOpData {
            op: PredicateOpKind::CountActive,
            element_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorSveCompute(Box::new(VectorSveComputeData {
            op: SveComputeKind::AddCarryBottom,
            element_bits: 0,
            rotation: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorReverseChunks(Box::new(VectorReverseChunksData {
            chunk_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorMatrixMulAcc(Box::new(VectorMatrixMulAccData {
            signed_a: false,
            signed_b: false,
            float: false,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorSmeOuterProduct(Box::new(VectorSmeOuterProductData {
            subtract: false,
            signed_a: false,
            signed_b: false,
            float: false,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorPredicateGen(Box::new(KindedVecData {
            kind: PredicateGenKind::True,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorFpHelper(Box::new(KindedVecData {
            kind: FpHelperKind::ReciprocalExponent,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorSvePermute(Box::new(KindedVecData {
            kind: SvePermuteKind::Index,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorTernaryLogic(Box::new(VecImm8Data {
            imm8: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorDotProduct(Box::new(VectorDotProductData {
            imm8: 0,
            element_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorMultiSad(Box::new(VecImm8Data {
            imm8: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorIntDotProduct(Box::new(VectorIntDotProductData {
            signed_a: false,
            signed_b: false,
            source_bits: 0,
            dest_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorStringCompare(Box::new(VectorStringCompareData {
            imm8: 0,
            explicit_length: false,
            result_index: false,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorBitfield(Box::new(VectorBitfieldData {
            insert: false,
            index: 0,
            length: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorIntersect(Box::new(VectorIntersectData {
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorShuffleBits(Box::new(VectorShuffleBitsData {
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorConditionalMove(Box::new(VectorConditionalMoveData {
            condition: ByteMoveCondition::Zero,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorHorizontalMinPos(Box::new(VectorHorizontalMinPosData {
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorComplexMul(Box::new(KindedVecData {
            kind: ComplexMulKind::Multiply,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorClassify(Box::new(VecImm8Data {
            imm8: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::VectorHorizontalReduce(Box::new(VectorHorizontalReduceData {
            subtract: false,
            unsigned: false,
            source_bits: 0,
            dest_bits: 0,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::BlockString(Box::new(BlockStringOpData {
            kind: BlockStringKind::Compare,
            prefix: BlockStringPrefix::Repeat,
            element_bits: 0,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![sv],
            inputs: vec![sv],
            clobbers: Vec::new(),
            reverse: false,
        })),
        SsaOp::WideCompareExchange(Box::new(WideCmpXchgData {
            wide: false,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![sv],
            inputs: vec![sv],
            clobbers: Vec::new(),
        })),
        SsaOp::ComputeFlags {
            dest: sv,
            inputs: vec![sv],
        },
        SsaOp::CallClobber { outputs: vec![sv] },
        SsaOp::CmpXchg {
            dest: sv,
            addr: sv,
            expected: sv,
            desired: sv,
        },
        SsaOp::AtomicRmw {
            dest: sv,
            addr: sv,
            value: sv,
            op: AtomicRmwOp::Xchg,
        },
        SsaOp::AtomicLoad {
            dest: sv,
            addr: sv,
            value_type: MockType::I32,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            volatile: false,
        },
        SsaOp::AtomicStore {
            addr: sv,
            value: sv,
            value_type: MockType::I32,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            volatile: false,
        },
        SsaOp::AtomicStoreConditional {
            status: sv,
            addr: sv,
            value: sv,
            value_type: MockType::I32,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            volatile: false,
        },
        SsaOp::AtomicPairLoad {
            first: sv,
            second: sv,
            addr: sv,
            first_type: MockType::I32,
            second_type: MockType::I32,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            volatile: false,
        },
        SsaOp::AtomicPairStoreConditional {
            status: sv,
            addr: sv,
            first_value: sv,
            second_value: sv,
            first_type: MockType::I32,
            second_type: MockType::I32,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            volatile: false,
        },
        SsaOp::AtomicExchange {
            dest: sv,
            addr: sv,
            value: sv,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            volatile: false,
        },
        SsaOp::AtomicLockRmw {
            dest: sv,
            addr: sv,
            value: sv,
            op: AtomicRmwOp::Xchg,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            volatile: false,
        },
        SsaOp::AtomicCmpXchg {
            old: sv,
            success: Some(sv),
            addr: sv,
            expected: sv,
            desired: sv,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            weak: false,
            volatile: false,
        },
        SsaOp::AtomicPairCmpXchg {
            old_first: sv,
            old_second: sv,
            addr: sv,
            expected_first: sv,
            expected_second: sv,
            desired_first: sv,
            desired_second: sv,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            weak: false,
            volatile: false,
        },
        SsaOp::InitObj {
            dest_addr: sv,
            value_type: 0u32,
        },
        SsaOp::CopyObj {
            dest_addr: sv,
            src_addr: sv,
            value_type: 0u32,
        },
        SsaOp::LoadObj {
            dest: sv,
            src_addr: sv,
            value_type: 0u32,
        },
        SsaOp::StoreObj {
            dest_addr: sv,
            value: sv,
            value_type: 0u32,
        },
        SsaOp::Nop,
        SsaOp::Break,
        SsaOp::Ckfinite {
            dest: sv,
            operand: sv,
        },
        SsaOp::FpClassify {
            dest: sv,
            operand: sv,
        },
        SsaOp::FpTranscendental(Box::new(KindedVecData {
            kind: TranscendentalKind::Sin,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::FpuControl(Box::new(KindedVecData {
            kind: FpuControlKind::LoadControlWord,
            outputs: vec![sv],
            inputs: vec![sv],
        })),
        SsaOp::LocalAlloc { dest: sv, size: sv },
        SsaOp::Constrained {
            constraint_type: 0u32,
        },
        SsaOp::Volatile,
        SsaOp::Unaligned { alignment: 0 },
        SsaOp::TailPrefix,
        SsaOp::Readonly,
        SsaOp::Phi {
            dest: sv,
            operands: vec![(0usize, sv)],
        },
    ]
}

/// Across ALL 200 variants: the operand visitor agrees with `defs()` /
/// `uses()`, the primary `dest()` is the first `Def`-role operand, and
/// `opcode_name()` is unique per variant. This is the runtime half of the
/// completeness guarantee (the sentinel is the compile-time half).
#[test]
fn all_variants_visitor_defs_uses_and_opcode_are_consistent() {
    let ops = all_sample_ops();
    assert_eq!(
        ops.len(),
        200,
        "every SsaOp variant must have exactly one sample"
    );
    let mut names = std::collections::HashSet::new();
    for op in &ops {
        let mut visitor_defs = Vec::new();
        let mut visitor_uses = Vec::new();
        let mut first_def = None;
        op.visit_operands(|role, var| match role {
            OperandRole::Def => {
                if first_def.is_none() {
                    first_def = Some(var);
                }
                visitor_defs.push(var);
            }
            OperandRole::FlagsDef => visitor_defs.push(var),
            OperandRole::Use => visitor_uses.push(var),
        });
        assert_eq!(
            visitor_defs,
            op.defs().collect::<Vec<_>>(),
            "defs mismatch for {op}"
        );
        assert_eq!(visitor_uses, op.uses(), "uses mismatch for {op}");
        assert_eq!(op.dest(), first_def, "dest is not the first def for {op}");
        assert!(
            names.insert(op.opcode_name()),
            "opcode_name {:?} is not unique",
            op.opcode_name()
        );
    }
    assert_eq!(names.len(), 200, "opcode_name must be unique per variant");
}

/// Cross-checks the classification methods against each other for every op
/// in the battery: `is_pure()` must mean exactly "pure kind and cannot
/// throw", and `effects().may_throw` must agree with the `may_throw()`
/// predicate. These invariants are what the generic passes (DCE, GVN, LICM)
/// rely on, so a mis-specified op is caught here. (`Rcl`/`Rcr` regressions,
/// for instance, surface as a purity mismatch.)
#[test]
fn classification_methods_are_self_consistent() {
    for op in all_sample_ops() {
        let eff = op.effects();
        assert_eq!(
            op.is_pure(),
            eff.kind == SsaEffectKind::Pure && !eff.may_throw,
            "is_pure() disagrees with effects() for {op}"
        );
        assert_eq!(
            eff.may_throw,
            op.may_throw(),
            "effects().may_throw disagrees with may_throw() for {op}"
        );
    }
}

/// The visitor must agree with `defs()` (definition set and order) and
/// `uses()` (use set and order) for every representative op shape.
#[test]
fn visit_operands_agrees_with_defs_and_uses() {
    for op in all_sample_ops().into_iter().chain(visitor_battery()) {
        let mut visited_defs = Vec::new();
        let mut visited_uses = Vec::new();
        op.visit_operands(|role, var| match role {
            OperandRole::Def | OperandRole::FlagsDef => visited_defs.push(var),
            OperandRole::Use => visited_uses.push(var),
        });

        let defs: Vec<SsaVarId> = op.defs().collect();
        assert_eq!(visited_defs, defs, "defs mismatch for {op}");
        assert_eq!(visited_uses, op.uses(), "uses mismatch for {op}");
        if matches!(op, SsaOp::FloatCompareFlags { .. }) {
            // Defines only a flags bundle; there is no primary destination.
            assert_eq!(op.dest(), None);
        } else {
            assert_eq!(op.dest(), defs.first().copied(), "dest mismatch for {op}");
        }
    }
}

/// `visit_operands_mut` must visit exactly the same operands, in the same
/// order and with the same roles, as `visit_operands` — the invariant its
/// doc comment states.
///
/// The two are ~1200-line parallel matches over all 200 `SsaOp` variants,
/// and they are consumed for *different* purposes: the shared visitor feeds
/// def-use index construction, while the mutable one performs variable
/// substitution (`replace_uses`, `replace_def`). Drift between them is
/// therefore silent SSA corruption rather than a crash — an operand visited
/// by only the immutable arm is recorded as a use but never rewritten by a
/// rename, leaving a dangling reference the verifier only catches later, if
/// at all. The compile-time exhaustiveness sentinel forces a new variant to
/// be *considered* in both, but cannot check that both got it right.
///
/// Runs over both batteries, which are complementary: `all_sample_ops()`
/// reaches all 200 variants but populates optional operands, while
/// `visitor_battery()` covers fewer variants and includes the absent shapes
/// (`flags: None`) that exercise the visitors' skip paths.
#[test]
fn visit_operands_mut_agrees_with_visit_operands() {
    for mut op in all_sample_ops().into_iter().chain(visitor_battery()) {
        let mut shared = Vec::new();
        op.visit_operands(|role, var| shared.push((role, var)));

        let mut mutable = Vec::new();
        op.visit_operands_mut(|role, var| mutable.push((role, *var)));

        assert_eq!(
            shared, mutable,
            "visit_operands and visit_operands_mut disagree for {op}"
        );
    }
}

/// `replace_def` rewrites secondary outputs uniformly, across every op that has
/// them — the native-intrinsic output list as well as `NativeOpaque`.
#[test]
fn replace_def_covers_native_intrinsic_outputs() {
    let old = SsaVarId::from_index(1);
    let new = SsaVarId::from_index(9);
    let mut op: SsaOp<MockTarget> = SsaOp::NativeIntrinsic(Box::new(NativeIntrinsicData {
        id: NativeIntrinsicId::Rdtsc,
        mnemonic: "rdtsc".to_string(),
        metadata: None,
        outputs: vec![SsaVarId::from_index(0), old],
        inputs: vec![old],
        clobbers: Vec::new(),
        effects: SsaEffects::new(SsaEffectKind::Opaque, false),
    }));
    assert!(op.replace_def(old, new));
    let SsaOp::NativeIntrinsic(data) = &op else {
        unreachable!()
    };
    assert_eq!(data.outputs, vec![SsaVarId::from_index(0), new]);
    assert_eq!(data.inputs, vec![old], "uses must stay untouched");
}

/// A typed `SystemOp` reports its outputs as defs and inputs as uses, its
/// effects derive from the kind (never opaque), and `remap_variables`
/// rewrites both lists while leaving the kind/metadata intact.
#[test]
fn system_op_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::ReadSysReg {
            namespace: SysRegNamespace::X86Msr,
        },
        mnemonic: "rdmsr".to_string(),
        metadata: None,
        outputs: vec![a],
        inputs: vec![b],
        clobbers: Vec::new(),
    }));

    // defs == outputs, uses == inputs.
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert!(op.uses().contains(&b), "input must appear in uses");

    // Effects are precise (Read for a sysreg read), never Opaque.
    assert_eq!(op.effects().kind, SsaEffectKind::Read);

    // remap rewrites both inputs and outputs.
    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(20)),
        _ => None,
    });
    let SsaOp::SystemOp(data) = &remapped else {
        unreachable!("remap must preserve the SystemOp variant")
    };
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
    assert_eq!(
        data.kind,
        SystemOpKind::ReadSysReg {
            namespace: SysRegNamespace::X86Msr
        },
        "kind must survive remap"
    );
}

/// A typed `ComputeOp` reports outputs as defs / inputs as uses, derives
/// precise effects from the kind (pure for bit-permute, `Read` for the
/// nondeterministic random source), and survives `remap_variables`.
#[test]
fn compute_op_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);
    let pdep: SsaOp<MockTarget> = SsaOp::ComputeOp(Box::new(NativeKindedData {
        kind: ComputeKind::BitDeposit,
        mnemonic: "pdep".to_string(),
        metadata: None,
        outputs: vec![a],
        inputs: vec![b],
        clobbers: Vec::new(),
    }));
    assert_eq!(pdep.defs().collect::<Vec<_>>(), vec![a]);
    assert!(pdep.uses().contains(&b));
    // pdep is pure; rdrand reads a nondeterministic entropy source.
    assert_eq!(pdep.effects().kind, SsaEffectKind::Pure);

    let rdrand: SsaOp<MockTarget> = SsaOp::ComputeOp(Box::new(NativeKindedData {
        kind: ComputeKind::Random {
            from_entropy: false,
        },
        mnemonic: "rdrand".to_string(),
        metadata: None,
        outputs: vec![a],
        inputs: Vec::new(),
        clobbers: Vec::new(),
    }));
    assert_eq!(rdrand.effects().kind, SsaEffectKind::Read);

    let remapped = pdep.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(20)),
        _ => None,
    });
    let SsaOp::ComputeOp(data) = &remapped else {
        unreachable!("remap must preserve the ComputeOp variant")
    };
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
    assert_eq!(data.kind, ComputeKind::BitDeposit);
}

/// A typed `BlockString` reports outputs as defs / inputs as uses, derives
/// precise effects from the kind (`ReadWrite` for compare, `Read` for load),
/// preserves prefix/element_bits, and survives `remap_variables`.
#[test]
fn block_string_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);
    let cmps: SsaOp<MockTarget> = SsaOp::BlockString(Box::new(BlockStringOpData {
        kind: BlockStringKind::Compare,
        prefix: BlockStringPrefix::RepeatEqual,
        element_bits: 8,
        mnemonic: "repe cmps".to_string(),
        metadata: None,
        outputs: vec![a],
        inputs: vec![b],
        clobbers: Vec::new(),
        reverse: false,
    }));
    assert_eq!(cmps.defs().collect::<Vec<_>>(), vec![a]);
    assert!(cmps.uses().contains(&b));
    assert_eq!(cmps.effects().kind, SsaEffectKind::ReadWrite);

    let lods: SsaOp<MockTarget> = SsaOp::BlockString(Box::new(BlockStringOpData {
        kind: BlockStringKind::Load,
        prefix: BlockStringPrefix::Repeat,
        element_bits: 32,
        mnemonic: "rep lods".to_string(),
        metadata: None,
        outputs: vec![a],
        inputs: Vec::new(),
        clobbers: Vec::new(),
        reverse: false,
    }));
    assert_eq!(lods.effects().kind, SsaEffectKind::Read);

    let remapped = cmps.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(20)),
        _ => None,
    });
    let SsaOp::BlockString(data) = &remapped else {
        unreachable!("remap must preserve the BlockString variant")
    };
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
    assert_eq!(data.prefix, BlockStringPrefix::RepeatEqual);
    assert_eq!(data.element_bits, 8);
}

/// A typed `WideCompareExchange` reports outputs as defs / inputs as uses,
/// has a sequentially-consistent atomic effect (never opaque), and survives
/// `remap_variables` with its `wide` flag intact.
#[test]
fn wide_compare_exchange_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::WideCompareExchange(Box::new(WideCmpXchgData {
        wide: true,
        mnemonic: "cmpxchg16b".to_string(),
        metadata: None,
        outputs: vec![a],
        inputs: vec![b],
        clobbers: Vec::new(),
    }));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert!(op.uses().contains(&b));
    assert_eq!(op.effects().kind, SsaEffectKind::Atomic);
    assert_eq!(op.effects().ordering, Some(AtomicOrdering::SeqCst));

    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(20)),
        _ => None,
    });
    let SsaOp::WideCompareExchange(data) = &remapped else {
        unreachable!("remap must preserve the WideCompareExchange variant")
    };
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(data.inputs, vec![SsaVarId::from_index(20)]);
    assert!(data.wide, "wide flag must survive remap");
}

/// A typed `ComputeFlags` defines its flags value, uses its inputs, is pure
/// (so optimization can eliminate it), and survives `remap_variables`.
#[test]
fn compute_flags_defs_uses_effects_and_remap() {
    let dest = SsaVarId::from_index(1);
    let a = SsaVarId::from_index(2);
    let b = SsaVarId::from_index(3);
    let op: SsaOp<MockTarget> = SsaOp::ComputeFlags {
        dest,
        inputs: vec![a, b],
    };
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![dest]);
    assert!(op.uses().contains(&a) && op.uses().contains(&b));
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);

    let remapped = op.remap_variables(|v| match v {
        v if v == dest => Some(SsaVarId::from_index(11)),
        v if v == a => Some(SsaVarId::from_index(12)),
        v if v == b => Some(SsaVarId::from_index(13)),
        _ => None,
    });
    let SsaOp::ComputeFlags { dest, inputs } = &remapped else {
        unreachable!("remap must preserve the ComputeFlags variant")
    };
    assert_eq!(*dest, SsaVarId::from_index(11));
    assert_eq!(
        inputs,
        &vec![SsaVarId::from_index(12), SsaVarId::from_index(13)]
    );
}

/// A typed `CallClobber` defines all its outputs, has no uses, is pure (so
/// dead clobbers are eliminable), and survives `remap_variables`.
#[test]
fn call_clobber_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::CallClobber {
        outputs: vec![a, b],
    };
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a, b]);
    assert!(op.uses().is_empty());
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);

    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(11)),
        v if v == b => Some(SsaVarId::from_index(12)),
        _ => None,
    });
    let SsaOp::CallClobber { outputs } = &remapped else {
        unreachable!("remap must preserve the CallClobber variant")
    };
    assert_eq!(
        outputs,
        &vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
    );
}

#[test]
fn payload_accessors_report_signedness_compare_and_memory() {
    let d = SsaVarId::from_index(0);
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);

    let udiv: SsaOp<MockTarget> = SsaOp::Div {
        dest: d,
        left: a,
        right: b,
        unsigned: true,
        flags: None,
    };
    assert_eq!(udiv.arith_signedness(), Some(Signedness::Unsigned));
    assert_eq!(udiv.compare_kind(), None);
    assert!(udiv.memory_effect().is_none());

    let clt: SsaOp<MockTarget> = SsaOp::Clt {
        dest: d,
        left: a,
        right: b,
        unsigned: false,
    };
    assert_eq!(clt.arith_signedness(), Some(Signedness::Signed));
    assert_eq!(clt.compare_kind(), Some(CmpKind::Lt));

    let branch_cmp: SsaOp<MockTarget> = SsaOp::BranchCmp {
        left: a,
        right: b,
        cmp: CmpKind::Ge,
        unsigned: true,
        true_target: 1,
        false_target: 2,
    };
    assert_eq!(branch_cmp.compare_kind(), Some(CmpKind::Ge));
    assert_eq!(branch_cmp.arith_signedness(), Some(Signedness::Unsigned));

    let add: SsaOp<MockTarget> = SsaOp::Add {
        dest: d,
        left: a,
        right: b,
        flags: None,
    };
    assert_eq!(add.arith_signedness(), None);
    assert_eq!(add.compare_kind(), None);
    assert!(add.memory_effect().is_none());

    let load: SsaOp<MockTarget> = SsaOp::LoadIndirect {
        dest: d,
        addr: a,
        value_type: MockType::I32,
        address_space: None,
    };
    let effect = load.memory_effect().expect("load has a memory effect");
    assert_eq!(effect.addr, a);
    assert!(effect.reads);
    assert!(!effect.writes);
    assert_eq!(effect.value_type, Some(&MockType::I32));

    let rmw: SsaOp<MockTarget> = SsaOp::AtomicRmw {
        dest: d,
        addr: a,
        value: b,
        op: AtomicRmwOp::Add,
    };
    let effect = rmw.memory_effect().expect("rmw has a memory effect");
    assert!(effect.reads);
    assert!(effect.writes);
    assert_eq!(effect.value_type, None);
}

#[test]
fn similarity_class_groups_feature_extraction_families() {
    let d = SsaVarId::from_index(0);
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);

    assert_eq!(
        SsaOp::<MockTarget>::Const {
            dest: d,
            value: ConstValue::I32(1),
        }
        .similarity_class(),
        SsaSimilarityClass::Constant
    );
    assert_eq!(
        SsaOp::<MockTarget>::Add {
            dest: d,
            left: a,
            right: b,
            flags: None,
        }
        .similarity_class(),
        SsaSimilarityClass::Arithmetic
    );
    assert_eq!(
        SsaOp::<MockTarget>::Add {
            dest: d,
            left: a,
            right: b,
            flags: Some(SsaVarId::from_index(3)),
        }
        .similarity_class(),
        SsaSimilarityClass::Flags
    );
    assert_eq!(
        SsaOp::<MockTarget>::Xor {
            dest: d,
            left: a,
            right: b,
            flags: None,
        }
        .similarity_class(),
        SsaSimilarityClass::Bitwise
    );
    assert_eq!(
        SsaOp::<MockTarget>::Ceq {
            dest: d,
            left: a,
            right: b,
        }
        .similarity_class(),
        SsaSimilarityClass::Compare
    );
    assert_eq!(
        SsaOp::<MockTarget>::VectorFaultingLoad {
            dest: d,
            fault: None,
            addr: a,
            mask: None,
            passthrough: None,
            vector_type: MockType::V4I32,
            fault_mode: VectorFaultMode::Normal,
            mask_mode: VectorMaskMode::Zero,
        }
        .similarity_class(),
        SsaSimilarityClass::MemoryRead
    );
    assert_eq!(
        SsaOp::<MockTarget>::VectorBinary {
            dest: d,
            left: a,
            right: b,
            kind: VectorBinaryKind::Add,
            element: VectorElement::default(),
        }
        .similarity_class(),
        SsaSimilarityClass::Vector
    );
    assert_eq!(
        SsaOp::<MockTarget>::AtomicExchange {
            dest: d,
            addr: a,
            value: b,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            volatile: false,
        }
        .similarity_class(),
        SsaSimilarityClass::Atomic
    );
    assert_eq!(
        SsaOp::<MockTarget>::Fence {
            kind: FenceKind::SeqCst,
        }
        .similarity_class(),
        SsaSimilarityClass::Fence
    );
    assert_eq!(
        SsaOp::<MockTarget>::NativeOpaque(Box::new(NativeOpaqueData {
            mnemonic: "ud2".to_string(),
            metadata: None,
            outputs: Vec::new(),
            inputs: Vec::new(),
            clobbers: Vec::new(),
            effects: SsaEffects::new(SsaEffectKind::Opaque, true),
        }))
        .similarity_class(),
        SsaSimilarityClass::NativeOpaque
    );
}

#[test]
fn feature_token_serializes_stable_target_generic_shape() {
    let d = SsaVarId::from_index(0);
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);

    let op: SsaOp<MockTarget> = SsaOp::AtomicCmpXchg {
        old: d,
        success: Some(SsaVarId::from_index(3)),
        addr: a,
        expected: b,
        desired: d,
        success_ordering: AtomicOrdering::SeqCst,
        failure_ordering: AtomicOrdering::Acquire,
        width: AtomicAccessWidth::Bits32,
        weak: false,
        volatile: true,
    };
    let token = op.feature_token();

    assert_eq!(token.opcode, "atomic.cmpxchg");
    assert_eq!(token.op_class, SsaOpClass::Atomic);
    assert_eq!(token.similarity_class, SsaSimilarityClass::Atomic);
    assert_eq!(token.effect_kind, SsaEffectKind::Atomic);
    assert_eq!(token.def_count, 2);
    assert_eq!(token.use_count, 3);
    assert!(token.may_throw);
    assert_eq!(
        token.to_string(),
        "op=atomic.cmpxchg;class=Atomic;sim=Atomic;effect=Atomic;defs=2;uses=3;throw=true"
    );
}

#[test]
fn native_register_aliases_track_subregister_overlap() {
    let rax = NativeRegister::new("x86_64", "gpr", "rax", "rax", 0, 64).unwrap();
    let eax = NativeRegister::new("x86_64", "gpr", "rax", "eax", 0, 32).unwrap();
    let ah = NativeRegister::new("x86_64", "gpr", "rax", "ah", 8, 8).unwrap();
    let rbx = NativeRegister::new("x86_64", "gpr", "rbx", "rbx", 0, 64).unwrap();
    let q0 = NativeRegister::new("aarch64", "simd", "v0", "q0", 0, 128).unwrap();

    assert!(rax.aliases(&eax));
    assert!(eax.aliases(&ah));
    assert!(!rax.aliases(&rbx));
    assert!(!rax.aliases(&q0));
    assert!(NativeRegister::new("x86_64", "gpr", "rax", "al", 0, 0).is_none());
}

#[test]
fn native_state_accesses_classify_implicit_machine_state() {
    let rflags = NativeStateAccess::implicit_read_write(
        NativeStateLocation::Flags("rflags".to_string()),
        Some(64),
    )
    .unwrap();
    assert!(rflags.reads());
    assert!(rflags.writes());
    assert!(rflags.implicit);

    let vl = NativeStateAccess::implicit_read(NativeStateLocation::VectorLength, None).unwrap();
    assert!(vl.reads());
    assert!(!vl.writes());
    assert!(
        NativeStateAccess::implicit_write(NativeStateLocation::StackPointer, Some(0)).is_none()
    );
}

#[test]
fn native_clobbers_expose_structured_machine_state_categories() {
    let rax = NativeRegister::new("x86_64", "gpr", "rax", "rax", 0, 64).unwrap();
    let reg = NativeClobber::MachineState(
        NativeStateAccess::implicit_read_write(NativeStateLocation::Register(rax), Some(64))
            .unwrap(),
    );
    let flags = NativeClobber::Flags("eflags".to_string());
    let memory = NativeClobber::MachineState(
        NativeStateAccess::implicit_write(NativeStateLocation::Memory("io".to_string()), None)
            .unwrap(),
    );

    assert!(reg.touches_registers());
    assert!(!reg.touches_memory());
    assert!(flags.touches_flags());
    assert!(memory.touches_memory());
}

#[test]
fn native_opaque_tracks_outputs_inputs_and_effects() {
    let out0 = SsaVarId::from_index(0);
    let out1 = SsaVarId::from_index(1);
    let in0 = SsaVarId::from_index(2);
    let in1 = SsaVarId::from_index(3);
    let op: SsaOp<MockTarget> = SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
        mnemonic: "mulx".to_string(),
        metadata: Some(NativeInstructionMetadata::new(
            Some("x86_64".to_string()),
            Some(0x1000),
            vec![0xc4, 0xe2, 0xfb, 0xf6],
        )),
        outputs: vec![out0, out1],
        inputs: vec![in0, in1],
        clobbers: vec![NativeClobber::Flags("eflags".to_string())],
        effects: SsaEffects::new(SsaEffectKind::ReadWrite, true),
    }));

    assert_eq!(op.dest(), Some(out0));
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![out0, out1]);
    assert_eq!(op.uses(), vec![in0, in1]);
    assert_eq!(op.stack_effect(), (2, 2));
    assert_eq!(
        op.effects(),
        SsaEffects::new(SsaEffectKind::ReadWrite, true)
    );
    assert!(op.may_throw());
    assert!(!op.is_pure());
}

#[test]
fn native_opaque_rewrites_defs_and_uses_separately() {
    let out0 = SsaVarId::from_index(0);
    let out1 = SsaVarId::from_index(1);
    let new_out = SsaVarId::from_index(9);
    let input = SsaVarId::from_index(2);
    let new_input = SsaVarId::from_index(10);
    let mut op: SsaOp<MockTarget> = SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
        mnemonic: "opaque".to_string(),
        metadata: None,
        outputs: vec![out0, out1],
        inputs: vec![input],
        clobbers: Vec::new(),
        effects: SsaEffects::pure(),
    }));

    assert!(op.replace_def(out1, new_out));
    assert_eq!(op.replace_uses(input, new_input), 1);
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![out0, new_out]);
    assert_eq!(op.uses(), vec![new_input]);

    let remapped = op.remap_variables(|var| {
        if var == out0 {
            Some(SsaVarId::from_index(20))
        } else if var == new_input {
            Some(SsaVarId::from_index(30))
        } else {
            None
        }
    });
    assert_eq!(
        remapped.defs().collect::<Vec<_>>(),
        vec![SsaVarId::from_index(20), new_out]
    );
    assert_eq!(remapped.uses(), vec![SsaVarId::from_index(30)]);
}

#[cfg(test)]
mod size_guards {
    //! Regression guards on the in-memory size of the core IR value types.
    //!
    //! These bounds were established after boxing the rare fat `SsaOp` /
    //! `ConstValue` variants (`NativeOpaque`, wide atomics, decrypted
    //! string/array/vector payloads) and giving `SsaVarId` a niche so
    //! `Option<SsaVarId>` is 4 bytes. Every `SsaBlock` stores a
    //! `Vec<SsaInstruction>`, so a regrowth here multiplies across an entire
    //! function. If adding a variant trips one of these, prefer boxing the new
    //! variant's payload over relaxing the bound.
    use super::*;
    use crate::{
        ir::{instruction::SsaInstruction, value::ConstValue},
        testing::MockTarget,
    };

    #[test]
    fn core_ir_types_stay_compact() {
        assert!(
            std::mem::size_of::<Option<SsaVarId>>() <= 4,
            "Option<SsaVarId> grew to {} bytes; SsaVarId lost its niche",
            std::mem::size_of::<Option<SsaVarId>>()
        );
        assert!(
            std::mem::size_of::<ConstValue<MockTarget>>() <= 24,
            "ConstValue grew to {} bytes; box the new heap-bearing arm",
            std::mem::size_of::<ConstValue<MockTarget>>()
        );
        assert!(
            std::mem::size_of::<SsaOp<MockTarget>>() <= 40,
            "SsaOp grew to {} bytes; box the new fat variant's payload",
            std::mem::size_of::<SsaOp<MockTarget>>()
        );
        assert!(
            std::mem::size_of::<SsaInstruction<MockTarget>>() <= 48,
            "SsaInstruction grew to {} bytes",
            std::mem::size_of::<SsaInstruction<MockTarget>>()
        );
    }
}

/// Every SSA operand an op *uses* must appear in its rendered form.
///
/// `Display` is the diagnostic surface for the whole IR — it is what pass
/// debugging, `verify_hard` failures, and golden snapshots read. An op that
/// silently drops an operand renders two semantically different instructions
/// identically, which is both misleading and invisible. This is the runtime
/// counterpart of the existing exhaustive `defs`/`visit_operands` cross-checks:
/// `BranchFlags` omitted the flags variable it branches on, so
/// `branchflags v5 zero` and `branchflags v9 zero` printed the same.
#[test]
fn display_renders_every_used_operand() {
    for op in all_sample_ops() {
        let rendered = format!("{op}");
        let mut missing: Vec<SsaVarId> = Vec::new();
        op.for_each_use(|used| {
            // Operands print as `v<index>`; require the exact token so `v1`
            // does not spuriously match inside `v13`.
            let token = format!("v{}", used.index());
            let found = rendered
                .split(|c: char| !c.is_ascii_alphanumeric())
                .any(|word| word == token);
            if !found && !missing.contains(&used) {
                missing.push(used);
            }
        });
        assert!(
            missing.is_empty(),
            "Display for `{rendered}` omits operand(s) {missing:?} that `for_each_use` reports"
        );
    }
}
