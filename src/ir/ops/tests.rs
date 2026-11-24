//! Unit tests for the [`SsaOp`] surface.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Write as _},
    mem::{self, discriminant},
    num::NonZeroU16,
};

use rustc_hash::{FxHashMap, FxHashSet};

use super::*;
use crate::{
    ir::{
        aes,
        float::{
            FloatClassLayout, FloatExceptionLayout, FloatExceptions, FloatFormat, FloatRounding,
        },
        ghash,
        lanes::VectorConst,
        sha, sm3, sm4,
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{VectorShuffleLane, VectorShuffleMask, VectorTailPolicy},
    testing::{MockTarget, MockType},
};

/// The coarse token deliberately merges variants a similarity fingerprint
/// should treat alike, folds signedness, and returns a bare `"call"` for a
/// call - leaving classification to a target-binding consumer.
#[test]
fn coarse_token_coarsens_merges_and_folds() {
    let v = SsaVarId::from_index(0);

    // `Add` and its overflow-checked sibling collapse to one token, while
    // `opcode_name` keeps them distinct - the coarsening is the point.
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
    let call: SsaOp<MockTarget> = SsaOp::from(CallData::new(Some(v), Callee::Direct(0), vec![]));
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

    let call: SsaOp<MockTarget> = SsaOp::from(CallData::new(
        Some(SsaVarId::from_index(4)),
        Callee::Direct(0xAB),
        vec![],
    ));
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

/// **The environment's rounding mode an operation reads is a use.** A lane
/// division and a raise of it read their operands, then the mode; renaming
/// the mode renames it within, the stack effect counts it, and display names
/// it.
#[test]
fn the_environments_mode_is_a_use() {
    let (left, right, mode, dest, renamed) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
        SsaVarId::from_index(3),
        SsaVarId::from_index(4),
    );
    let divide: SsaOp<MockTarget> = SsaOp::VectorBinary {
        dest,
        left,
        right,
        kind: VectorBinaryKind::Div,
        element: VectorElement::float(64).as_scalar(),
        environment: Some(mode),
    };
    assert_eq!(divide.uses(), vec![left, right, mode]);
    assert_eq!(divide.stack_effect(), (3, 1));
    assert!(divide.to_string().ends_with(" mode=v2"), "{divide}");
    let raise: SsaOp<MockTarget> = SsaOp::FloatRaise {
        dest: SsaVarId::from_index(5),
        op: Box::new(divide.clone()),
        lanes: None,
        raises: FloatExceptions::IEEE,
        layout: FloatExceptionLayout::IEEE,
    };
    assert_eq!(raise.uses(), vec![left, right, mode]);
    let mut renaming = divide;
    assert_eq!(renaming.replace_uses(mode, renamed), 1);
    assert_eq!(renaming.environment(), Some(renamed));
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

    let op: SsaOp<MockTarget> = SsaOp::AtomicCmpXchg {
        old: d,
        success: None,
        addr: a,
        expected: e,
        desired: v,
        success_ordering: AtomicOrdering::SeqCst,
        failure_ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        weak: false,
        access: AccessQualifiers::FLAT,
    };
    assert_eq!(op.dest(), Some(d));
    assert_eq!(op.uses().len(), 3);

    let op2: SsaOp<MockTarget> = SsaOp::AtomicLockRmw {
        dest: d,
        addr: a,
        value: v,
        op: AtomicRmwOp::Xchg,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT,
    };
    assert_eq!(op2.dest(), Some(d));
    assert_eq!(op2.uses().len(), 2);
}

/// **A conditional atomic read-modify-write is one atomic access that defines
/// the old value.** x86 `cmp<cc>xadd`: the old value is its definition, the
/// address, the compared value and the addend its uses; it reads and writes
/// memory atomically at its ordering, and its condition, a relation or one
/// flag of the comparison, shows in its display.
#[test]
fn atomic_conditional_rmw_reports_defs_uses_effects_and_condition() {
    let [old, addr, compare, value] = [0, 1, 2, 3].map(SsaVarId::from_index);
    let op = |condition| SsaOp::<MockTarget>::AtomicCondRmw {
        old,
        addr,
        compare,
        value,
        op: AtomicRmwOp::Add,
        condition,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits64,
        access: AccessQualifiers::FLAT,
    };
    let below = op(CompareCondition::Relation {
        cmp: CmpKind::Lt,
        unsigned: true,
    });
    assert_eq!(below.dest(), Some(old));
    assert_eq!(below.uses(), vec![addr, compare, value]);
    let effects = below.effects();
    assert_eq!(effects.kind, SsaEffectKind::Atomic);
    assert_eq!(effects.ordering, Some(AtomicOrdering::SeqCst));
    let memory = below.memory_effect().expect("an atomic access");
    assert!(memory.reads && memory.writes);
    assert!(!below.is_pure());
    assert!(
        below.to_string().contains("atomicrmw.cond.add.ult"),
        "{below}"
    );
    let signed = op(CompareCondition::Relation {
        cmp: CmpKind::Le,
        unsigned: false,
    });
    assert!(
        signed.to_string().contains("atomicrmw.cond.add.le."),
        "{signed}"
    );
    let overflow = op(CompareCondition::Flag(FlagCondition::Overflow));
    assert!(
        overflow.to_string().contains("atomicrmw.cond.add.overflow"),
        "{overflow}"
    );

    let renamed = SsaVarId::from_index(9);
    let remapped = below.remap_variables(|var| (var == compare).then_some(renamed));
    assert_eq!(remapped.uses(), vec![addr, renamed, value]);
}

/// Native atomic compare-exchange and exchange report both output registers,
/// their operand uses, stack shape, and atomic memory semantics.
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
        access: AccessQualifiers::FLAT,
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
        access: AccessQualifiers::FLAT.made_volatile(),
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
        "v0 = atomicxchg.acqrel.i32 volatile v2, v4"
    );
}

/// Boolean `and`/`not` are pure, expose their operands, and survive remapping.
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

/// Wide multiply and divide report both of their output registers as defs.
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

/// Expanded vector memory ops report their operands, read/write effects, and
/// stack shape, including the multi-output faulting and segment forms.
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
        access: AccessQualifiers::FLAT,
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
        access: AccessQualifiers::FLAT,
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
        access: AccessQualifiers::FLAT,
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
        access: AccessQualifiers::FLAT,
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
        access: AccessQualifiers::FLAT,
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
        !SsaOp::<MockTarget>::AtomicCmpXchg {
            old: SsaVarId::from_index(0),
            success: None,
            addr: SsaVarId::from_index(1),
            expected: SsaVarId::from_index(2),
            desired: SsaVarId::from_index(3),
            success_ordering: AtomicOrdering::SeqCst,
            failure_ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            weak: false,
            access: AccessQualifiers::FLAT,
        }
        .is_pure()
    );
    assert!(
        !SsaOp::<MockTarget>::AtomicLockRmw {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            value: SsaVarId::from_index(2),
            op: AtomicRmwOp::Add,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            access: AccessQualifiers::FLAT,
        }
        .is_pure()
    );
}

/// InterruptReturn is a terminator.
#[test]
fn interrupt_return_is_terminator() {
    assert!(SsaOp::<MockTarget>::InterruptReturn.is_terminator());
}

/// A compare-exchange and a read-modify-write may throw.
#[test]
fn atomic_ops_may_throw() {
    assert!(
        SsaOp::<MockTarget>::AtomicCmpXchg {
            old: SsaVarId::from_index(0),
            success: None,
            addr: SsaVarId::from_index(1),
            expected: SsaVarId::from_index(2),
            desired: SsaVarId::from_index(3),
            success_ordering: AtomicOrdering::SeqCst,
            failure_ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            weak: false,
            access: AccessQualifiers::FLAT,
        }
        .may_throw()
    );
    assert!(
        SsaOp::<MockTarget>::AtomicLockRmw {
            dest: SsaVarId::from_index(0),
            addr: SsaVarId::from_index(1),
            value: SsaVarId::from_index(2),
            op: AtomicRmwOp::Xchg,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            access: AccessQualifiers::FLAT,
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

    // AtomicCmpXchg: pop 3, push 1
    assert_eq!(
        SsaOp::<MockTarget>::AtomicCmpXchg {
            old: d,
            success: None,
            addr: v,
            expected: a,
            desired: d,
            success_ordering: AtomicOrdering::SeqCst,
            failure_ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            weak: false,
            access: AccessQualifiers::FLAT,
        }
        .stack_effect(),
        (3, 1)
    );

    // AtomicLockRmw: pop 2, push 1
    assert_eq!(
        SsaOp::<MockTarget>::AtomicLockRmw {
            dest: d,
            addr: v,
            value: a,
            op: AtomicRmwOp::Add,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            access: AccessQualifiers::FLAT,
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

    // Test AtomicCmpXchg
    let mut op4: SsaOp<MockTarget> = SsaOp::AtomicCmpXchg {
        old: d,
        success: None,
        addr: old,
        expected: other,
        desired: old,
        success_ordering: AtomicOrdering::SeqCst,
        failure_ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        weak: false,
        access: AccessQualifiers::FLAT,
    };
    assert_eq!(op4.replace_uses(old, new), 2);

    // Test AtomicLockRmw
    let mut op5: SsaOp<MockTarget> = SsaOp::AtomicLockRmw {
        dest: d,
        addr: old,
        value: other,
        op: AtomicRmwOp::Xor,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT,
    };
    assert_eq!(op5.replace_uses(old, new), 1);
}

/// remap_variables works for new ops.
#[test]
fn remap_variables_new_ops() {
    let d0 = SsaVarId::from_index(0);
    let d99 = SsaVarId::from_index(99);
    let v1 = SsaVarId::from_index(1);
    let v55 = SsaVarId::from_index(55);
    let a2 = SsaVarId::from_index(2);

    let mut map = FxHashMap::default();
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
    assert_eq!(format!("{}", FenceKind::Loads), "loads");
    assert_eq!(format!("{}", FenceKind::Stores), "stores");
}

/// A fence of loads orders as an acquire does and one of stores as a release:
/// the effect summary a pass reads keeps a later access above neither.
#[test]
fn a_fence_of_loads_or_stores_orders_as_an_acquire_or_a_release() {
    assert_eq!(FenceKind::Loads.ordering(), AtomicOrdering::Acquire);
    assert_eq!(FenceKind::Stores.ordering(), AtomicOrdering::Release);
    let effects = SsaOp::<MockTarget>::Fence {
        kind: FenceKind::Stores,
    }
    .effects();
    assert_eq!(effects.ordering, Some(AtomicOrdering::Release));
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
    assert_eq!(format!("{}", AtomicRmwOp::AndNot), "andnot");
    assert_eq!(format!("{}", AtomicRmwOp::MinU), "minu");
    assert_eq!(format!("{}", AtomicRmwOp::MaxU), "maxu");
    assert_eq!(format!("{}", AtomicRmwOp::Neg), "neg");
    assert_eq!(format!("{}", AtomicRmwOp::FAdd), "fadd");
    assert_eq!(format!("{}", AtomicRmwOp::FMaxNum), "fmaxnm");
}

/// `SsaOp::opcode_name` is one flat namespace, so two system-op identities
/// sharing a `kind_str` are two instructions nothing downstream can tell apart.
///
/// The shapes that break it: a variant naming a register *file* standing beside
/// one naming an instruction, so both answer `"system.sysreg.read"`; and a
/// payload field read as data rather than identity, so `rdtsc` and `rdtscp`
/// answer `"system.timestamp"` between them.
#[test]
fn system_op_kind_str_is_injective_over_every_identity() {
    assert_kind_strs_are_injective(
        "SystemOpKind",
        SystemOpKind::identities()
            .into_iter()
            .map(|kind| (kind.kind_str(), format!("{kind:?}"))),
    );
}

/// The same invariant for the kernels of a whole value, whose `PointerAuth`
/// payload is four instructions - signing, authenticating, stripping, generic
/// MAC - that one family-wide spelling would collapse onto a single key.
#[test]
fn kernel_kind_str_is_injective_over_every_identity() {
    assert_kind_strs_are_injective(
        "Kernel",
        Kernel::identities()
            .into_iter()
            .map(|kind| (kind.kind_str(), format!("{kind:?}"))),
    );
}

/// **Every kernel of a whole value has an identity.** One representative of
/// each variant, built through a match that names every variant, is among
/// [`Kernel::identities`] or runs only under lane control, so a kernel added
/// without an identity fails here rather than escaping the injectivity check.
#[test]
fn every_whole_value_kernel_is_an_identity() {
    let whole = |kernel: Kernel| -> Option<Kernel> {
        match kernel {
            Kernel::Move
            | Kernel::Unary(_)
            | Kernel::Binary(_)
            | Kernel::IntegerDivide { .. }
            | Kernel::ModuloShift { .. }
            | Kernel::MultiplyHigh { .. }
            | Kernel::Compare { .. }
            | Kernel::Cast { .. }
            | Kernel::Extend { .. }
            | Kernel::WidenBinary { .. }
            | Kernel::FusedMultiplyAdd { .. }
            | Kernel::IntegerMultiplyAdd { .. }
            | Kernel::Reduce { .. }
            | Kernel::Gather
            | Kernel::Slide { .. }
            | Kernel::Index
            | Kernel::Memory { .. }
            | Kernel::InsertScalar
            | Kernel::NarrowShift { .. }
            | Kernel::Fixed { .. }
            | Kernel::WidenMultiplyAdd { .. }
            | Kernel::MaskBinary(_)
            | Kernel::IntegerCarry { .. }
            | Kernel::Select
            | Kernel::MaskPrefix(_)
            | Kernel::MaskIota
            | Kernel::MaskSummary { .. }
            | Kernel::ReduceWide { .. }
            | Kernel::Compress
            | Kernel::SignInject { .. }
            | Kernel::FloatClassify
            | Kernel::FloatToInteger { .. }
            | Kernel::Estimate7 { .. } => None,
            Kernel::CarrylessMultiply { .. }
            | Kernel::Crypto(_)
            | Kernel::BitDeposit
            | Kernel::BitExtract
            | Kernel::BitGroup
            | Kernel::Checksum(_)
            | Kernel::Random { .. }
            | Kernel::PointerAuth(_)
            | Kernel::ScalarCipher(_)
            | Kernel::FloatToIntModular
            | Kernel::MemoryTag(_)
            | Kernel::VectorShapeSupported
            | Kernel::VectorLengthChoice
            | Kernel::VectorFloatSupported
            | Kernel::VectorCryptoSupported
            | Kernel::DecimalAdjust { .. }
            | Kernel::FloatHelper(_)
            | Kernel::TernaryLogic { .. }
            | Kernel::MultiSad { .. }
            | Kernel::DoubleBlockSad { .. }
            | Kernel::StringCompare { .. }
            | Kernel::Bitfield { .. }
            | Kernel::Intersect
            | Kernel::ShuffleBits
            | Kernel::HorizontalMinPos
            | Kernel::CounterPredicate { .. }
            | Kernel::PredicateCounter { .. }
            | Kernel::FloatControlled { .. } => Some(kernel),
        }
    };
    let identities: BTreeSet<String> = Kernel::identities()
        .into_iter()
        .map(|kernel| format!("{:?}", discriminant(&kernel)))
        .collect();
    for kernel in [
        Kernel::CarrylessMultiply {
            part: CarrylessPart::Low,
        },
        Kernel::Crypto(VectorCryptoKind::AesEncrypt),
        Kernel::BitDeposit,
        Kernel::BitExtract,
        Kernel::BitGroup,
        Kernel::Checksum(CrcPolynomial::Ieee),
        Kernel::Random {
            from_entropy: false,
        },
        Kernel::PointerAuth(PacKind::Sign),
        Kernel::ScalarCipher(ScalarCipherKind::Aes32EncryptByte),
        Kernel::FloatToIntModular,
        Kernel::MemoryTag(MemoryTagKind::Add),
        Kernel::VectorShapeSupported,
        Kernel::VectorLengthChoice,
        Kernel::VectorFloatSupported,
        Kernel::VectorCryptoSupported,
        Kernel::DecimalAdjust {
            kind: BcdAdjustKind::DecimalAddAdjust,
            base: 10,
        },
        Kernel::FloatHelper(FpHelperKind::ExpAccelerate),
        Kernel::TernaryLogic { table: 0 },
        Kernel::MultiSad { offsets: 0 },
        Kernel::DoubleBlockSad { selector: 0 },
        Kernel::StringCompare {
            control: 0,
            explicit_length: false,
            result_index: false,
        },
        Kernel::Bitfield {
            insert: false,
            position: BitfieldPosition::Register,
        },
        Kernel::Intersect,
        Kernel::ShuffleBits,
        Kernel::HorizontalMinPos,
        Kernel::CounterPredicate { part: 0 },
        Kernel::PredicateCounter { element_bits: 8 },
        Kernel::FloatControlled {
            kind: ControlledKind::RoundScale,
            control: 0,
        },
    ]
    .into_iter()
    .filter_map(whole)
    {
        assert!(
            identities.contains(&format!("{:?}", discriminant(&kernel))),
            "{kernel:?} has no identity"
        );
    }
}

/// Fails with every colliding key and the identities that produce it, so one
/// run names the whole repair rather than the first of it.
fn assert_kind_strs_are_injective(
    label: &str,
    identities: impl Iterator<Item = (&'static str, String)>,
) {
    let mut seen: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for (key, described) in identities {
        seen.entry(key).or_default().push(described);
    }
    let collisions: Vec<String> = seen
        .iter()
        .filter(|(_, kinds)| kinds.len() > 1)
        .map(|(key, kinds)| format!("{key} <- {}", kinds.join(" | ")))
        .collect();
    assert!(
        collisions.is_empty(),
        "{} kind_str collision(s) across {label} identities:\n{}",
        collisions.len(),
        collisions.join("\n"),
    );
}

/// `xend`, `xtest`, `xabort` and the load-tracking pair live in exactly one
/// enum.
///
/// Two homes for one instruction means two effect summaries for it: a commit is
/// a sequentially consistent fence under one spelling and a plain `Write` under
/// the other, so whether a store may move across `xend` would depend on which
/// spelling a front-end picks.
#[test]
fn transactional_memory_has_one_home() {
    for op in BarrierOp::all() {
        assert!(
            !op.kind_str().contains("txn") && !op.mnemonic().starts_with('x'),
            "{op:?} is a transactional operation sitting in the barrier family"
        );
    }

    assert_eq!(
        SystemTransactionKind::Commit.effects().kind,
        SsaEffectKind::Fence,
        "a commit publishes the transaction's writes"
    );
    assert_eq!(
        SystemTransactionKind::Cancel.effects().kind,
        SsaEffectKind::Fence,
    );
    assert_eq!(
        SystemTransactionKind::Test.effects().kind,
        SsaEffectKind::Read,
        "`xtest` reads transactional status and touches no memory"
    );
    for op in [
        SystemTransactionKind::Start,
        SystemTransactionKind::ResumeLoadTracking,
        SystemTransactionKind::SuspendLoadTracking,
    ] {
        assert_eq!(
            op.effects().kind,
            SsaEffectKind::Write,
            "{op:?} writes state"
        );
    }

    // The carrier reports what the payload reports, so there is one answer.
    for op in SystemTransactionKind::all() {
        assert_eq!(
            SystemOpKind::Transaction(op).effects(),
            op.effects(),
            "{op:?} must not gain a second effect summary through its carrier"
        );
    }
}

/// A special-register access reports what the *instruction* does, not what its
/// operand shape suggests.
///
/// `writes_destination_operand` answers a lift-shape question and cannot stand
/// in for this one: RISC-V `csrrw`/`csrrs`/`csrrc` define a destination *and*
/// write the CSR, and ARM `ldc` defines no destination operand while only
/// reading.
#[test]
fn sysreg_effects_follow_the_operation() {
    for op in [
        SysRegOp::ExchangeSystemRegister,
        SysRegOp::SetSystemRegisterBits,
        SysRegOp::ClearSystemRegisterBits,
    ] {
        assert_eq!(
            op.effects().kind,
            SsaEffectKind::ReadWrite,
            "{op:?} reads the register and writes it back in one instruction"
        );
    }
    assert_eq!(
        SysRegOp::CoprocessorLoad.effects().kind,
        SsaEffectKind::Read,
        "`ldc` loads a coprocessor register rather than writing machine state"
    );
    assert_eq!(
        SysRegOp::ReadModelSpecific.effects().kind,
        SsaEffectKind::RuntimeDefined,
        "`rdmsr` reads the machine, not memory"
    );
    assert_eq!(
        SysRegOp::WriteModelSpecific.effects().kind,
        SsaEffectKind::Write
    );

    // No access is pure: every one of them touches a register file the IR does
    // not model as SSA values.
    for op in SysRegOp::all() {
        assert!(!op.effects().is_pure(), "{op:?} must not be pure");
        assert_eq!(
            SystemOpKind::ControlRegister(op).effects(),
            op.effects(),
            "{op:?} must not gain a second effect summary through its carrier"
        );
    }

    // `rdmsr` and the ARM/MIPS register move are separate operations with
    // separate keys, so neither collides with the other.
    assert_eq!(
        SysRegOp::ReadModelSpecific.kind_str(),
        "system.sysreg.msr.read"
    );
    assert_eq!(
        SysRegOp::ReadSystemRegister.kind_str(),
        "system.sysreg.read"
    );
}

/// **A floating-point state image is the memory it moves.** A save writes
/// the image at the address its first input is and a restore reads it, as
/// many bytes as the layout holds (`fxsave`'s 512, the 16-bit environment's
/// 14) or an extent no layout fixes (`xsave`'s).
#[test]
fn a_floating_point_state_image_is_the_memory_it_moves() {
    let image = |kind| {
        SsaOp::<MockTarget>::FpuControl(Box::new(KindedVecData {
            kind,
            outputs: Vec::new(),
            inputs: vec![SsaVarId::from_index(0)],
        }))
    };
    for (kind, effect, bits) in [
        (
            FpuControlKind::SaveExtended,
            SsaEffectKind::Write,
            Some(4096),
        ),
        (
            FpuControlKind::RestoreExtended,
            SsaEffectKind::Read,
            Some(4096),
        ),
        (
            FpuControlKind::LoadEnvironmentShort,
            SsaEffectKind::Read,
            Some(112),
        ),
        (
            FpuControlKind::SaveExtendedMasked,
            SsaEffectKind::Write,
            None,
        ),
    ] {
        let op = image(kind);
        assert_eq!(op.effects().kind, effect, "{kind:?}");
        assert_eq!(
            op.effects().memory,
            MemoryEffectLocation::Unknown,
            "{kind:?}"
        );
        let mut accesses = Vec::new();
        op.for_each_addressed_access(|access| accesses.push(access));
        assert_eq!(accesses.len(), 1, "{kind:?}");
        assert_eq!(
            (accesses[0].addr, accesses[0].width_bits, accesses[0].writes),
            (
                SsaVarId::from_index(0),
                bits,
                effect == SsaEffectKind::Write
            ),
            "{kind:?}"
        );
    }
}

/// A hint has a carrier, an identity, and an effect that is never `Pure` - and
/// never a memory barrier either.
///
/// `Pure` would let dead-code elimination delete exactly the hints that matter:
/// `endbr64` is a control-flow-integrity landing pad and `pause`/`wfe` are
/// synchronization. Plain `Opaque` carries `MemoryEffectLocation::Unknown`,
/// which `MemoryLocator::classify` turns into a
/// `MemoryOp::Barrier` - making every lifted `nop` block store forwarding
/// across instruction padding. Non-removable *and* not a barrier is the only
/// correct answer, and the location is what pins it.
#[test]
fn hint_ops_are_carried_and_never_pure() {
    for hint in HintOp::all() {
        let effects = hint.effects();
        assert!(
            !effects.is_pure(),
            "{hint:?} must survive dead-code removal"
        );
        assert_eq!(
            SystemOpKind::Hint(hint).effects(),
            effects,
            "{hint:?} must not gain a second effect summary through its carrier"
        );
        assert!(
            hint.kind_str().starts_with("system.hint."),
            "{hint:?} must be spelled under its carrier's prefix, not `hint.*`"
        );

        match effects.kind {
            SsaEffectKind::Read => assert!(
                hint.mnemonic().starts_with("prefetch"),
                "{hint:?} reads memory but is not a prefetch"
            ),
            SsaEffectKind::Opaque => assert_eq!(
                effects.memory,
                MemoryEffectLocation::None,
                "{hint:?} touches no memory, so it must not classify as a barrier"
            ),
            other => panic!("{hint:?} declares an unexpected effect class {other:?}"),
        }
    }

    // The carrier is the point: it is what puts a hint's identity into an
    // operation the IR can hold.
    let op: SsaOp<MockTarget> = SsaOp::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::Hint(HintOp::IndirectBranchLandingPad),
        mnemonic: String::from("endbr64"),
        metadata: None,
        outputs: Vec::new(),
        inputs: Vec::new(),
    }));
    assert_eq!(op.opcode_name(), "system.hint.landingpad");
    assert!(!op.is_pure());
}

/// The `_ =>` catch-all in `MachineStateOp::effects` must fall to a
/// conservative class, never to `Pure`.
///
/// A machine-state operation that classifies pure and defines nothing is one
/// dead-code elimination deletes. The catch-all is deliberate - 140 variants -
/// so what makes it safe is that its default over-approximates, which is what
/// this pins.
#[test]
fn no_machine_state_op_is_pure() {
    for op in MachineStateOp::all() {
        assert!(
            !op.effects().is_pure(),
            "{op:?} classifies pure, so an operation with no SSA result is deletable"
        );
        assert_eq!(
            SystemOpKind::MachineState(op).effects(),
            op.effects(),
            "{op:?} must not gain a second effect summary through its carrier"
        );
    }
}

/// A query of the machine reads no memory: what it yields is the value the
/// processor supplies where control reaches it.
///
/// `cpuid`, the counters, a system register read, a segment base or
/// protection-key read and a hardware random draw are each
/// [`SsaEffectKind::RuntimeDefined`]: deletable when nothing reads the result,
/// never moved or shared, and no reader of memory, so a frame store before
/// one is not kept alive by it. The operations that do sample memory (a
/// descriptor-table query, a far-pointer or coprocessor load, the reverse-map
/// query) stay reads, a system instruction returning a result may also write
/// (a guarded control stack pop), and `slwpcb` writes the control block it
/// flushes.
#[test]
fn a_machine_query_reads_no_memory() {
    let queries = [
        SystemOpKind::CpuId,
        SystemOpKind::Timestamp { aux: false },
        SystemOpKind::Timestamp { aux: true },
        SystemOpKind::ReadPerfCounter,
    ]
    .into_iter()
    .chain(
        [
            SysRegOp::ReadSystemRegister,
            SysRegOp::ReadSystemRegisterPair,
            SysRegOp::ReadModelSpecific,
            SysRegOp::ReadExtendedControl,
            SysRegOp::ReadControlRegister,
            SysRegOp::ReadDebugRegister,
        ]
        .map(SystemOpKind::ControlRegister),
    )
    .chain(
        [
            MachineStateOp::ReadFsBase,
            MachineStateOp::ReadGsBase,
            MachineStateOp::ShadowStackReadDword,
            MachineStateOp::ShadowStackReadQword,
            MachineStateOp::UserInterruptTest,
            MachineStateOp::ReadProtectionKeyRights,
            MachineStateOp::ReadProcessorId,
            MachineStateOp::ReadProcessorPower,
            MachineStateOp::VendorCpuRead,
            MachineStateOp::VendorReadModelRegister,
            MachineStateOp::VendorReadShadowRegister,
            MachineStateOp::VendorReadDebugRegister,
            MachineStateOp::TestTarget,
            MachineStateOp::TestTargetAlternate,
            MachineStateOp::TestTargetAlternateUnprivileged,
            MachineStateOp::TestTargetUnprivileged,
        ]
        .map(SystemOpKind::MachineState),
    )
    .map(|kind| (format!("{kind:?}"), kind.effects()))
    .chain(
        [
            Kernel::Random {
                from_entropy: false,
            },
            Kernel::Random { from_entropy: true },
            Kernel::MemoryTag(MemoryTagKind::Random),
        ]
        .map(|kernel| {
            let computation = Computation::whole(
                kernel,
                VectorElement::integer(64, Some(false)),
                VectorRounding::None,
            );
            (format!("{kernel:?}"), computation.effects())
        }),
    );
    for (name, effects) in queries {
        assert_eq!(effects.kind, SsaEffectKind::RuntimeDefined, "{name}");
        assert!(
            !effects.reads_memory() && !effects.writes_memory(),
            "{name} touches memory"
        );
        assert!(!effects.is_pure(), "{name} is movable");
        assert!(effects.removable_when_unused(), "{name} is not deletable");
    }
    let samplers = [
        SystemOpKind::ControlRegister(SysRegOp::CoprocessorLoad),
        SystemOpKind::MachineState(MachineStateOp::ReadAccessRights),
        SystemOpKind::MachineState(MachineStateOp::ReadSegmentLimit),
        SystemOpKind::MachineState(MachineStateOp::VerifySegmentReadable),
        SystemOpKind::MachineState(MachineStateOp::VerifySegmentWritable),
        SystemOpKind::MachineState(MachineStateOp::LoadFarPointerDs),
        SystemOpKind::MachineState(MachineStateOp::LoadFarPointerSs),
        SystemOpKind::MachineState(MachineStateOp::ReverseMapQuery),
    ];
    for kind in samplers {
        assert_eq!(kind.effects().kind, SsaEffectKind::Read, "{kind:?}");
    }
    assert_eq!(
        SystemOpKind::MachineState(MachineStateOp::SystemInstructionRead)
            .effects()
            .kind,
        SsaEffectKind::ReadWrite
    );
    assert_eq!(
        SystemOpKind::MachineState(MachineStateOp::ProfilingStoreControlBlock)
            .effects()
            .kind,
        SsaEffectKind::Write
    );
}

/// Every flag a flag adjustment writes is a bit [`FlagsMask`] defines.
#[test]
fn flag_adjust_writes_named_flags() {
    for kind in every::<FlagAdjustKind>() {
        assert!(
            kind.flags_written().is_valid(),
            "{kind:?} names a flag bit FlagsMask does not define"
        );
    }
}

/// `setf8`/`setf16` exist to recover a narrow add's overflow: they write `N`,
/// `Z` and `V` and leave `C`, and naming only `N` and `Z` would tell a caller
/// a stale `V` is still live.
#[test]
fn a_narrow_flag_set_writes_the_overflow() {
    for kind in [FlagAdjustKind::SetNzFrom8, FlagAdjustKind::SetNzFrom16] {
        let written = kind.flags_written();
        assert!(written.contains(FlagsMask::OVERFLOW), "{kind:?} writes V");
        assert!(written.contains(FlagsMask::SIGN), "{kind:?} writes N");
        assert!(written.contains(FlagsMask::ZERO), "{kind:?} writes Z");
        assert!(
            !written.contains(FlagsMask::CARRY),
            "{kind:?} leaves C alone"
        );
    }
}

/// A flag constant with no producer is an unreachable path: the constant, the
/// `NativeFlagBit` variant and the `from_flag_bit` arm all exist, and nothing
/// can build a mask that names IF. `cli`/`sti` are what make it reachable.
#[test]
fn interrupt_flag_has_a_producer() {
    let producers: Vec<FlagAdjustKind> = every::<FlagAdjustKind>()
        .into_iter()
        .filter(|kind| kind.flags_written().intersects(FlagsMask::INTERRUPT))
        .collect();
    assert_eq!(
        producers,
        vec![FlagAdjustKind::ClearInterrupt, FlagAdjustKind::SetInterrupt],
        "IF is an EFLAGS bit exactly as DF and AC are, so `cli`/`sti` write it here"
    );
    assert_eq!(
        FlagsMask::from_flag_bit(NativeFlagBit::Interrupt),
        FlagsMask::INTERRUPT
    );
    assert!(!FlagsMask::x86_status().contains(FlagsMask::INTERRUPT));
}

/// Named `FlagsMask` bits are distinct, sit at their pinned positions, and compose via union.
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

/// `FlagsMask` renders its set bits by name, comma-joined, and the empty mask as `none`.
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

/// The rendering is the table, not a second copy of it.
///
/// A per-bit `if` chain beside the constants is two statements of the same
/// list, and the one a reviewer does not re-read is the one that drifts: a bit
/// added to the constants and not to the renderer prints as nothing at all.
#[test]
fn flags_mask_display_matches_the_named_table() {
    for (bit, name) in FlagsMask::NAMED {
        assert_eq!(&format!("{bit}"), name, "{name} must render as itself");
        assert!(
            FlagsMask::ALL.contains(*bit),
            "{name} must be part of the derived ALL"
        );
        assert!(bit.is_valid());
    }

    assert_eq!(
        FlagsMask::NAMED.len(),
        usize::try_from(FlagsMask::ALL.bits().count_ones()).unwrap_or_default(),
        "every named entry must contribute exactly one distinct bit to ALL"
    );

    // Order is CF,PF,AF,ZF,SF,OF,DF,IF,AC - bit order, so the rendering of a
    // multi-bit mask does not depend on which bits happen to be set.
    assert_eq!(format!("{}", FlagsMask::ALL), "CF,PF,AF,ZF,SF,OF,DF,IF,AC");
    assert_eq!(format!("{}", FlagsMask::x86_status()), "CF,PF,AF,ZF,SF,OF");
}

/// An undefined bit survives the mask and is reported, in both directions.
///
/// `from_bits` is unmasked so a persisted mask round-trips exactly; masking
/// inside it would turn a diagnosable error into silent data loss. What makes
/// that safe is that the bit is then visible everywhere it matters:
/// [`FlagsMask::is_valid`] is false, [`FlagsMask::from_bits_checked`] refuses
/// it, and the rendering appends `?0x…` instead of printing a mask that looks
/// well-formed.
#[test]
fn flags_mask_round_trips_undefined_bits_and_reports_them_invalid() {
    let undefined_bit = 1u16 << 15;
    assert_eq!(FlagsMask::ALL.bits() & undefined_bit, 0);

    let mask = FlagsMask::from_bits(FlagsMask::CARRY.bits() | undefined_bit);
    assert_eq!(
        mask.bits(),
        FlagsMask::CARRY.bits() | undefined_bit,
        "from_bits must not drop the bit it was handed"
    );
    assert!(!mask.is_valid());
    assert_eq!(mask.undefined_bits(), undefined_bit);
    assert_eq!(format!("{mask}"), "CF,?0x8000");

    // A mask of nothing but undefined bits must not read as `none`.
    let only_undefined = FlagsMask::from_bits(undefined_bit);
    assert_eq!(format!("{only_undefined}"), "?0x8000");

    assert_eq!(FlagsMask::from_bits_checked(undefined_bit), None);
    assert_eq!(
        FlagsMask::from_bits_checked(FlagsMask::ALL.bits()),
        Some(FlagsMask::ALL)
    );
    // The empty mask selects no flags, which is a legal thing to ask for.
    assert_eq!(
        FlagsMask::from_bits_checked(0),
        Some(FlagsMask::from_bits(0))
    );
    assert!(FlagsMask::from_bits(0).is_valid());
}

/// Each `FlagCondition` renders as its snake_case mnemonic.
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

/// Complementary and unrelated `FlagCondition` variants compare unequal.
#[test]
fn flag_condition_variants_are_distinct() {
    assert_ne!(FlagCondition::Carry, FlagCondition::Zero);
    assert_ne!(FlagCondition::Overflow, FlagCondition::NotOverflow);
    assert_ne!(FlagCondition::Negative, FlagCondition::Positive);
    assert_ne!(FlagCondition::ParityEven, FlagCondition::ParityOdd);
}

/// Each `FlagCondition` reports the flag bit its test consults.
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

/// Each `FlagProducerSemantics` reports the flags it defines and its undefined and
/// cleared writes per ISA family.
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

/// `flags_dest` returns the flags output on arithmetic ops carrying a `flags` operand.
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

/// `flags_dest` is `None` when a flag-capable op carries no `flags` operand.
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

/// `flags_dest` is `None` for ops that never write flags.
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

    let call: SsaOp<MockTarget> = SsaOp::from(CallData::new(Some(d), Callee::Direct(0), vec![]));
    assert_eq!(call.flags_dest(), None);
}

/// `ReadFlags` reports its dest and flags operand and classifies as a pure,
/// non-terminating read.
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

/// `ReadFlags` pops its flags operand and pushes one result.
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

/// `ReadFlags` rewrites its flags operand under `replace_uses`.
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

/// `ReadFlags` remaps both its dest and its flags operand.
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

/// `ReadFlags` renders its dest, flags operand, and mask.
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

/// `BranchFlags` is a non-pure terminator that uses its flags operand and has two
/// successors.
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

/// `BranchFlags` pops its flags operand and pushes nothing.
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

/// `BranchFlags::redirect_target` rewrites a matching successor and no-ops otherwise.
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

/// `BranchFlags` remaps both branch targets.
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

/// `BranchFlags` rewrites its flags operand under `replace_uses`.
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

/// `BranchFlags` renders its flags operand, condition, and both targets.
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

/// `Unreachable` is a non-pure terminator with no dest, uses, or successors.
#[test]
fn unreachable_is_terminator_with_no_successors() {
    assert!(SsaOp::<MockTarget>::Unreachable.is_terminator());
    assert!(!SsaOp::<MockTarget>::Unreachable.is_pure());
    assert!(!SsaOp::<MockTarget>::Unreachable.may_throw());
    assert_eq!(SsaOp::<MockTarget>::Unreachable.dest(), None);
    assert!(SsaOp::<MockTarget>::Unreachable.uses().is_empty());
    assert!(SsaOp::<MockTarget>::Unreachable.successors().is_empty());
}

/// `Unreachable` neither pops nor pushes.
#[test]
fn unreachable_stack_effect() {
    assert_eq!(SsaOp::<MockTarget>::Unreachable.stack_effect(), (0, 0));
}

/// `Unreachable` renders as `unreachable`.
#[test]
fn unreachable_display() {
    assert_eq!(
        format!("{}", SsaOp::<MockTarget>::Unreachable),
        "unreachable"
    );
}

/// Remapping `Unreachable` visits no variables and leaves it unchanged.
#[test]
fn unreachable_no_variable_remap() {
    let op = SsaOp::<MockTarget>::Unreachable;
    let remapped = op.remap_variables(|_| unreachable!());
    assert_eq!(remapped, SsaOp::Unreachable);
}

/// An arithmetic op carrying a `flags` operand reports both its value dest and its
/// flags dest.
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

/// Remapping a flag-setting op rewrites both its value dest and its flags dest.
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

/// A flag-setting op appends `flags=` only when it carries a flags operand.
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

/// `effects()` classifies pure, read, write, atomic, fence, first-fault, and call
/// ops, keeping hidden-machine-state ops impure and undeletable.
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
        access: AccessQualifiers::FLAT,
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
        access: AccessQualifiers::FLAT,
    }
    .effects();
    assert_eq!(store.kind, SsaEffectKind::Write);
    assert!(store.writes_memory());

    let atomic = SsaOp::<MockTarget>::AtomicLockRmw {
        dest: d,
        addr: a,
        value: b,
        op: AtomicRmwOp::Add,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT,
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

    // `dsb`/`dmb`/`isb` order memory: they are fences, not plain writes.
    let barrier = SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::Barrier(BarrierOp::Serialize),
        mnemonic: "dmb".into(),
        metadata: None,
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
        SsaOp::<MockTarget>::from(CallData::new(Some(d), Callee::Direct(1), vec![a]))
            .effects()
            .kind,
        SsaEffectKind::Call
    );
}

/// A [`SystemOpKind`] is always carried by [`SsaOp::SystemOp`], which is
/// **not** a structural terminator ([`SsaOp::is_terminator`] excludes it).
/// The `check_native_effects` verifier invariant rejects any op whose
/// `effects().control` is block-ending (`Terminator`/`Return`/`Throw`)
/// unless it is a terminator - so **no** system-op kind may declare a
/// block-ending control effect. `iret`/`eret`
/// (`SystemOpKind::InterruptReturn`) is the sharp case: declaring
/// `ControlEffect::Return` for it makes every mid-block `iret` fail lift SSA
/// verification, so it declares `ControlEffect::Call` - a control transfer to
/// the interrupted context, exactly like `sysret`.
///
/// Quantified over [`SystemOpKind::identities`] rather than over a hand-written
/// list, because a hand-written list silently omits whole families - every
/// special-register move, say - and omits each family added after it.
#[test]
fn system_op_kinds_never_declare_a_block_ending_control_effect() {
    for kind in SystemOpKind::identities() {
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
        SystemOpKind::InterruptReturn(InterruptReturnOp::ExceptionReturn)
            .effects()
            .control,
        ControlEffect::Call,
    );
    assert!(
        !SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::InterruptReturn(InterruptReturnOp::ExceptionReturn),
            mnemonic: String::from("iret"),
            metadata: None,
            outputs: Vec::new(),
            inputs: Vec::new(),
        }))
        .is_terminator()
    );
}

/// `class()` groups ops into scalar, flags, vector, atomic, control, and
/// native-opaque buckets.
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
            environment: None,
        }
        .class(),
        SsaOpClass::Vector
    );
    assert_eq!(
        SsaOp::<MockTarget>::AtomicLockRmw {
            dest: d,
            addr: a,
            value: b,
            op: AtomicRmwOp::Add,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            access: AccessQualifiers::FLAT,
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
}

/// Builds a battery of representative ops covering every visitor arm shape,
/// each paired with the number of definitions and uses it is expected to have.
///
/// The counts are written out here by hand, so they are an account of the
/// operands independent of any walk over the op. `visit_operands`, `defs()` and
/// `uses()` are all checked against them, which catches an operand nobody
/// visits as well as one visited in the wrong role -- omission is invisible to
/// a check that only compares three readers with each other.
fn visitor_battery() -> Vec<(SsaOp<MockTarget>, usize, usize)> {
    let v: Vec<SsaVarId> = (0..8).map(SsaVarId::from_index).collect();
    vec![
        // A soft compare-and-swap reads both masks and the compared value; a
        // hard clear neither the soft mask nor a compared value.
        (
            SsaOp::AtomicCheckedRmw {
                old: v[0],
                status: v[1],
                addr: v[2],
                compare: Some(v[3]),
                value: v[4],
                op: AtomicRmwOp::Xchg,
                rcw_mask: v[5],
                rcws_mask: Some(v[6]),
                protection: RcwProtection::Regime,
                ordering: AtomicOrdering::AcqRel,
                width: AtomicAccessWidth::Bits64,
                access: AccessQualifiers::FLAT,
            },
            2,
            5,
        ),
        (
            SsaOp::AtomicCheckedRmw {
                old: v[0],
                status: v[1],
                addr: v[2],
                compare: None,
                value: v[4],
                op: AtomicRmwOp::AndNot,
                rcw_mask: v[5],
                rcws_mask: None,
                protection: RcwProtection::Enabled,
                ordering: AtomicOrdering::Relaxed,
                width: AtomicAccessWidth::Bits128,
                access: AccessQualifiers::FLAT,
            },
            2,
            3,
        ),
        (
            SsaOp::Const {
                dest: v[0],
                value: ConstValue::I32(7),
            },
            1,
            0,
        ),
        (
            SsaOp::MergeFlags {
                dest: v[0],
                base: v[1],
                update: v[2],
                mask: FlagsMask::ZERO.union(FlagsMask::SIGN),
            },
            1,
            2,
        ),
        (
            SsaOp::InsertFlag {
                dest: v[0],
                base: v[1],
                bit: v[2],
                mask: FlagsMask::CARRY,
            },
            1,
            2,
        ),
        (
            SsaOp::float_raise(
                v[0],
                SsaOp::VectorTernary {
                    dest: v[4],
                    first: v[1],
                    second: v[2],
                    third: v[3],
                    kind: VectorTernaryKind::Fma,
                    element: VectorElement::default(),
                    environment: None,
                },
                None,
                FloatExceptions::IEEE,
                FloatExceptionLayout::IEEE,
            ),
            1,
            3,
        ),
        (
            SsaOp::Add {
                dest: v[0],
                left: v[1],
                right: v[2],
                flags: Some(v[3]),
            },
            2,
            2,
        ),
        (
            SsaOp::Neg {
                dest: v[0],
                operand: v[1],
                flags: None,
            },
            1,
            1,
        ),
        (
            SsaOp::Shr {
                dest: v[0],
                value: v[1],
                amount: v[2],
                unsigned: true,
                flags: Some(v[3]),
            },
            2,
            2,
        ),
        (
            SsaOp::WideMul {
                low: v[0],
                high: v[1],
                left: v[2],
                right: v[3],
                unsigned: false,
            },
            2,
            2,
        ),
        (
            SsaOp::WideDiv {
                quotient: v[0],
                remainder: v[1],
                high: v[2],
                low: v[3],
                divisor: v[4],
                unsigned: false,
            },
            2,
            3,
        ),
        (
            SsaOp::FloatCompareFlags {
                flags: v[0],
                left: v[1],
                right: v[2],
                signaling: false,
            },
            1,
            2,
        ),
        (
            SsaOp::Select {
                dest: v[0],
                condition: v[1],
                true_val: v[2],
                false_val: v[3],
            },
            1,
            3,
        ),
        (
            SsaOp::StoreIndirect {
                addr: v[0],
                value: v[1],
                value_type: MockType::I32,
                access: AccessQualifiers::FLAT,
            },
            0,
            2,
        ),
        (
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
                access: AccessQualifiers::FLAT,
            },
            2,
            3,
        ),
        (
            SsaOp::AtomicPairLoad {
                first: v[0],
                second: v[1],
                addr: v[2],
                first_type: MockType::I64,
                second_type: MockType::I64,
                ordering: AtomicOrdering::Acquire,
                width: AtomicAccessWidth::Bits128,
                access: AccessQualifiers::FLAT,
            },
            2,
            1,
        ),
        (
            SsaOp::from(CallData::new(
                Some(v[0]),
                Callee::Direct(3),
                vec![v[1], v[2], v[3]],
            )),
            1,
            3,
        ),
        (
            SsaOp::from(CallData::new(
                None,
                Callee::Indirect {
                    fptr: v[0],
                    signature: 0,
                },
                vec![v[1]],
            )),
            0,
            2,
        ),
        (
            SsaOp::from(CallData {
                staged: vec![
                    StagedValue {
                        offset: 8,
                        value: v[2],
                    },
                    StagedValue {
                        offset: 12,
                        value: v[3],
                    },
                ],
                ..CallData::new(Some(v[0]), Callee::Direct(3), vec![v[1]])
            }),
            1,
            3,
        ),
        (SsaOp::returns(Some(v[0])), 0, 1),
        (SsaOp::returns(None), 0, 0),
        (
            SsaOp::Phi {
                dest: v[0],
                operands: vec![(0, v[1]), (1, v[2])],
            },
            1,
            2,
        ),
        (
            SsaOp::SystemOp(Box::new(NativeKindedData {
                kind: SystemOpKind::CpuId,
                mnemonic: "cpuid".to_string(),
                metadata: None,
                outputs: vec![v[0], v[1]],
                inputs: vec![v[2], v[3]],
            })),
            2,
            2,
        ),
        (
            whole_compute(
                Kernel::DecimalAdjust {
                    kind: BcdAdjustKind::AsciiMulAdjust,
                    base: 10,
                },
                vec![v[0], v[1]],
                vec![v[2]],
            ),
            2,
            1,
        ),
        (
            SsaOp::contract(
                Contraction::MaskedDot {
                    control: 0xff,
                    element_bits: 32,
                },
                vec![v[0]],
                vec![v[1], v[2]],
            ),
            1,
            2,
        ),
        (
            whole_compute(
                Kernel::MultiSad { offsets: 0x05 },
                vec![v[0]],
                vec![v[1], v[2]],
            ),
            1,
            2,
        ),
        (
            whole_compute(
                Kernel::StringCompare {
                    control: 0x0c,
                    explicit_length: true,
                    result_index: true,
                },
                vec![v[0], v[1]],
                vec![v[2], v[3]],
            ),
            2,
            2,
        ),
        (
            whole_compute(Kernel::HorizontalMinPos, vec![v[0]], vec![v[1]]),
            1,
            1,
        ),
        (
            whole_compute(Kernel::Intersect, vec![v[0], v[1]], vec![v[2], v[3]]),
            2,
            2,
        ),
        (
            whole_compute(Kernel::ShuffleBits, vec![v[0]], vec![v[1], v[2]]),
            1,
            2,
        ),
        (
            whole_compute(
                Kernel::Bitfield {
                    insert: false,
                    position: BitfieldPosition::Immediate {
                        index: 4,
                        length: 8,
                    },
                },
                vec![v[0]],
                vec![v[1]],
            ),
            1,
            1,
        ),
        (SsaOp::Jump { target: 4 }, 0, 0),
        (SsaOp::Nop, 0, 0),
        (SsaOp::ArgList { dest: v[0] }, 1, 0),
        (
            SsaOp::ArgNext {
                dest: v[0],
                list: v[1],
                ty: MockType::I32,
            },
            1,
            1,
        ),
        (
            SsaOp::ArgStep {
                dest: v[0],
                next: v[1],
                list: v[2],
                ty: MockType::I32,
            },
            2,
            1,
        ),
        (
            SsaOp::MakeTypedRef {
                dest: v[0],
                address: v[1],
                value_type: 0u32,
            },
            1,
            1,
        ),
        (
            SsaOp::TypedRefValue {
                dest: v[0],
                typed_ref: v[1],
                value_type: 0u32,
            },
            1,
            1,
        ),
        (
            SsaOp::TypedRefType {
                dest: v[0],
                typed_ref: v[1],
            },
            1,
            1,
        ),
    ]
}

/// **A computation's results are its defs and its sources its uses**, it is
/// pure where its kernel is, and a variable remap keeps its kernel and every
/// parameter the kernel carries: a bit field's position, a string
/// comparison's control, a multi-SAD's offsets, a decimal adjust's radix,
/// the minimum-position search's operands alone.
#[test]
fn a_computation_keeps_its_kernel_through_a_remap() {
    let v: Vec<SsaVarId> = (0..4).map(SsaVarId::from_index).collect();
    let integer = VectorElement::integer(64, None);
    for (kernel, outputs, inputs) in [
        (
            Kernel::Bitfield {
                insert: true,
                position: BitfieldPosition::Immediate {
                    index: 16,
                    length: 8,
                },
            },
            vec![v[0]],
            vec![v[1], v[2]],
        ),
        (Kernel::HorizontalMinPos, vec![v[0]], vec![v[1]]),
        (
            Kernel::StringCompare {
                control: 0x0c,
                explicit_length: false,
                result_index: false,
            },
            vec![v[0], v[1]],
            vec![v[2], v[3]],
        ),
        (
            Kernel::MultiSad { offsets: 0x05 },
            vec![v[0]],
            vec![v[1], v[2]],
        ),
        (
            Kernel::DecimalAdjust {
                kind: BcdAdjustKind::AsciiDivAdjust,
                base: 16,
            },
            vec![v[0], v[1]],
            vec![v[2]],
        ),
    ] {
        let kind = Computation::whole(kernel, integer, VectorRounding::None);
        let op: SsaOp<MockTarget> = SsaOp::compute(kind, outputs.clone(), inputs.clone());
        assert_eq!(op.defs().collect::<Vec<_>>(), outputs, "{kernel:?}");
        assert_eq!(op.uses(), inputs, "{kernel:?}");
        assert_eq!(op.effects().kind, SsaEffectKind::Pure, "{kernel:?}");
        assert!(!op.effects().may_throw, "{kernel:?}");
        let moved = |vars: &[SsaVarId]| -> Vec<SsaVarId> {
            vars.iter()
                .map(|var| SsaVarId::from_index(var.index() + 10))
                .collect()
        };
        let remapped = op.remap_variables(|x| Some(SsaVarId::from_index(x.index() + 10)));
        let SsaOp::Compute(data) = &remapped else {
            unreachable!("remap must preserve the computation")
        };
        assert_eq!(data.kind, kind, "{kernel:?}");
        assert_eq!(data.outputs, moved(&outputs), "{kernel:?}");
        assert_eq!(data.inputs, moved(&inputs), "{kernel:?}");
    }
}

/// A contraction reports its output as a def and its inputs as uses, is
/// pure, and survives a variable remap with its kind intact.
#[test]
fn a_contraction_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(0);
    let b = SsaVarId::from_index(1);
    let c = SsaVarId::from_index(2);
    let kind = Contraction::MaskedDot {
        control: 0x31,
        element_bits: 64,
    };
    let op: SsaOp<MockTarget> = SsaOp::contract(kind, vec![a], vec![b, c]);
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert_eq!(op.uses(), vec![b, c]);
    assert_eq!(op.effects().kind, SsaEffectKind::Pure);
    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(10)),
        v if v == b => Some(SsaVarId::from_index(11)),
        v if v == c => Some(SsaVarId::from_index(12)),
        _ => None,
    });
    let SsaOp::Contract(data) = &remapped else {
        unreachable!("remap must preserve the contraction")
    };
    assert_eq!(data.kind, kind);
    assert_eq!(data.outputs, vec![SsaVarId::from_index(10)]);
    assert_eq!(
        data.inputs,
        vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
    );
}

/// Each matrix, outer and complex contraction is its own operation: every
/// one has its own name, fingerprint key and display, states the inputs it
/// takes (an outer product its tile, two predicates and two factors; a
/// rotation the source added to and the one turned) and computes floats
/// exactly where its elements are floats.
#[test]
fn matrix_outer_and_complex_contractions_are_distinct_operations() {
    let int8 = ContractElements::Integer {
        source_bits: 8,
        dest_bits: 32,
        signed_a: false,
        signed_b: true,
    };
    let bfloat = ContractElements::Float(FloatElements::new(
        FloatSource::Format(FloatFormat::BFloat16),
        FloatFormat::Single,
    ));
    let matrix = |elements, layout, complex| {
        Contraction::Matrix(MatrixShape {
            elements,
            layout,
            complex,
        })
    };
    let segments = MatrixLayout::Segments { segment_bits: 128 };
    let complex = |product, element: VectorElement, saturates| {
        Contraction::Complex(ComplexShape {
            product,
            element,
            saturates,
        })
    };
    let half = VectorElement {
        kind: VectorElementKind::Float,
        bits: 16,
        scalar: false,
        signed: None,
        rounding: None,
    };
    let word = VectorElement::integer(32, Some(true));
    let multiply = |conjugate, accumulates| ComplexProduct::Multiply {
        conjugate,
        accumulates,
    };
    let cases = [
        (
            matrix(int8, segments, None),
            3,
            false,
            "contract.matrix u8xi8 -> 32 seg128",
        ),
        (
            matrix(bfloat, MatrixLayout::Tiles, None),
            6,
            true,
            "contract.matrix bf16 -> f32 tiles",
        ),
        (
            matrix(
                ContractElements::Float(FloatElements::new(
                    FloatSource::Format(FloatFormat::Half),
                    FloatFormat::Single,
                )),
                MatrixLayout::Tiles,
                Some(ComplexPart::Imaginary),
            ),
            6,
            true,
            "contract.matrix.complex.im f16 -> f32 tiles",
        ),
        (
            Contraction::Outer(OuterShape {
                elements: int8,
                subtract: false,
                product: OuterProduct::Multiply,
                governed: true,
                selection: None,
            }),
            5,
            false,
            "contract.outer.add u8xi8 -> 32",
        ),
        (
            Contraction::Outer(OuterShape {
                elements: int8,
                subtract: true,
                product: OuterProduct::MatchingBits,
                governed: true,
                selection: None,
            }),
            5,
            false,
            "contract.outer.sub.xnor_count u8xi8 -> 32",
        ),
        (
            Contraction::Outer(OuterShape {
                elements: bfloat,
                subtract: true,
                product: OuterProduct::Multiply,
                governed: true,
                selection: None,
            }),
            5,
            true,
            "contract.outer.sub bf16 -> f32",
        ),
        (
            Contraction::Outer(OuterShape {
                elements: ContractElements::Float(FloatElements::new(
                    FloatSource::Fp8,
                    FloatFormat::Half,
                )),
                subtract: false,
                product: OuterProduct::Multiply,
                governed: true,
                selection: None,
            }),
            6,
            true,
            "contract.outer.add fp8 -> f16",
        ),
        (
            Contraction::Outer(OuterShape {
                elements: int8,
                subtract: false,
                product: OuterProduct::Multiply,
                governed: false,
                selection: Some(OuterSelection {
                    values: 2,
                    candidates: 4,
                    per_value: true,
                }),
            }),
            5,
            false,
            "contract.outer.add u8xi8 -> 32 sparse4of8.each",
        ),
        (
            complex(multiply(false, false), half, false),
            2,
            true,
            "contract.complex.mul e16",
        ),
        (
            complex(multiply(true, true), half, false),
            3,
            true,
            "contract.complex.muladd.conj e16",
        ),
        (
            complex(ComplexProduct::Rotate(ComplexRotation::Quarter), word, true),
            2,
            false,
            "contract.complex.add.rot90 e32.s sat",
        ),
        (
            complex(
                ComplexProduct::Rotate(ComplexRotation::ThreeQuarters),
                half,
                false,
            ),
            2,
            true,
            "contract.complex.add.rot270 e16",
        ),
    ];
    let mut names = FxHashSet::default();
    let mut keys = FxHashSet::default();
    for (kind, inputs, float, text) in cases {
        assert_eq!(kind.input_count(), inputs, "{kind:?}");
        assert_eq!(kind.is_float(), float, "{kind:?}");
        let sources: Vec<_> = (1..=inputs).map(SsaVarId::from_index).collect();
        let listed = sources
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let op: SsaOp<MockTarget> = SsaOp::contract(kind, vec![SsaVarId::from_index(0)], sources);
        let (head, shape) = text.split_once(' ').unwrap_or((text, ""));
        assert_eq!(op.to_string(), format!("v0 = {head} {listed} {shape}"));
        names.insert(kind.name());
        keys.insert(kind.kind_str());
    }
    // The two bfloat16 matrices of different layouts share their name, as
    // integer matrices of different widths do: the shape tells them apart.
    assert_eq!(names.len(), 9);
    assert_eq!(keys.len(), 9);
}

/// **Arm's FP8 arithmetic is a rule of its own, raising nothing, and a
/// widening multiply-add is a dot product of one.** An FP8 dot, outer
/// product or matrix product raises no exception whatever FPCR states, and
/// reads FPMR as its last input; a half-precision one raises and reads no
/// FPMR. `fmlalb` of bytes multiplies one factor per lane, so its elements
/// state a group of one, which its display and its content state and a dot
/// of the widths' group leaves out.
#[test]
fn fp8_arithmetic_is_a_rule_of_its_own_and_raises_nothing() {
    let quad = FloatElements::new(FloatSource::Fp8, FloatFormat::Single);
    let single = FloatElements::new(FloatSource::Fp8, FloatFormat::Single).with_group(1);
    let halves = FloatElements::new(FloatSource::Format(FloatFormat::Half), FloatFormat::Single);
    assert_eq!((quad.group(), single.group(), halves.group()), (4, 1, 2));
    assert!(quad.group_is_the_widths() && !single.group_is_the_widths());
    let dot = |elements, rule| Contraction::FloatDot { elements, rule };
    assert!(!dot(quad, FloatDotRule::Fp8).raises_float_exceptions());
    assert!(!dot(single, FloatDotRule::Fp8).raises_float_exceptions());
    assert!(dot(halves, FloatDotRule::DotAdd).raises_float_exceptions());
    assert_eq!(dot(single, FloatDotRule::Fp8).input_count(), 4);
    let outer = |elements| {
        Contraction::Outer(OuterShape {
            elements: ContractElements::Float(elements),
            subtract: false,
            product: OuterProduct::Multiply,
            governed: true,
            selection: None,
        })
    };
    let matrix = |elements| {
        Contraction::Matrix(MatrixShape {
            elements: ContractElements::Float(elements),
            layout: MatrixLayout::Segments { segment_bits: 128 },
            complex: None,
        })
    };
    assert!(!outer(quad).raises_float_exceptions());
    assert!(!matrix(quad).raises_float_exceptions());
    // Each reads FPMR last.
    assert_eq!(
        (outer(quad).input_count(), matrix(quad).input_count()),
        (6, 4)
    );
    assert_eq!(
        (outer(halves).input_count(), matrix(halves).input_count()),
        (5, 3)
    );
    assert!(outer(halves).raises_float_exceptions(), "the control");
    let content = |kind: Contraction| {
        let op: SsaOp<MockTarget> = SsaOp::contract(
            kind,
            vec![SsaVarId::from_index(0)],
            (1..=kind.input_count()).map(SsaVarId::from_index).collect(),
        );
        let mut probe = crate::ir::content::tests::Probe::new(usize::MAX);
        assert_eq!(op.write_content(&mut probe), Ok(()));
        (probe.data, op.to_string())
    };
    let (of_one, shown) = content(dot(single, FloatDotRule::Fp8));
    let (of_four, shown_four) = content(dot(quad, FloatDotRule::Fp8));
    assert_ne!(of_one, of_four);
    assert!(shown.ends_with("fp8 -> f32 x1"), "{shown}");
    assert!(shown_four.ends_with("fp8 -> f32"), "{shown_four}");
    assert_ne!(
        content(outer(single)).0,
        content(outer(quad)).0,
        "an outer product's elements state their group too"
    );
}

/// Compile-time completeness guard. This exhaustive match (deliberately no
/// `_` arm) names every `SsaOp` variant, so adding a new variant breaks the
/// build here until it is consciously handled. That is the signal to (1)
/// classify it in the exhaustive `effects()` match, (2) cover its operands
/// in `visit_operands` / `visit_operands_mut`, and (3) add a sample to
/// `all_sample_ops` - and, for a new operand shape, to `visitor_battery`
/// with its operand census - so the invariant tests exercise it. Never
/// executed - it exists purely for the exhaustiveness check.
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
        SsaOp::VectorSlice { .. } => {}
        SsaOp::VectorInsertSlice { .. } => {}
        SsaOp::VectorConcat { .. } => {}
        SsaOp::VectorPack { .. } => {}
        SsaOp::VectorPackLoad { .. } => {}
        SsaOp::VectorPackStore { .. } => {}
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
        SsaOp::ArgList { .. } => {}
        SsaOp::ArgNext { .. } => {}
        SsaOp::ArgStep { .. } => {}
        SsaOp::MakeTypedRef { .. } => {}
        SsaOp::TypedRefValue { .. } => {}
        SsaOp::TypedRefType { .. } => {}
        SsaOp::Call(_) => {}
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
        SsaOp::MoveBlk { .. } => {}
        SsaOp::Fence { .. } => {}
        SsaOp::SystemOp(_) => {}
        SsaOp::Compute(_) => {}
        SsaOp::Contract(_) => {}
        SsaOp::VectorPredicateWhile(_) => {}
        SsaOp::VectorExtendInLane(_) => {}
        SsaOp::VectorElementCount(_) => {}
        SsaOp::VectorStructLoadReplicate(_) => {}
        SsaOp::VectorSveCompute(_) => {}
        SsaOp::CompareBlk(_) => {}
        SsaOp::ScanBlk { .. } => {}
        SsaOp::LoadBlk { .. } => {}
        SsaOp::MergeFlags { .. } => {}
        SsaOp::InsertFlag { .. } => {}
        SsaOp::CallClobber { .. } => {}
        SsaOp::AtomicCondRmw { .. } => {}
        SsaOp::AtomicLoad { .. } => {}
        SsaOp::AtomicStore { .. } => {}
        SsaOp::AtomicStoreConditional { .. } => {}
        SsaOp::AtomicPairLoad { .. } => {}
        SsaOp::AtomicPairStoreConditional { .. } => {}
        SsaOp::AtomicExchange { .. } => {}
        SsaOp::AtomicLockRmw { .. } => {}
        SsaOp::AtomicCmpXchg { .. } => {}
        SsaOp::AtomicPairCmpXchg { .. } => {}
        SsaOp::AtomicPairRmw { .. } => {}
        SsaOp::AtomicCheckedRmw { .. } => {}
        SsaOp::InitObj { .. } => {}
        SsaOp::CopyObj { .. } => {}
        SsaOp::LoadObj { .. } => {}
        SsaOp::StoreObj { .. } => {}
        SsaOp::Nop => {}
        SsaOp::Break(_) => {}
        SsaOp::Ckfinite { .. } => {}
        SsaOp::FpClassify { .. } => {}
        SsaOp::FloatRaise { .. } => {}
        SsaOp::FloatFunction(_) => {}
        SsaOp::FpuControl(_) => {}
        SsaOp::LocalAlloc { .. } => {}
        SsaOp::Phi { .. } => {}
        SsaOp::RuntimeValue { .. } => {}
    }
}

/// A computation of `kernel` over a whole 64-bit integer, defining `outputs`
/// from `inputs`.
fn whole_compute(
    kernel: Kernel,
    outputs: Vec<SsaVarId>,
    inputs: Vec<SsaVarId>,
) -> SsaOp<MockTarget> {
    SsaOp::compute(
        Computation::whole(
            kernel,
            VectorElement::integer(64, None),
            VectorRounding::None,
        ),
        outputs,
        inputs,
    )
}

/// Constructs one sample of every `SsaOp` variant.
///
/// Field values are placeholders chosen only to be type-valid; the tests
/// assert structural invariants, not semantics. Kept complete by
/// `op_variant_exhaustiveness_sentinel`: adding a variant breaks the build
/// there until a sample is added here too.
///
/// Every operand is drawn from one of two disjoint id spaces according to its
/// role: definitions from `{d0, d1}` (90, 91) and uses from `{u0, u1}` (80,
/// 81). One id for every field would make a transposed role -- an input
/// reported as a definition, or the reverse -- invisible to every test in this
/// file, because the transposed operand would carry the id the correct one
/// carries. Two ids per role additionally make an ordering change visible.
/// `payload_roles_are_not_transposed` is the universal gate over that
/// property; it needs no per-variant list.
///
/// Every payload-carrying sample gets two outputs and two inputs, so a
/// truncating definition walk cannot pass by reporting the single output it
/// happened to see.
fn all_sample_ops() -> Vec<SsaOp<MockTarget>> {
    // Definitions and uses occupy disjoint id spaces, so a transposed role --
    // an input reported as a definition, or the reverse -- shows up as an id
    // from the wrong space rather than as the same id in both lists. Two ids
    // per role make an *ordering* change visible as well.
    let [d0, d1, d2, d3] = [90, 91, 92, 93].map(SsaVarId::from_index);
    let [u0, u1, u2] = [80, 81, 82].map(SsaVarId::from_index);
    vec![
        SsaOp::Const {
            dest: d0,
            value: ConstValue::I32(0),
        },
        SsaOp::Add {
            dest: d0,
            left: u0,
            right: u1,
            flags: Some(d1),
        },
        SsaOp::AddOvf {
            dest: d0,
            left: u0,
            right: u1,
            unsigned: false,
            flags: Some(d1),
        },
        SsaOp::Sub {
            dest: d0,
            left: u0,
            right: u1,
            flags: Some(d1),
        },
        SsaOp::SubOvf {
            dest: d0,
            left: u0,
            right: u1,
            unsigned: false,
            flags: Some(d1),
        },
        SsaOp::Mul {
            dest: d0,
            left: u0,
            right: u1,
            flags: Some(d1),
        },
        SsaOp::MulOvf {
            dest: d0,
            left: u0,
            right: u1,
            unsigned: false,
            flags: Some(d1),
        },
        SsaOp::WideMul {
            low: d0,
            high: d1,
            left: u0,
            right: u1,
            unsigned: false,
        },
        SsaOp::Div {
            dest: d0,
            left: u0,
            right: u1,
            unsigned: false,
            flags: Some(d1),
        },
        SsaOp::Rem {
            dest: d0,
            left: u0,
            right: u1,
            unsigned: false,
            flags: Some(d1),
        },
        SsaOp::FloatCompareFlags {
            flags: d0,
            left: u0,
            right: u1,
            signaling: false,
        },
        SsaOp::WideDiv {
            quotient: d0,
            remainder: d1,
            high: u0,
            low: u1,
            divisor: u0,
            unsigned: false,
        },
        SsaOp::Neg {
            dest: d0,
            operand: u0,
            flags: Some(d1),
        },
        SsaOp::And {
            dest: d0,
            left: u0,
            right: u1,
            flags: Some(d1),
        },
        SsaOp::Or {
            dest: d0,
            left: u0,
            right: u1,
            flags: Some(d1),
        },
        SsaOp::Xor {
            dest: d0,
            left: u0,
            right: u1,
            flags: Some(d1),
        },
        SsaOp::Not {
            dest: d0,
            operand: u0,
            flags: Some(d1),
        },
        SsaOp::Shl {
            dest: d0,
            value: u0,
            amount: u1,
            flags: Some(d1),
        },
        SsaOp::Shr {
            dest: d0,
            value: u0,
            amount: u1,
            unsigned: false,
            flags: Some(d1),
        },
        SsaOp::Rol {
            dest: d0,
            value: u0,
            amount: u1,
        },
        SsaOp::Ror {
            dest: d0,
            value: u0,
            amount: u1,
        },
        SsaOp::BSwap { dest: d0, src: u0 },
        SsaOp::BRev { dest: d0, src: u0 },
        SsaOp::BitScanForward { dest: d0, src: u0 },
        SsaOp::BitScanReverse { dest: d0, src: u0 },
        SsaOp::Popcount { dest: d0, src: u0 },
        SsaOp::Parity { dest: d0, src: u0 },
        SsaOp::Ceq {
            dest: d0,
            left: u0,
            right: u1,
            signaling: false,
        },
        SsaOp::Clt {
            dest: d0,
            left: u0,
            right: u1,
            unsigned: false,
            signaling: false,
        },
        SsaOp::Cgt {
            dest: d0,
            left: u0,
            right: u1,
            unsigned: false,
            signaling: false,
        },
        SsaOp::BoolAnd {
            dest: d0,
            left: u0,
            right: u1,
        },
        SsaOp::BoolOr {
            dest: d0,
            left: u0,
            right: u1,
        },
        SsaOp::BoolXor {
            dest: d0,
            left: u0,
            right: u1,
        },
        SsaOp::BoolNot {
            dest: d0,
            value: u0,
        },
        SsaOp::IntConv {
            dest: d0,
            operand: u0,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        },
        SsaOp::IntToPtr {
            dest: d0,
            operand: u0,
            target: MockType::I32,
        },
        SsaOp::PtrToInt {
            dest: d0,
            operand: u0,
            target: MockType::I32,
        },
        SsaOp::IntToFloat {
            dest: d0,
            operand: u0,
            target: MockType::I32,
            unsigned: false,
            rounding: None,
            environment: None,
        },
        SsaOp::FloatToInt {
            dest: d0,
            operand: u0,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
            source: None,
        },
        SsaOp::FloatConv {
            dest: d0,
            operand: u0,
            target: MockType::I32,
            rounding: None,
            source: None,
            environment: None,
        },
        SsaOp::Bitcast {
            dest: d0,
            operand: u0,
            target: MockType::I32,
        },
        SsaOp::Select {
            dest: d0,
            condition: u0,
            true_val: u1,
            false_val: u0,
        },
        SsaOp::ReadFlags {
            dest: d0,
            flags: u0,
            mask: FlagsMask::from_bits(0),
        },
        SsaOp::VectorUnary {
            dest: d0,
            value: u0,
            kind: VectorUnaryKind::Neg,
            element: VectorElement {
                kind: VectorElementKind::Integer,
                bits: 32,
                scalar: false,
                signed: None,
                rounding: None,
            },
            environment: None,
        },
        SsaOp::VectorBinary {
            dest: d0,
            left: u0,
            right: u1,
            kind: VectorBinaryKind::Add,
            element: VectorElement {
                kind: VectorElementKind::Integer,
                bits: 32,
                scalar: false,
                signed: None,
                rounding: None,
            },
            environment: None,
        },
        SsaOp::VectorTernary {
            dest: d0,
            first: u0,
            second: u1,
            third: u0,
            kind: VectorTernaryKind::Fma,
            element: VectorElement::float(32),
            environment: None,
        },
        SsaOp::VectorPredicatedUnary {
            dest: d0,
            value: u0,
            mask: u1,
            passthrough: Some(u0),
            kind: VectorUnaryKind::Neg,
            element: VectorElement::default(),
            mode: VectorMaskMode::Merge,
            environment: None,
        },
        SsaOp::VectorPredicatedBinary {
            dest: d0,
            left: u0,
            right: u1,
            mask: u0,
            passthrough: Some(u1),
            kind: VectorBinaryKind::Add,
            element: VectorElement::default(),
            mode: VectorMaskMode::Merge,
            environment: None,
        },
        SsaOp::VectorPredicatedTernary {
            dest: d0,
            first: u0,
            second: u1,
            third: u0,
            mask: u1,
            passthrough: Some(u0),
            kind: VectorTernaryKind::Fma,
            element: VectorElement::float(32),
            mode: VectorMaskMode::Merge,
            environment: None,
        },
        SsaOp::VectorCompare {
            dest: d0,
            left: u0,
            right: u1,
            kind: VectorCompareKind::Eq,
            element: VectorElement::default(),
            signaling: false,
        },
        SsaOp::VectorLoad {
            dest: d0,
            addr: u0,
            vector_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorStore {
            addr: u0,
            value: u1,
            vector_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorMaskedLoad {
            dest: d0,
            addr: u0,
            mask: u1,
            passthrough: Some(u0),
            vector_type: MockType::I32,
            mode: VectorMaskMode::Merge,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorMaskedStore {
            addr: u0,
            value: u1,
            mask: u0,
            vector_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorBroadcastLoad {
            dest: d0,
            addr: u0,
            vector_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorGather {
            dest: d0,
            base: u0,
            indices: u1,
            mask: u0,
            passthrough: Some(u1),
            vector_type: MockType::I32,
            mode: VectorMaskMode::Merge,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorFaultingLoad {
            dest: d0,
            fault: Some(d1),
            addr: u0,
            mask: Some(u1),
            passthrough: Some(u0),
            vector_type: MockType::I32,
            fault_mode: VectorFaultMode::Normal,
            mask_mode: VectorMaskMode::Merge,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorSegmentLoad {
            dests: vec![d0, d1],
            base: u0,
            mask: Some(u1),
            vector_type: MockType::I32,
            segments: 0,
            layout: VectorSegmentLayout::Interleaved,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorScatter {
            base: u0,
            indices: u1,
            value: u0,
            mask: u1,
            vector_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorSegmentStore {
            base: u0,
            values: vec![u1],
            mask: Some(u0),
            vector_type: MockType::I32,
            segments: 0,
            layout: VectorSegmentLayout::Interleaved,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorExtract {
            dest: d0,
            vector: u0,
            lane: 0,
        },
        SsaOp::VectorInsert {
            dest: d0,
            vector: u0,
            lane: 0,
            value: u1,
        },
        SsaOp::VectorSplat {
            dest: d0,
            value: u0,
            vector_type: MockType::I32,
        },
        SsaOp::VectorShuffle {
            dest: d0,
            left: u0,
            right: Some(u1),
            selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                VectorShuffleLane::Zero,
            ])),
        },
        SsaOp::VectorShuffle {
            dest: d0,
            left: u0,
            right: Some(u1),
            selector: VectorShuffleSelector::Index(u2),
        },
        SsaOp::VectorCast {
            dest: d0,
            value: u0,
            target_type: MockType::I32,
            from: VectorCastKind::Signed,
            to: VectorCastKind::Signed,
            rounding: None,
            saturate: true,
            environment: None,
            format: None,
        },
        SsaOp::VectorReinterpret {
            dest: d0,
            value: u0,
            target_type: MockType::I32,
        },
        SsaOp::VectorPack {
            dest: d0,
            value: u0,
            mask: u1,
            passthrough: Some(u0),
            vector_type: MockType::I32,
            element_bits: 0,
            kind: VectorPackKind::Compress,
            mode: VectorMaskMode::Merge,
        },
        SsaOp::VectorPackLoad {
            dest: d0,
            addr: u0,
            mask: u1,
            passthrough: Some(u0),
            vector_type: MockType::I32,
            element_bits: 0,
            kind: VectorPackKind::Compress,
            mode: VectorMaskMode::Merge,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorPackStore {
            addr: u0,
            value: u1,
            mask: u0,
            vector_type: MockType::I32,
            element_bits: 0,
            kind: VectorPackKind::Compress,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::VectorReduce {
            dest: d0,
            value: u0,
            kind: VectorReduceKind::Add,
            element: VectorElement::default(),
            initial: None,
            ordered: false,
            groups: None,
            environment: None,
        },
        SsaOp::VectorBitmask {
            dest: d0,
            value: u0,
            kind: VectorBitmaskKind::LaneMostSignificantBits,
        },
        SsaOp::Jump { target: 0 },
        SsaOp::Branch {
            condition: u0,
            true_target: 0,
            false_target: 0,
        },
        SsaOp::BranchCmp {
            left: u0,
            right: u1,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 0,
            false_target: 0,
        },
        SsaOp::BranchFlags {
            flags: u0,
            condition: FlagCondition::Carry,
            true_target: 0,
            false_target: 0,
        },
        SsaOp::Switch {
            value: u0,
            targets: vec![0usize],
            default: 0,
        },
        SsaOp::IndirectBranch {
            target: u0,
            resolved_targets: vec![0usize],
            state: Box::default(),
        },
        SsaOp::returns(Some(u0)),
        SsaOp::LoadField {
            access: AccessQualifiers::FLAT,
            dest: d0,
            object: u0,
            field: 0u32,
        },
        SsaOp::StoreField {
            access: AccessQualifiers::FLAT,
            object: u0,
            field: 0u32,
            value: u1,
        },
        SsaOp::LoadStaticField {
            access: AccessQualifiers::FLAT,
            dest: d0,
            field: 0u32,
        },
        SsaOp::StoreStaticField {
            access: AccessQualifiers::FLAT,
            field: 0u32,
            value: u0,
        },
        SsaOp::LoadFieldAddr {
            dest: d0,
            object: u0,
            field: 0u32,
        },
        SsaOp::LoadStaticFieldAddr {
            dest: d0,
            field: 0u32,
        },
        SsaOp::LoadElement {
            dest: d0,
            array: u0,
            index: u1,
            elem_type: MockType::I32,
        },
        SsaOp::StoreElement {
            array: u0,
            index: u1,
            value: u0,
            elem_type: MockType::I32,
        },
        SsaOp::LoadElementAddr {
            readonly: false,
            dest: d0,
            array: u0,
            index: u1,
            elem_type: 0u32,
        },
        SsaOp::PtrAdd {
            dest: d0,
            base: u0,
            index: Some(u1),
            stride: 4,
            offset: 8,
            result_type: MockType::I64,
        },
        SsaOp::ArrayLength {
            dest: d0,
            array: u0,
        },
        SsaOp::LoadIndirect {
            dest: d0,
            addr: u0,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::StoreIndirect {
            addr: u0,
            value: u1,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::NewObj {
            dest: d0,
            ctor: 0u32,
            args: vec![u0],
        },
        SsaOp::NewArr {
            dest: d0,
            elem_type: 0u32,
            length: u0,
        },
        SsaOp::CastClass {
            dest: d0,
            object: u0,
            target_type: 0u32,
        },
        SsaOp::IsInst {
            dest: d0,
            object: u0,
            target_type: 0u32,
        },
        SsaOp::Box {
            dest: d0,
            value: u0,
            value_type: 0u32,
        },
        SsaOp::Unbox {
            dest: d0,
            object: u0,
            value_type: 0u32,
        },
        SsaOp::UnboxAny {
            dest: d0,
            object: u0,
            value_type: 0u32,
        },
        SsaOp::SizeOf {
            dest: d0,
            value_type: 0u32,
        },
        SsaOp::LoadToken {
            dest: d0,
            token: 0u32,
        },
        SsaOp::from(CallData::new(Some(d0), Callee::Direct(0u32), vec![u0])),
        SsaOp::from(CallData::new(Some(d0), Callee::Virtual(0u32), vec![u0])),
        SsaOp::from(CallData::new(
            Some(d0),
            Callee::Constrained {
                method: 0u32,
                constraint_type: 0u32,
                instance: true,
            },
            vec![u0],
        )),
        SsaOp::from(CallData::new(
            Some(d0),
            Callee::Constrained {
                method: 0u32,
                constraint_type: 0u32,
                instance: false,
            },
            vec![],
        )),
        SsaOp::from(CallData::new(
            Some(d0),
            Callee::Indirect {
                fptr: u0,
                signature: 0u32,
            },
            vec![u1],
        )),
        SsaOp::LoadFunctionPtr {
            dest: d0,
            method: 0u32,
            constraint: None,
        },
        SsaOp::LoadFunctionPtr {
            dest: d0,
            method: 0u32,
            constraint: Some(0u32),
        },
        SsaOp::LoadVirtFunctionPtr {
            dest: d0,
            object: u0,
            method: 0u32,
        },
        SsaOp::LoadArg {
            dest: d0,
            arg_index: 0,
        },
        SsaOp::LoadLocal {
            dest: d0,
            local_index: 0,
        },
        SsaOp::LoadArgAddr {
            dest: d0,
            arg_index: 0,
        },
        SsaOp::LoadLocalAddr {
            dest: d0,
            local_index: 0,
        },
        SsaOp::Copy { dest: d0, src: u0 },
        SsaOp::Pop { value: u0 },
        SsaOp::Throw { exception: u0 },
        SsaOp::Rethrow,
        SsaOp::EndFinally,
        SsaOp::EndFilter { result: u0 },
        SsaOp::RuntimeValue {
            dest: d0,
            kind: RuntimeValueKind::ExceptionObject,
        },
        SsaOp::InterruptReturn,
        SsaOp::Unreachable,
        SsaOp::Leave { target: 0 },
        SsaOp::InitBlk {
            dest_addr: u0,
            value: u1,
            size: u0,
            backward: None,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::CopyBlk {
            dest_addr: u0,
            src_addr: u1,
            size: u0,
            element_bytes: 1,
            backward: None,
            dest_access: AccessQualifiers::FLAT,
            src_access: AccessQualifiers::FLAT,
        },
        SsaOp::MoveBlk {
            dest_addr: u0,
            src_addr: u1,
            size: u0,
            dest_access: AccessQualifiers::FLAT,
            src_access: AccessQualifiers::FLAT,
        },
        SsaOp::Fence {
            kind: FenceKind::Full,
        },
        SsaOp::SystemOp(Box::new(NativeKindedData {
            kind: SystemOpKind::CpuId,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
        whole_compute(Kernel::BitDeposit, vec![d0, d1], vec![u0, u1]),
        SsaOp::contract(
            Contraction::MultiplyAdd52 { high: false },
            vec![d0, d1],
            vec![u0, u1],
        ),
        SsaOp::VectorPredicateWhile(Box::new(VectorPredicateWhileData {
            kind: VectorCompareKind::Eq,
            unsigned: false,
            elements: 8,
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
        SsaOp::VectorExtendInLane(Box::new(VectorExtendInLaneData {
            signed: false,
            source_bits: 0,
            element_bits: 0,
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
        SsaOp::VectorElementCount(Box::new(VectorElementCountData {
            element_bits: 0,
            multiplier: 0,
            pattern: ElementPattern::All,
            streaming: Some(u0),
            outputs: vec![d0, d1],
        })),
        SsaOp::VectorStructLoadReplicate(Box::new(VectorStructLoadReplicateData {
            count: 0,
            element_bits: 0,
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
        SsaOp::VectorSveCompute(Box::new(VectorSveComputeData {
            op: SveComputeKind::AddCarryBottom,
            element_bits: 0,
            rotation: 0,
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
        SsaOp::CompareBlk(Box::new(CompareBlkData {
            remaining: d0,
            left_next: d1,
            right_next: d2,
            flags: d3,
            left: u0,
            right: u1,
            count: u2,
            flags_in: Some(u0),
            backward: Some(u1),
            element_bytes: 1,
            while_equal: true,
            left_access: AccessQualifiers::FLAT,
            right_access: AccessQualifiers::FLAT,
        })),
        SsaOp::ScanBlk {
            remaining: d0,
            next: d1,
            flags: d2,
            address: u0,
            value: u1,
            count: u2,
            flags_in: Some(u0),
            backward: None,
            element_bytes: 2,
            while_equal: false,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::LoadBlk {
            value: d0,
            next: d1,
            address: u0,
            count: u1,
            value_in: Some(u2),
            backward: None,
            element_bytes: 4,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::MergeFlags {
            dest: d0,
            base: u0,
            update: u1,
            mask: FlagsMask::CARRY,
        },
        SsaOp::InsertFlag {
            dest: d0,
            base: u0,
            bit: u1,
            mask: FlagsMask::CARRY,
        },
        SsaOp::CallClobber {
            outputs: vec![d0, d1],
            machine: None,
        },
        SsaOp::AtomicCondRmw {
            old: d0,
            addr: u0,
            compare: u1,
            value: u0,
            op: AtomicRmwOp::Add,
            condition: CompareCondition::Flag(FlagCondition::Overflow),
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Bits32,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicLoad {
            dest: d0,
            addr: u0,
            value_type: MockType::I32,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            reserves: true,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicStore {
            addr: u0,
            value: u1,
            value_type: MockType::I32,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicStoreConditional {
            status: d0,
            status_encoding: AtomicStatus::Unspecified,
            consumes_reservation: false,
            addr: u0,
            value: u1,
            value_type: MockType::I32,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicPairLoad {
            first: d0,
            second: d1,
            addr: u0,
            first_type: MockType::I32,
            second_type: MockType::I32,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicPairStoreConditional {
            status: d0,
            status_encoding: AtomicStatus::Unspecified,
            consumes_reservation: false,
            addr: u0,
            first_value: u1,
            second_value: u0,
            first_type: MockType::I32,
            second_type: MockType::I32,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicExchange {
            dest: d0,
            addr: u0,
            value: u1,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicLockRmw {
            dest: d0,
            addr: u0,
            value: u1,
            op: AtomicRmwOp::Xchg,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicCmpXchg {
            old: d0,
            success: Some(d1),
            addr: u0,
            expected: u1,
            desired: u0,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            weak: false,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicPairCmpXchg {
            old_first: d0,
            old_second: d1,
            addr: u0,
            expected_first: u1,
            expected_second: u0,
            desired_first: u1,
            desired_second: u0,
            success_ordering: AtomicOrdering::Relaxed,
            failure_ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            weak: false,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicPairRmw {
            old_first: d0,
            old_second: d1,
            addr: u0,
            value_first: u1,
            value_second: u0,
            op: AtomicRmwOp::Or,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits8,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::AtomicCheckedRmw {
            old: d0,
            status: d1,
            addr: u0,
            compare: Some(u1),
            value: u0,
            op: AtomicRmwOp::Xchg,
            rcw_mask: u1,
            rcws_mask: Some(u0),
            protection: RcwProtection::Regime,
            ordering: AtomicOrdering::Relaxed,
            width: AtomicAccessWidth::Bits64,
            access: AccessQualifiers::FLAT,
        },
        SsaOp::InitObj {
            dest_addr: u0,
            value_type: 0u32,
        },
        SsaOp::CopyObj {
            dest_addr: u0,
            src_addr: u1,
            value_type: 0u32,
        },
        SsaOp::LoadObj {
            access: AccessQualifiers::FLAT,
            dest: d0,
            src_addr: u0,
            value_type: 0u32,
        },
        SsaOp::StoreObj {
            access: AccessQualifiers::FLAT,
            dest_addr: u0,
            value: u1,
            value_type: 0u32,
        },
        SsaOp::Nop,
        SsaOp::Break(BreakpointOp::UndefinedInstruction),
        SsaOp::Ckfinite {
            dest: d0,
            operand: u0,
        },
        SsaOp::FpClassify {
            dest: d0,
            operand: u0,
            layout: FloatClassLayout::RiscV,
        },
        SsaOp::float_raise(
            d0,
            SsaOp::FloatCompareFlags {
                flags: d1,
                left: u0,
                right: u1,
                signaling: true,
            },
            None,
            FloatExceptions::INVALID,
            FloatExceptionLayout::IEEE,
        ),
        SsaOp::FloatFunction(Box::new(KindedVecData {
            kind: FloatFunctionKind::Sin,
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
        SsaOp::FpuControl(Box::new(KindedVecData {
            kind: FpuControlKind::RestoreExtended,
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
        SsaOp::LocalAlloc {
            dest: d0,
            size: u0,
            zeroed: true,
        },
        SsaOp::Phi {
            dest: d0,
            operands: vec![(0usize, u0)],
        },
        SsaOp::ArgList { dest: d0 },
        SsaOp::ArgNext {
            dest: d0,
            list: u0,
            ty: MockType::I32,
        },
        SsaOp::ArgStep {
            dest: d0,
            next: d1,
            list: u0,
            ty: MockType::I32,
        },
        SsaOp::MakeTypedRef {
            dest: d0,
            address: u0,
            value_type: 0u32,
        },
        SsaOp::TypedRefValue {
            dest: d0,
            typed_ref: u0,
            value_type: 0u32,
        },
        SsaOp::TypedRefType {
            dest: d0,
            typed_ref: u0,
        },
        SsaOp::VectorSlice {
            dest: d0,
            vector: u0,
            offset: u1,
            length: u0,
            target_type: MockType::V4I32,
        },
        SsaOp::VectorInsertSlice {
            dest: d0,
            vector: u0,
            offset: u1,
            value: u0,
            target_type: MockType::V4I32,
        },
        SsaOp::VectorConcat {
            dest: d0,
            left: u0,
            right: u1,
            target_type: MockType::V4I32,
        },
        // A computation over lanes as well as the whole-value one above.
        SsaOp::Compute(Box::new(KindedVecData {
            kind: Computation {
                canonical_nan: false,
                kernel: Kernel::Binary(VectorBinaryKind::Add),
                element: VectorElement::integer(32, None),
                predicate: VectorPredicateLayout::Packed,
                rounding: VectorRounding::None,
                status: VectorStatus::None,
                lanes: Some(LaneControl {
                    mask_tail: false,
                    tail: VectorTailPolicy::Undisturbed,
                    inactive: None,
                    no_passthrough: false,
                }),
            },
            outputs: vec![d0, d1],
            inputs: vec![u0, u1],
        })),
    ]
}

/// Across every variant, and a call once for each [`Callee`]: the operand
/// visitor agrees with `defs()` / `uses()`, the primary `dest()` is the first
/// `Def`-role operand, and `opcode_name()` is unique per sample. This is the
/// runtime half of the completeness guarantee (the sentinel is the
/// compile-time half).
#[test]
fn all_variants_visitor_defs_uses_and_opcode_are_consistent() {
    let ops = all_sample_ops();
    assert_eq!(
        ops.len(),
        176,
        "every SsaOp variant has one sample, a call or a function pointer one per form, a \
         computation one over a whole value and one over lanes, and a shuffle one per selector"
    );
    let mut names = FxHashSet::default();
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
    assert_eq!(names.len(), 176, "opcode_name must be unique per sample");
}

/// How many samples in `all_sample_ops` both pop two operands and push two.
///
/// Every one of the boxed-payload variants is built with two inputs and two
/// outputs, so this is their count plus the handful of non-payload variants
/// that happen to have the same shape. Pinning it makes
/// `payload_variants_report_all_outputs_as_defs` notice a variant that quietly
/// leaves the population.
const PAYLOAD_SAMPLE_COUNT: usize = 12;

/// No sample reports an operand in the wrong role.
///
/// `all_sample_ops` draws definitions from `{d0, .., d3}` and uses from
/// `{u0, u1, u2}`, two disjoint id spaces, so this one assertion covers every variant at once:
/// a definition that is really an input, or an input reported as a definition,
/// lands an id from the wrong space. It replaces a per-variant list of expected
/// operands, which would have to be extended by hand for variant 201 and would
/// silently omit anything nobody remembered to add.
#[test]
fn payload_roles_are_not_transposed() {
    let defs_space = [90, 91, 92, 93].map(SsaVarId::from_index);
    let uses_space = [80, 81, 82].map(SsaVarId::from_index);
    for op in all_sample_ops() {
        let defs: Vec<SsaVarId> = op.defs().collect();
        let uses = op.uses();
        for def in &defs {
            assert!(
                defs_space.contains(def),
                "{op} reports {def:?} as a definition, but it is an input"
            );
        }
        for used in &uses {
            assert!(
                uses_space.contains(used),
                "{op} reports {used:?} as a use, but it is an output"
            );
        }
        assert!(
            !defs.iter().any(|d| uses.contains(d)),
            "{op} reports the same variable as both a definition and a use"
        );
    }
}

/// Every payload-carrying sample reports both of its outputs as definitions and
/// both of its inputs as uses.
///
/// `all_sample_ops` gives each of them two of each, so a definition walk that
/// truncates after the first output -- the failure mode a `_` fallback arm
/// hides -- shows up here as a count of one. The population is pinned rather
/// than listed: a new payload variant changes the number and forces a look.
#[test]
fn payload_variants_report_all_outputs_as_defs() {
    let mut two_by_two = 0_usize;
    for op in all_sample_ops() {
        if op.stack_effect() == (2, 2) {
            assert_eq!(op.defs().count(), 2, "{op} loses a definition");
            assert_eq!(op.uses().len(), 2, "{op} loses a use");
            two_by_two += 1;
        }
    }
    assert_eq!(
        two_by_two, PAYLOAD_SAMPLE_COUNT,
        "the population of two-in/two-out samples changed"
    );
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
/// `uses()` (use set and order) for every representative op shape, and the
/// `visitor_battery` entries must additionally match the operand census that
/// battery states.
///
/// `defs()`, `uses()` and `visit_operands` are three readers of one walk, so
/// agreement between them cannot catch an operand the walk never reaches. The
/// hand-written census is the independent account that can.
#[test]
fn visit_operands_agrees_with_defs_and_uses() {
    let census = visitor_battery();
    let all = all_sample_ops()
        .into_iter()
        .map(|op| (op, None))
        .chain(census.into_iter().map(|(op, d, u)| (op, Some((d, u)))));
    for (op, expected) in all {
        let mut visited_defs = Vec::new();
        let mut visited_uses = Vec::new();
        op.visit_operands(|role, var| match role {
            OperandRole::Def | OperandRole::FlagsDef => visited_defs.push(var),
            OperandRole::Use => visited_uses.push(var),
        });

        let defs: Vec<SsaVarId> = op.defs().collect();
        assert_eq!(visited_defs, defs, "defs mismatch for {op}");
        assert_eq!(visited_uses, op.uses(), "uses mismatch for {op}");
        if let Some((expected_defs, expected_uses)) = expected {
            assert_eq!(
                defs.len(),
                expected_defs,
                "{op} does not have the {expected_defs} definitions the battery declares"
            );
            assert_eq!(
                visited_uses.len(),
                expected_uses,
                "{op} does not have the {expected_uses} uses the battery declares"
            );
        }
        if matches!(op, SsaOp::FloatCompareFlags { .. }) {
            // Defines only a flags bundle; there is no primary destination.
            assert_eq!(op.dest(), None);
        } else {
            assert_eq!(op.dest(), defs.first().copied(), "dest mismatch for {op}");
        }
    }
}

/// `visit_operands_mut` must visit exactly the same operands, in the same
/// order and with the same roles, as `visit_operands` - the invariant its
/// doc comment states.
///
/// The two are ~1200-line parallel matches over every `SsaOp` variant,
/// and they are consumed for *different* purposes: the shared visitor feeds
/// def-use index construction, while the mutable one performs variable
/// substitution (`replace_uses`, `replace_def`). Drift between them is
/// therefore silent SSA corruption rather than a crash - an operand visited
/// by only the immutable arm is recorded as a use but never rewritten by a
/// rename, leaving a dangling reference the verifier only catches later, if
/// at all. The compile-time exhaustiveness sentinel forces a new variant to
/// be *considered* in both, but cannot check that both got it right.
///
/// Runs over both batteries, which are complementary: `all_sample_ops()`
/// reaches every variant but populates optional operands, while
/// `visitor_battery()` covers fewer variants and includes the absent shapes
/// (`flags: None`) that exercise the visitors' skip paths.
#[test]
fn visit_operands_mut_agrees_with_visit_operands() {
    let battery = visitor_battery().into_iter().map(|(op, _, _)| op);
    for mut op in all_sample_ops().into_iter().chain(battery) {
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

/// `first_use` is the allocation-free spelling of `uses().first()`, so the two
/// must agree on every variant - including the ops that read nothing, where the
/// `Vec` spelling answers `None` from an empty vector and this one answers it
/// without building the vector at all.
///
/// Runs over both batteries: `all_sample_ops()` reaches every variant with
/// their optional operands populated, `visitor_battery()` adds the absent
/// shapes (`flags: None`) that shift which operand comes first.
#[test]
fn first_use_agrees_with_uses_first() {
    let battery = visitor_battery().into_iter().map(|(op, _, _)| op);
    for op in all_sample_ops().into_iter().chain(battery) {
        assert_eq!(
            op.first_use(),
            op.uses().first().copied(),
            "first_use disagrees with uses().first() for {op}"
        );
    }
}

/// `replace_def` rewrites secondary outputs uniformly, across every op that has
/// them - a typed system op's output list as well as `NativeOpaque`'s.
#[test]
fn replace_def_covers_secondary_outputs() {
    let old = SsaVarId::from_index(1);
    let new = SsaVarId::from_index(9);
    let mut op: SsaOp<MockTarget> = SsaOp::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::Timestamp { aux: true },
        mnemonic: "rdtscp".to_string(),
        metadata: None,
        outputs: vec![SsaVarId::from_index(0), old],
        inputs: vec![old],
    }));
    assert!(op.replace_def(old, new));
    let SsaOp::SystemOp(data) = &op else {
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
        kind: SystemOpKind::ControlRegister(SysRegOp::ReadModelSpecific),
        mnemonic: "rdmsr".to_string(),
        metadata: None,
        outputs: vec![a],
        inputs: vec![b],
    }));

    // defs == outputs, uses == inputs.
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a]);
    assert!(op.uses().contains(&b), "input must appear in uses");

    // Effects are precise (the machine supplies a sysreg read), never Opaque.
    assert_eq!(op.effects().kind, SsaEffectKind::RuntimeDefined);

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
        SystemOpKind::ControlRegister(SysRegOp::ReadModelSpecific),
        "kind must survive remap"
    );
}

/// **A computation's effects are its kernel's**: a bit deposit is pure, a
/// random draw (`rdrand`) and a random memory tag (`irg`) are what the
/// hardware supplies where control reaches them, while adding to a tag is
/// pure, and the two tag computations keep two fingerprints.
#[test]
fn a_computation_has_its_kernels_effects() {
    let (a, b) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
    let effects = |kernel| whole_compute(kernel, vec![a], vec![b]).effects().kind;
    assert_eq!(effects(Kernel::BitDeposit), SsaEffectKind::Pure);
    assert_eq!(
        effects(Kernel::Random {
            from_entropy: false
        }),
        SsaEffectKind::RuntimeDefined
    );
    assert_eq!(
        effects(Kernel::MemoryTag(MemoryTagKind::Add)),
        SsaEffectKind::Pure
    );
    assert_eq!(
        effects(Kernel::MemoryTag(MemoryTagKind::Random)),
        SsaEffectKind::RuntimeDefined
    );
    assert_ne!(
        Kernel::MemoryTag(MemoryTagKind::Add).kind_str(),
        Kernel::MemoryTag(MemoryTagKind::Random).kind_str()
    );
    assert!(Kernel::identities().contains(&Kernel::MemoryTag(MemoryTagKind::Random)));
}

/// A block compare reads its blocks, count, prior flags and direction flag,
/// defines what it leaves, reads memory and may fault, and survives
/// `remap_variables`; a block load defines its value and next address.
#[test]
fn block_compare_and_load_defs_uses_effects_and_remap() {
    let v = SsaVarId::from_index;
    let compare: SsaOp<MockTarget> = SsaOp::CompareBlk(Box::new(CompareBlkData {
        remaining: v(1),
        left_next: v(2),
        right_next: v(3),
        flags: v(4),
        left: v(5),
        right: v(6),
        count: v(7),
        flags_in: Some(v(8)),
        backward: Some(v(9)),
        element_bytes: 1,
        while_equal: true,
        left_access: AccessQualifiers::FLAT,
        right_access: AccessQualifiers::FLAT,
    }));
    assert_eq!(
        compare.defs().collect::<Vec<_>>(),
        vec![v(1), v(2), v(3), v(4)]
    );
    assert_eq!(compare.uses(), vec![v(5), v(6), v(7), v(8), v(9)]);
    assert_eq!(compare.effects().kind, SsaEffectKind::Read);
    assert!(compare.may_throw());
    let load: SsaOp<MockTarget> = SsaOp::LoadBlk {
        value: v(1),
        next: v(2),
        address: v(5),
        count: v(7),
        value_in: Some(v(8)),
        backward: None,
        element_bytes: 4,
        access: AccessQualifiers::FLAT,
    };
    assert_eq!(load.defs().collect::<Vec<_>>(), vec![v(1), v(2)]);
    assert_eq!(load.uses(), vec![v(5), v(7), v(8)]);
    let remapped = compare.remap_variables(|var| (var == v(9)).then(|| v(19)));
    let SsaOp::CompareBlk(data) = remapped else {
        unreachable!("remap keeps the block compare")
    };
    assert_eq!(data.backward, Some(v(19)));
}

/// `MergeFlags` and `InsertFlag` define their flags value, use both inputs, are
/// pure, and survive `remap_variables` with their mask.
#[test]
fn flag_composition_defs_uses_effects_and_remap() {
    let (dest, base, other) = (
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
        SsaVarId::from_index(3),
    );
    let rename = |v: SsaVarId| Some(SsaVarId::from_index(v.index() + 10));
    for op in [
        SsaOp::<MockTarget>::MergeFlags {
            dest,
            base,
            update: other,
            mask: FlagsMask::ZERO,
        },
        SsaOp::<MockTarget>::InsertFlag {
            dest,
            base,
            bit: other,
            mask: FlagsMask::CARRY,
        },
    ] {
        assert_eq!(op.defs().collect::<Vec<_>>(), vec![dest]);
        assert!(op.uses().contains(&base) && op.uses().contains(&other));
        assert_eq!(op.effects().kind, SsaEffectKind::Pure);
        match op.remap_variables(rename) {
            SsaOp::MergeFlags {
                dest,
                base,
                update,
                mask,
            } => {
                assert_eq!((dest.index(), base.index(), update.index()), (11, 12, 13));
                assert_eq!(mask, FlagsMask::ZERO);
            }
            SsaOp::InsertFlag {
                dest,
                base,
                bit,
                mask,
            } => {
                assert_eq!((dest.index(), base.index(), bit.index()), (11, 12, 13));
                assert_eq!(mask, FlagsMask::CARRY);
            }
            other => unreachable!("remap changed the variant: {other:?}"),
        }
    }
}

/// A typed `CallClobber` defines all its outputs, has no uses, is pure (so
/// dead clobbers are eliminable), and survives `remap_variables`.
#[test]
fn call_clobber_defs_uses_effects_and_remap() {
    let a = SsaVarId::from_index(1);
    let b = SsaVarId::from_index(2);
    let op: SsaOp<MockTarget> = SsaOp::CallClobber {
        outputs: vec![a, b],
        machine: None,
    };
    assert_eq!(op.defs().collect::<Vec<_>>(), vec![a, b]);
    assert!(op.uses().is_empty());
    // What the call before it leaves in its registers: unread, it goes, but
    // it never moves away from that call (out of a loop, past another call).
    assert_eq!(op.effects().kind, SsaEffectKind::RuntimeDefined);
    assert!(op.effects().removable_when_unused());
    assert!(!op.effects().is_pure());

    let remapped = op.remap_variables(|v| match v {
        v if v == a => Some(SsaVarId::from_index(11)),
        v if v == b => Some(SsaVarId::from_index(12)),
        _ => None,
    });
    let SsaOp::CallClobber { outputs, .. } = &remapped else {
        unreachable!("remap must preserve the CallClobber variant")
    };
    assert_eq!(
        outputs,
        &vec![SsaVarId::from_index(11), SsaVarId::from_index(12)]
    );
}

/// An operation with more definitions than [`SsaDefs`] holds in its named
/// slots reports every one of them, in order.
///
/// No operation the crate builds reaches the spill, so nothing else covers it;
/// a host that lifts a native instruction writing four registers does.
#[test]
fn defs_past_the_named_slots_are_all_reported() {
    let outputs: Vec<SsaVarId> = (40..45).map(SsaVarId::from_index).collect();
    let op: SsaOp<MockTarget> = SsaOp::CallClobber {
        outputs: outputs.clone(),
        machine: None,
    };
    assert_eq!(op.defs().collect::<Vec<_>>(), outputs);
    assert_eq!(op.defs().count(), outputs.len());
    assert_eq!(op.defs().len(), outputs.len());
}

/// An operation that defines nothing renders no assignment prefix.
///
/// `CallClobber` with an empty output list is the case that reaches this:
/// a clobber of no registers is still a well-formed op, and a bare `" = "`
/// in front of its mnemonic reads as an assignment with a missing left-hand
/// side. The shared `write_defs` prefix answers it for every op at once.
#[test]
fn an_op_defining_nothing_renders_no_assignment_prefix() {
    let op: SsaOp<MockTarget> = SsaOp::CallClobber {
        outputs: Vec::new(),
        machine: None,
    };
    assert_eq!(format!("{op}"), "call.clobber");
    assert_eq!(op.defs().count(), 0);
}

/// Payload accessors report arithmetic signedness, compare kind, and memory effect,
/// returning `None` where each does not apply.
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
        signaling: false,
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
        access: AccessQualifiers::FLAT,
    };
    let effect = load.memory_effect().expect("load has a memory effect");
    assert_eq!(effect.addr, a);
    assert!(effect.reads);
    assert!(!effect.writes);
    assert_eq!(effect.value_type, Some(&MockType::I32));

    let rmw: SsaOp<MockTarget> = SsaOp::AtomicLockRmw {
        dest: d,
        addr: a,
        value: b,
        op: AtomicRmwOp::Add,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT,
    };
    let effect = rmw.memory_effect().expect("rmw has a memory effect");
    assert!(effect.reads);
    assert!(effect.writes);
    assert_eq!(effect.value_type, None);
}

/// `similarity_class` groups ops into the coarse families a feature fingerprint
/// treats alike.
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
            signaling: false,
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
            access: AccessQualifiers::FLAT,
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
            environment: None,
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
            access: AccessQualifiers::FLAT,
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
}

/// `feature_token` captures an op's target-generic shape and serializes it stably.
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
        access: AccessQualifiers::FLAT.made_volatile(),
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
    use std::mem;

    use super::*;
    use crate::{
        ir::{instruction::SsaInstruction, value::ConstValue},
        testing::MockTarget,
    };

    /// Pins the in-memory size bounds of the core IR value types against silent regrowth.
    #[test]
    fn core_ir_types_stay_compact() {
        assert!(
            mem::size_of::<Option<SsaVarId>>() <= 4,
            "Option<SsaVarId> grew to {} bytes; SsaVarId lost its niche",
            mem::size_of::<Option<SsaVarId>>()
        );
        assert!(
            mem::size_of::<ConstValue<MockTarget>>() <= 24,
            "ConstValue grew to {} bytes; box the new heap-bearing arm",
            mem::size_of::<ConstValue<MockTarget>>()
        );
        // A call is boxed (`CallData`), whose method reference is the
        // largest payload a real host has (analysir's alone is 88 bytes), so
        // the widest operations are the memory accesses: a segment load's
        // destination list and its access qualifiers.
        assert!(
            mem::size_of::<SsaOp<MockTarget>>() <= 48,
            "SsaOp grew to {} bytes; box the new fat variant's payload",
            mem::size_of::<SsaOp<MockTarget>>()
        );
        assert!(
            mem::size_of::<SsaInstruction<MockTarget>>() <= 56,
            "SsaInstruction grew to {} bytes",
            mem::size_of::<SsaInstruction<MockTarget>>()
        );
    }
}

/// Every SSA operand an op *uses* must appear in its rendered form.
///
/// `Display` is the diagnostic surface for the whole IR - it is what pass
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

/// Renders one op's whole observable surface as one golden block.
///
/// Every reader an operand policy has is recorded together: the opcode name,
/// the rendered form, the ordered `(role, variable)` visitor trace, `defs()`,
/// `uses()`, the stack effect, and the result of renaming the first definition
/// through `replace_def` plus the form that renaming produces. A change in any
/// one of them is a one-line diff naming the opcode it belongs to.
fn golden_block(source: &str, index: usize, op: &SsaOp<MockTarget>) -> String {
    let mut trace: Vec<String> = Vec::new();
    op.visit_operands(|role, var| trace.push(format!("{role:?}:v{}", var.index())));
    let defs: Vec<String> = op.defs().map(|v| format!("v{}", v.index())).collect();
    let uses: Vec<String> = op
        .uses()
        .iter()
        .map(|v| format!("v{}", v.index()))
        .collect();
    let (pops, pushes) = op.stack_effect();

    let mut renamed = op.clone();
    let first_def = op.defs().next();
    let changed = first_def.is_some_and(|old| renamed.replace_def(old, GOLDEN_RENAME));

    let mut block = String::new();
    block.push_str(&format!("## {source}[{index}] {}\n", op.opcode_name()));
    block.push_str(&format!("display  {op}\n"));
    block.push_str(&format!("visit    {}\n", join_or_dash(&trace)));
    block.push_str(&format!("defs     {}\n", join_or_dash(&defs)));
    block.push_str(&format!("uses     {}\n", join_or_dash(&uses)));
    block.push_str(&format!("stack    pops={pops} pushes={pushes}\n"));
    block.push_str(&format!("rename   {changed} {renamed}\n"));
    block
}

/// Joins golden list fields, spelling the empty list as `-` so a dropped
/// operand cannot hide as trailing whitespace.
fn join_or_dash(items: &[String]) -> String {
    if items.is_empty() {
        "-".to_string()
    } else {
        items.join(" ")
    }
}

/// The variable the golden renames every first definition to. Outside both the
/// definition space (90, 91) and the use space (80, 81) of `all_sample_ops`, so
/// a rename that lands on the wrong operand is visible in the rendered form.
const GOLDEN_RENAME: SsaVarId = SsaVarId::from_index(99);

/// Renders the whole behaviour golden.
fn golden_text() -> String {
    let mut out = String::from(GOLDEN_HEADER);
    for (index, op) in all_sample_ops().iter().enumerate() {
        out.push('\n');
        out.push_str(&golden_block("sample", index, op));
    }
    for (index, (op, _, _)) in visitor_battery().iter().enumerate() {
        out.push('\n');
        out.push_str(&golden_block("battery", index, op));
    }
    out
}

/// Leading comment of `testdata/ops_golden.txt`, kept with the generator so the
/// file explains itself to whoever first sees it in a diff.
const GOLDEN_HEADER: &str = "\
# Behaviour golden for every `SsaOp` sample.
#
# One block per op: the opcode name, `Display`, the ordered `(role, variable)`
# `visit_operands` trace, `defs()`, `uses()`, `stack_effect()`, and the outcome
# of renaming the first definition to v99 with `replace_def` followed by the
# form that produces. `-` is an empty list.
#
# Operands come from disjoint id spaces: definitions are v90 to v93 and uses
# v80 to v82 in the `sample` blocks, so a transposed role is legible here as well
# as caught by `payload_roles_are_not_transposed`. The `battery` blocks use
# v0..v7 and exist for the operand shapes `all_sample_ops` does not have.
#
# Regenerate with `scripts/update-golden.sh` and review the diff; it is the
# record that a change to the operand matches was behaviour-preserving.
";

/// Opening fence around the generated golden printed on a mismatch.
const GOLDEN_BEGIN: &str = "----- BEGIN ops_golden.txt -----";

/// Closing fence around the generated golden printed on a mismatch.
const GOLDEN_END: &str = "----- END ops_golden.txt -----";

/// Every op's observable behaviour matches the checked-in golden.
///
/// The operand policy -- outputs are definitions, inputs are uses, in that
/// order -- has six readers, and this pins all of them for every variant at
/// once. A restructuring of the matches those readers are built from is
/// behaviour-preserving exactly when this file does not move.
#[test]
fn ops_behaviour_golden() {
    let generated = golden_text();
    let expected = include_str!("testdata/ops_golden.txt");
    if generated != expected {
        // The whole generated golden, fenced so `scripts/update-golden.sh` can
        // lift it out of the test output. The test itself never writes a file.
        println!("{GOLDEN_BEGIN}\n{generated}{GOLDEN_END}");
    }
    assert_eq!(
        generated, expected,
        "op behaviour moved; review the diff and regenerate with scripts/update-golden.sh"
    );
}

/// **A staged value is a use of the call, and no argument of it.** Every
/// operand walk reads it - so it lives as long as the call does - while the
/// arguments do not hold it until [`CallData::promote`] makes it the next one.
/// The call reads it only as staged ([`CallData::only_stages`]) until then,
/// and not where it is an argument or the pointer called through as well.
#[test]
fn a_staged_value_is_a_use_and_no_argument() {
    let (dest, argument, staged) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
    );
    let mut call = CallData::<MockTarget> {
        staged: vec![StagedValue {
            offset: 8,
            value: staged,
        }],
        ..CallData::new(Some(dest), Callee::Direct(3), vec![argument])
    };
    let op = SsaOp::from(call.clone());
    assert_eq!(op.uses(), vec![argument, staged]);
    assert_eq!(
        op.as_call().map(|call| call.args.clone()),
        Some(vec![argument])
    );
    assert!(call.only_stages(staged));
    assert!(!call.only_stages(argument), "an argument is read as one");
    assert!(!call.only_stages(dest), "the result is not staged");
    let passed_too = CallData::<MockTarget> {
        staged: vec![StagedValue {
            offset: 8,
            value: argument,
        }],
        ..CallData::new(Some(dest), Callee::Direct(3), vec![argument])
    };
    assert!(
        !passed_too.only_stages(argument),
        "passed as well as staged"
    );
    let called_through = CallData::<MockTarget> {
        staged: vec![StagedValue {
            offset: 8,
            value: staged,
        }],
        ..CallData::new(
            Some(dest),
            Callee::Indirect {
                fptr: staged,
                signature: 0,
            },
            vec![argument],
        )
    };
    assert!(
        !called_through.only_stages(staged),
        "the call reads its pointer"
    );
    assert_eq!(call.promote(4), None, "nothing is staged at 4");
    assert_eq!(call.promote(8), Some(staged));
    assert_eq!(call.args, vec![argument, staged]);
    assert!(call.staged.is_empty());
    assert!(!call.only_stages(staged), "promoted, it is an argument");
}

/// Every value of a codec-numbered enum, found by offering each `u16` codec
/// number. A variant added to the enum is found without touching this helper.
fn every<T: TryFrom<u16>>() -> Vec<T> {
    (0..=u16::MAX)
        .filter_map(|number| T::try_from(number).ok())
        .collect()
}

/// The IR name of every value of `T`, in codec order.
fn names_of<T: TryFrom<u16>>(name: fn(T) -> &'static str) -> Vec<&'static str> {
    every::<T>().into_iter().map(name).collect()
}

/// The IR names of every shared sub-operation enum, keyed by the enum's name.
///
/// The IR holds all of these enums in one operation namespace, so they are
/// listed together: a name is unique within its enum and across the others.
fn ir_names() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("HintOp", names_of(HintOp::name)),
        ("MachineStateOp", names_of(MachineStateOp::name)),
        ("SysRegOp", names_of(SysRegOp::name)),
        ("CacheMaintenanceOp", names_of(CacheMaintenanceOp::name)),
        ("TlbMaintenanceOp", names_of(TlbMaintenanceOp::name)),
        ("BarrierOp", names_of(BarrierOp::name)),
        ("HardwareEngineOp", names_of(HardwareEngineOp::name)),
        ("HypervisorOp", names_of(HypervisorOp::name)),
        ("TrapOp", names_of(TrapOp::name)),
        ("BreakpointOp", names_of(BreakpointOp::name)),
        ("InterruptReturnOp", names_of(InterruptReturnOp::name)),
        (
            "SystemTransactionKind",
            names_of(SystemTransactionKind::name),
        ),
        ("FlagAdjustKind", names_of(FlagAdjustKind::name)),
        ("BcdAdjustKind", names_of(BcdAdjustKind::name)),
        ("FloatFunctionKind", names_of(FloatFunctionKind::name)),
        ("FpuControlKind", names_of(FpuControlKind::name)),
        ("VectorCryptoKind", names_of(VectorCryptoKind::name)),
        ("AtomicRmwOp", names_of(AtomicRmwOp::name)),
    ]
}

/// Whether `name` matches `^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$`.
fn follows_name_grammar(name: &str) -> bool {
    let word = |word: &str| {
        !word.is_empty()
            && word
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    };
    let mut words = name.split('.');
    let leading = words.next().unwrap_or_default();
    leading.starts_with(|c: char| c.is_ascii_lowercase()) && word(leading) && words.all(word)
}

/// The names that occur more than once in `names`, each reported once.
fn repeated<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut seen = BTreeSet::new();
    let mut repeats = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            repeats.insert(name);
        }
    }
    repeats.into_iter().collect()
}

/// The checks that find a bad name are themselves tested against bad names,
/// and every enum yields values to check.
#[test]
fn ir_name_checks_detect_bad_names() {
    assert_eq!(repeated(["a.b", "c", "a.b", "c", "a.b"]), ["a.b", "c"]);
    assert!(repeated(["a", "b"]).is_empty());
    for good in [
        "nop",
        "hint.spin",
        "sha1.rounds4",
        "port.in.string.8",
        "a_b.c_d",
    ] {
        assert!(follows_name_grammar(good), "{good} is a valid name");
    }
    for bad in [
        "", "Nop", "1a", "_a", "a..b", "a.", ".a", "a-b", "a b", "a.B",
    ] {
        assert!(!follows_name_grammar(bad), "{bad:?} is not a valid name");
    }
    for (kind, names) in ir_names() {
        assert!(!names.is_empty(), "{kind} enumerated no values");
    }
}

/// Every IR name follows the naming grammar.
#[test]
fn ir_names_follow_the_grammar() {
    for (kind, names) in ir_names() {
        for name in names {
            assert!(follows_name_grammar(name), "{kind}: {name:?}");
        }
    }
}

/// No two values of one enum share an IR name: each value is one operation.
#[test]
fn ir_names_are_unique_within_each_enum() {
    for (kind, names) in ir_names() {
        assert_eq!(
            repeated(names.iter().copied()),
            Vec::<&str>::new(),
            "{kind} repeats a name"
        );
    }
}

/// No two values of any two of these enums share an IR name. The IR renders
/// all of them into one operation namespace.
#[test]
fn ir_names_are_unique_across_enums() {
    let every_name = ir_names()
        .into_iter()
        .flat_map(|(_, names)| names)
        .collect::<Vec<_>>();
    assert_eq!(
        repeated(every_name.iter().copied()),
        Vec::<&str>::new(),
        "an IR name is shared between operations"
    );
}

/// The spellings the reference documents are the ones returned.
#[test]
fn ir_names_spell_the_documented_operations() {
    assert_eq!(HintOp::SpinWait.name(), "hint.spin");
    assert_eq!(SystemTransactionKind::Cancel.name(), "txn.abort");
    assert_eq!(FlagAdjustKind::SetCarry.name(), "flags.cf.set");
    assert_eq!(AtomicRmwOp::MinU.name(), "atomic.min.u");
    assert_eq!(FloatFunctionKind::Atan.name(), "atan2");
    assert_eq!(CacheMaintenanceOp::FlushLine.name(), "cache.flush.line");
    assert_eq!(
        CacheMaintenanceOp::WriteBackInvalidateAll.name(),
        "cache.flush.all"
    );
    assert_eq!(TrapOp::TrapIfLessUnsigned.name(), "trap.cmp.ult");
    assert_eq!(HintOp::PrefetchVector.name(), "prefetch.vec");
    assert_eq!(
        MachineStateOp::VendorReadDebugRegister.name(),
        "vendor.read_debug_reg"
    );
}

/// The display names the signedness a vector element states, so `umax` and
/// `smax` are told apart, and states nothing where the element is silent.
#[test]
fn vector_display_names_the_stated_signedness() {
    let (d, a, b, m) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
        SsaVarId::from_index(3),
    );
    let lanes = |signed| VectorElement {
        kind: VectorElementKind::Integer,
        bits: 8,
        scalar: false,
        signed,
        rounding: None,
    };
    let binary = |signed| SsaOp::<MockTarget>::VectorBinary {
        dest: d,
        left: a,
        right: b,
        kind: VectorBinaryKind::SatAdd,
        element: lanes(signed),
        environment: None,
    };
    assert_eq!(
        binary(Some(true)).to_string(),
        "v0 = vbinary.SatAdd.s v1, v2"
    );
    assert_eq!(
        binary(Some(false)).to_string(),
        "v0 = vbinary.SatAdd.u v1, v2"
    );
    assert_eq!(binary(None).to_string(), "v0 = vbinary.SatAdd v1, v2");
    let predicated = SsaOp::<MockTarget>::VectorPredicatedBinary {
        dest: d,
        left: a,
        right: b,
        mask: m,
        passthrough: None,
        kind: VectorBinaryKind::Max,
        element: lanes(Some(false)),
        mode: VectorMaskMode::Merge,
        environment: None,
    };
    assert_eq!(
        predicated.to_string(),
        "v0 = vbinary.pred.Max.u.Merge v1, v2, v3"
    );
    let reduce = SsaOp::<MockTarget>::VectorReduce {
        dest: d,
        value: a,
        kind: VectorReduceKind::Min,
        element: lanes(Some(true)),
        initial: None,
        ordered: false,
        groups: None,
        environment: None,
    };
    assert_eq!(reduce.to_string(), "v0 = vreduce.Min.s v1");
    let ordered = SsaOp::<MockTarget>::VectorReduce {
        dest: d,
        value: a,
        kind: VectorReduceKind::Add,
        element: VectorElement::float(64),
        initial: Some(b),
        ordered: true,
        groups: None,
        environment: None,
    };
    assert_eq!(ordered.to_string(), "v0 = vreduce.Add.ordered v1 from v2");
}

/// **A stated rounding is part of a vector operation's text**: the rule's
/// name after the kind, and nothing where the operation rounds by the
/// environment's mode.
#[test]
fn vector_display_names_the_stated_rounding() {
    let (d, a, b, c) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
        SsaVarId::from_index(3),
    );
    let element = VectorElement::float(32);
    let up = element.rounded(FloatRounding::Up);
    let binary = |element| SsaOp::<MockTarget>::VectorBinary {
        dest: d,
        left: a,
        right: b,
        kind: VectorBinaryKind::Add,
        element,
        environment: None,
    };
    assert_eq!(binary(up).to_string(), "v0 = vbinary.Add.up v1, v2");
    assert_eq!(binary(element).to_string(), "v0 = vbinary.Add v1, v2");
    let sqrt = SsaOp::<MockTarget>::VectorUnary {
        dest: d,
        value: a,
        kind: VectorUnaryKind::Sqrt,
        element: element.rounded(FloatRounding::TowardZero),
        environment: None,
    };
    assert_eq!(sqrt.to_string(), "v0 = vunary.Sqrt.tozero v1");
    let fma = SsaOp::<MockTarget>::VectorTernary {
        dest: d,
        first: a,
        second: b,
        third: c,
        kind: VectorTernaryKind::Fma,
        element: element.rounded(FloatRounding::Down),
        environment: None,
    };
    assert_eq!(fma.to_string(), "v0 = vternary.Fma.down v1, v2, v3");
    let reduce = SsaOp::<MockTarget>::VectorReduce {
        dest: d,
        value: a,
        kind: VectorReduceKind::Add,
        element: up,
        initial: None,
        ordered: true,
        groups: None,
        environment: None,
    };
    assert_eq!(reduce.to_string(), "v0 = vreduce.Add.up.ordered v1");
}

/// **A reduction's initial value is one of its operands**: it is read
/// before the vector, as it combines first, and a rewrite of the value it
/// reads rewrites it.
#[test]
fn a_reductions_initial_value_is_an_operand() {
    let [dest, vector, initial, other] = [0, 1, 2, 3].map(SsaVarId::from_index);
    let mut reduce = SsaOp::<MockTarget>::VectorReduce {
        dest,
        value: vector,
        kind: VectorReduceKind::Add,
        element: VectorElement::float(64),
        initial: Some(initial),
        ordered: true,
        groups: None,
        environment: None,
    };
    assert_eq!(reduce.uses(), vec![initial, vector]);
    assert_eq!(reduce.replace_uses(initial, other), 1);
    assert!(matches!(
        reduce,
        SsaOp::VectorReduce {
            initial: Some(replaced),
            ..
        } if replaced == other
    ));
}

/// An operation keeps the signedness its element states only where its result
/// depends on it, and never on float lanes.
#[test]
fn unread_signedness_is_forgotten() {
    let (d, a, b) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
    );
    let lanes = |kind| VectorElement {
        kind,
        bits: 16,
        scalar: false,
        signed: Some(true),
        rounding: None,
    };
    let stated = |kind, lane_kind| {
        let mut op = SsaOp::<MockTarget>::VectorBinary {
            dest: d,
            left: a,
            right: b,
            kind,
            element: lanes(lane_kind),
            environment: None,
        };
        op.forget_unread_signedness();
        match op {
            SsaOp::VectorBinary { element, .. } => element.signed,
            _ => unreachable!(),
        }
    };
    for kind in [
        VectorBinaryKind::Div,
        VectorBinaryKind::Rem,
        VectorBinaryKind::Min,
        VectorBinaryKind::Max,
        VectorBinaryKind::Shr,
        VectorBinaryKind::MulHigh,
        VectorBinaryKind::MulHighRound,
        VectorBinaryKind::Avg,
        VectorBinaryKind::AbsDiff,
        VectorBinaryKind::SatAdd,
        VectorBinaryKind::SatSub,
    ] {
        assert_eq!(
            stated(kind, VectorElementKind::Integer),
            Some(true),
            "{kind:?}"
        );
        assert_eq!(
            stated(kind, VectorElementKind::Unknown),
            Some(true),
            "{kind:?}"
        );
        assert_eq!(stated(kind, VectorElementKind::Float), None, "{kind:?}");
    }
    for kind in [
        VectorBinaryKind::Add,
        VectorBinaryKind::Sub,
        VectorBinaryKind::Mul,
        VectorBinaryKind::Shl,
        VectorBinaryKind::And,
    ] {
        assert_eq!(stated(kind, VectorElementKind::Integer), None, "{kind:?}");
    }
    let mut reduce = SsaOp::<MockTarget>::VectorReduce {
        dest: d,
        value: a,
        kind: VectorReduceKind::Add,
        element: lanes(VectorElementKind::Integer),
        initial: None,
        ordered: false,
        groups: None,
        environment: None,
    };
    reduce.forget_unread_signedness();
    assert!(matches!(
        reduce,
        SsaOp::VectorReduce {
            element: VectorElement { signed: None, .. },
            ..
        }
    ));
    // A sign reads it: an unsigned lane is never negative. A negation does
    // not.
    let unary = |kind| {
        let mut op = SsaOp::<MockTarget>::VectorUnary {
            dest: d,
            value: a,
            kind,
            element: lanes(VectorElementKind::Integer),
            environment: None,
        };
        op.forget_unread_signedness();
        match op {
            SsaOp::VectorUnary { element, .. } => element.signed,
            _ => unreachable!(),
        }
    };
    assert_eq!(unary(VectorUnaryKind::Sign), Some(true));
    assert_eq!(unary(VectorUnaryKind::Neg), None);
    // A grouped sum extends each lane, so it keeps reading the sign.
    let mut grouped = SsaOp::<MockTarget>::VectorReduce {
        dest: d,
        value: a,
        kind: VectorReduceKind::Add,
        element: lanes(VectorElementKind::Integer),
        initial: None,
        ordered: false,
        groups: Some(ReduceGroups { lanes: 2, bits: 64 }),
        environment: None,
    };
    grouped.forget_unread_signedness();
    assert!(matches!(
        grouped,
        SsaOp::VectorReduce {
            element: VectorElement {
                signed: Some(true),
                ..
            },
            ..
        }
    ));
    let mut reduce = SsaOp::<MockTarget>::VectorReduce {
        dest: d,
        value: a,
        kind: VectorReduceKind::Max,
        element: lanes(VectorElementKind::Integer),
        initial: None,
        ordered: false,
        groups: None,
        environment: None,
    };
    reduce.forget_unread_signedness();
    assert!(matches!(
        reduce,
        SsaOp::VectorReduce {
            element: VectorElement {
                signed: Some(true),
                ..
            },
            ..
        }
    ));
    let mut unary = SsaOp::<MockTarget>::VectorUnary {
        dest: d,
        value: a,
        kind: VectorUnaryKind::Abs,
        element: lanes(VectorElementKind::Integer),
        environment: None,
    };
    unary.forget_unread_signedness();
    assert!(matches!(
        unary,
        SsaOp::VectorUnary {
            element: VectorElement { signed: None, .. },
            ..
        }
    ));
}

/// Runtime vector queries are pure, while committing configuration is a
/// non-removable machine write that can trap without touching memory.
#[test]
fn vector_configuration_queries_and_commit_have_precise_effects() {
    let output = SsaVarId::from_index(0);
    for (kind, count) in [
        (Kernel::VectorShapeSupported, 3),
        (Kernel::VectorLengthChoice, 2),
        (Kernel::VectorFloatSupported, 2),
        (Kernel::VectorCryptoSupported, 2),
    ] {
        let inputs: Vec<_> = (1..=count).map(SsaVarId::from_index).collect();
        let op = whole_compute(kind, vec![output], inputs);
        assert!(op.is_pure());
        assert_eq!(op.uses().len(), count);
        assert_eq!(op.defs().collect::<Vec<_>>(), [output]);
        assert_eq!(op.effects().memory, MemoryEffectLocation::None);
        assert!(Kernel::identities().contains(&kind));
    }
    let kind = MachineStateOp::WriteVectorConfiguration;
    assert!(MachineStateOp::all().any(|value| value == kind));
    assert_eq!(kind.effects().kind, SsaEffectKind::Write);
    assert_eq!(kind.effects().memory, MemoryEffectLocation::None);
    assert!(kind.effects().may_throw);
    assert!(!kind.effects().is_pure());
    assert!(SystemOpKind::identities().contains(&SystemOpKind::MachineState(kind)));
    let progress = MachineStateOp::WriteVectorProgress;
    assert!(MachineStateOp::all().any(|value| value == progress));
    assert_eq!(progress.effects().kind, SsaEffectKind::Write);
    assert_eq!(progress.effects().memory, MemoryEffectLocation::None);
    assert!(!progress.effects().may_throw);
    assert!(!progress.effects().removable_when_unused());
}

/// Arm's AES round with mix-columns adds the round key first, then substitutes,
/// shifts and mixes: FIPS-197's Appendix B round 1, from the plaintext and the
/// cipher key to the state after MixColumns.
#[test]
fn an_arm_aes_round_with_mix_columns_adds_the_key_first() {
    let bytes = |hex: &str| {
        let mut out = [0u8; 16];
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
        }
        u128::from_le_bytes(out)
    };
    let plaintext = bytes("3243f6a8885a308d313198a2e0370734");
    let key = bytes("2b7e151628aed2a6abf7158809cf4f3c");
    let mixed = bytes("046681e5e0cb199a48f8d37a2806264c");
    assert_eq!(
        VectorCryptoKind::AesEncryptRoundMix.aes_round(plaintext, key),
        Some(mixed)
    );
}

/// Each SHA and SM3 step of a register computes the step `ir::sha` or
/// `ir::sm3` states of its inputs in the order the lifts state them: the
/// register an instruction accumulates into first, `sha1rnds4`'s round
/// group, `vsm3rnds2`'s round and `sha256rnds2`'s message words (`xmm0`)
/// last, Arm's `sha1c` the hash, E and the words, and `sm3tt1a` its
/// registers and the element of the last it names. SHA-512's steps read
/// 64-bit words of registers of any lane layout.
#[test]
fn each_hash_step_computes_its_step_of_its_inputs_in_order() {
    let wide = |words: &[u64]| {
        VectorConst::from_lanes(64, words.iter().map(|&word| u128::from(word))).unwrap()
    };
    let words = |base: u32| -> [u32; 4] {
        std::array::from_fn(|index| base.wrapping_mul(0x9e37_79b9).rotate_left(index as u32 * 7))
    };
    let vector = |words: [u32; 4]| VectorConst::from_lanes(32, words.map(u128::from)).unwrap();
    let (a, b, c) = (words(1), words(2), words(3));
    let scalar = |value: u32| VectorConst::from_lanes(32, [u128::from(value)]).unwrap();
    let byte = |value: u8| VectorConst::from_lanes(8, [u128::from(value)]).unwrap();
    // SHA-512's registers of 64-bit words, given as the 32-bit words of the
    // same bits: a step reads its words of any lane layout.
    let pair = |words: [u32; 4]| -> [u64; 2] {
        let [w0, w1, w2, w3] = words.map(u64::from);
        [w0 | w1 << 32, w2 | w3 << 32]
    };
    let quad = |low: [u32; 4], high: [u32; 4]| -> [u64; 4] {
        let ([w0, w1], [w2, w3]) = (pair(low), pair(high));
        [w0, w1, w2, w3]
    };
    let pairs = vector;
    let quads = |low: [u32; 4], high: [u32; 4]| {
        VectorConst::from_lanes(32, low.into_iter().chain(high).map(u128::from)).unwrap()
    };
    for (kind, inputs, expected) in [
        (
            VectorCryptoKind::Sha1Rounds4,
            vec![vector(a), vector(b), byte(2)],
            vector(sha::sha1_rounds4(a, b, 2)),
        ),
        (
            VectorCryptoKind::Sha1NextE,
            vec![vector(a), vector(b)],
            vector(sha::sha1_next_e(a, b)),
        ),
        (
            VectorCryptoKind::Sha1Msg1,
            vec![vector(a), vector(b)],
            vector(sha::sha1_message1(a, b)),
        ),
        (
            VectorCryptoKind::Sha1Msg2,
            vec![vector(a), vector(b)],
            vector(sha::sha1_message2(a, b)),
        ),
        (
            VectorCryptoKind::Sha256Rounds2,
            vec![vector(a), vector(b), vector(c)],
            vector(sha::sha2_rounds2(a, b, [c[0], c[1]])),
        ),
        (
            VectorCryptoKind::Sha256Msg1,
            vec![vector(a), vector(b)],
            vector(sha::sha2_message1(a, b[0])),
        ),
        (
            VectorCryptoKind::Sha256Msg2,
            vec![vector(a), vector(b)],
            vector(sha::sha2_message2(a, b)),
        ),
        (
            VectorCryptoKind::Sha512Rounds2,
            vec![quads(a, b), quads(b, c), pairs(c)],
            wide(&sha::sha2_rounds2(quad(a, b), quad(b, c), pair(c))),
        ),
        (
            VectorCryptoKind::Sha512Msg1,
            vec![quads(a, b), pairs(c)],
            wide(&sha::sha2_message1(quad(a, b), pair(c)[0])),
        ),
        (
            VectorCryptoKind::Sha512Msg2,
            vec![quads(a, b), quads(b, c)],
            wide(&sha::sha2_message2(quad(a, b), quad(b, c))),
        ),
        (
            VectorCryptoKind::Sha512Hash,
            vec![pairs(a), pairs(b), pairs(c)],
            wide(&sha::arm_sha512_hash(pair(a), pair(b), pair(c))),
        ),
        (
            VectorCryptoKind::Sha512Hash2,
            vec![pairs(a), pairs(b), pairs(c)],
            wide(&sha::arm_sha512_hash2(pair(a), pair(b), pair(c))),
        ),
        (
            VectorCryptoKind::Sha512ScheduleUpdate0,
            vec![pairs(a), pairs(b)],
            wide(&sha::arm_sha512_schedule0(pair(a), pair(b))),
        ),
        (
            VectorCryptoKind::Sha512ScheduleUpdate1,
            vec![pairs(a), pairs(b), pairs(c)],
            wide(&sha::arm_sha512_schedule1(pair(a), pair(b), pair(c))),
        ),
        (
            VectorCryptoKind::Sha1HashChoose,
            vec![vector(a), scalar(7), vector(c)],
            vector(sha::arm_sha1_hash(a, 7, c, 0)),
        ),
        (
            VectorCryptoKind::Sha1HashParity,
            vec![vector(a), scalar(7), vector(c)],
            vector(sha::arm_sha1_hash(a, 7, c, 1)),
        ),
        (
            VectorCryptoKind::Sha1HashMajority,
            vec![vector(a), scalar(7), vector(c)],
            vector(sha::arm_sha1_hash(a, 7, c, 2)),
        ),
        (
            VectorCryptoKind::Sha1FixedRotate,
            vec![scalar(0x8000_0001)],
            scalar(sha::arm_sha1_fixed_rotate(0x8000_0001)),
        ),
        (
            VectorCryptoKind::Sha1ScheduleUpdate0,
            vec![vector(a), vector(b), vector(c)],
            vector(sha::arm_sha1_schedule0(a, b, c)),
        ),
        (
            VectorCryptoKind::Sha1ScheduleUpdate1,
            vec![vector(a), vector(b)],
            vector(sha::arm_sha1_schedule1(a, b)),
        ),
        (
            VectorCryptoKind::Sha256Hash,
            vec![vector(a), vector(b), vector(c)],
            vector(sha::arm_sha256_hash(a, b, c, true)),
        ),
        (
            VectorCryptoKind::Sha256Hash2,
            vec![vector(a), vector(b), vector(c)],
            vector(sha::arm_sha256_hash(a, b, c, false)),
        ),
        (
            VectorCryptoKind::Sha256ScheduleUpdate0,
            vec![vector(a), vector(b)],
            vector(sha::arm_sha256_schedule0(a, b)),
        ),
        (
            VectorCryptoKind::Sha256ScheduleUpdate1,
            vec![vector(a), vector(b), vector(c)],
            vector(sha::arm_sha256_schedule1(a, b, c)),
        ),
        (
            VectorCryptoKind::Sm3Msg1,
            vec![vector(a), vector(b), vector(c)],
            vector(sm3::sm3_message1(a, b, c)),
        ),
        (
            VectorCryptoKind::Sm3Msg2,
            vec![vector(a), vector(b), vector(c)],
            vector(sm3::sm3_message2(a, b, c)),
        ),
        (
            VectorCryptoKind::Sm3Rounds2,
            vec![vector(a), vector(b), vector(c), byte(18)],
            vector(sm3::sm3_rounds2(a, b, c, 18)),
        ),
        (
            VectorCryptoKind::Sm3PartW1,
            vec![vector(a), vector(b), vector(c)],
            vector(sm3::arm_sm3_partw1(a, b, c)),
        ),
        (
            VectorCryptoKind::Sm3PartW2,
            vec![vector(a), vector(b), vector(c)],
            vector(sm3::arm_sm3_partw2(a, b, c)),
        ),
        (
            VectorCryptoKind::Sm3SS1,
            vec![vector(a), vector(b), vector(c)],
            vector(sm3::arm_sm3_ss1(a, b, c)),
        ),
        (
            VectorCryptoKind::Sm3TT1A,
            vec![vector(a), vector(b), vector(c), byte(2)],
            vector(sm3::arm_sm3_tt1(a, b, c[2], false)),
        ),
        (
            VectorCryptoKind::Sm3TT1B,
            vec![vector(a), vector(b), vector(c), byte(1)],
            vector(sm3::arm_sm3_tt1(a, b, c[1], true)),
        ),
        (
            VectorCryptoKind::Sm3TT2A,
            vec![vector(a), vector(b), vector(c), byte(3)],
            vector(sm3::arm_sm3_tt2(a, b, c[3], false)),
        ),
        (
            VectorCryptoKind::Sm3TT2B,
            vec![vector(a), vector(b), vector(c), byte(0)],
            vector(sm3::arm_sm3_tt2(a, b, c[0], true)),
        ),
    ] {
        assert_eq!(kind.evaluate(&inputs, 0), Some(vec![expected]), "{kind:?}");
    }
}

/// Each SM4 step computes four rounds of each 128-bit block of its state by
/// the block's keys: `vsm4rnds4 ymm` two blocks, each its own, by the cipher's
/// transform, `vsm4key4` by the key schedule's, of registers of any lane
/// layout and with the result laid out as the state is.
#[test]
fn each_sm4_step_computes_four_rounds_of_each_block() {
    let words = |base: u32| -> [u32; 4] {
        std::array::from_fn(|index| base.wrapping_mul(0x9e37_79b9).rotate_left(index as u32 * 5))
    };
    let (low_state, high_state, low_keys, high_keys) = (words(1), words(2), words(3), words(4));
    let register = |low: [u32; 4], high: [u32; 4]| {
        VectorConst::from_lanes(32, low.into_iter().chain(high).map(u128::from)).unwrap()
    };
    for (kind, transform) in [
        (VectorCryptoKind::Sm4Rounds, sm4::Transform::Cipher),
        (VectorCryptoKind::Sm4Key, sm4::Transform::Key),
    ] {
        let expected = register(
            transform.rounds4(low_state, low_keys).unwrap(),
            transform.rounds4(high_state, high_keys).unwrap(),
        );
        let keys = register(low_keys, high_keys).relaned(64).unwrap();
        assert_eq!(
            kind.evaluate(&[register(low_state, high_state), keys], 32),
            Some(vec![expected]),
            "{kind:?}"
        );
    }
}

/// Each RISC-V element-group step computes its step of each group of its
/// inputs, in the order the lift states them: two groups of a register
/// group each their own (SHA-2's of 32-bit and of 64-bit elements, SM3's of
/// 256 bits), the round an immediate after the registers, and a `.vs` form's
/// one group serving every group of the state.
#[test]
fn each_riscv_group_step_computes_each_group_of_its_inputs() {
    let words = |base: u32| -> [u32; 4] {
        std::array::from_fn(|index| base.wrapping_mul(0x9e37_79b9).rotate_left(index as u32 * 3))
    };
    let eight = |base: u32| -> [u32; 8] {
        let ([a, b, c, d], [e, f, g, h]) = (words(base), words(base + 7));
        [a, b, c, d, e, f, g, h]
    };
    let wide = |base: u32| -> [u64; 4] {
        let ([a, b, c, d], [e, f, g, h]) = (words(base), words(base + 7));
        [(a, b), (c, d), (e, f), (g, h)].map(|(low, high)| u64::from(low) | u64::from(high) << 32)
    };
    let register = |groups: &[&[u32]]| {
        VectorConst::from_lanes(
            32,
            groups
                .iter()
                .flat_map(|group| group.iter())
                .map(|&word| u128::from(word)),
        )
        .unwrap()
    };
    let doubles = |groups: &[[u64; 4]]| {
        VectorConst::from_lanes(64, groups.iter().flatten().map(|&word| u128::from(word))).unwrap()
    };
    let byte = |value: u8| VectorConst::from_lanes(8, [u128::from(value)]).unwrap();
    let block = |words: [u32; 4]| {
        words
            .iter()
            .rev()
            .fold(0u128, |block, word| block << 32 | u128::from(*word))
    };
    let unblock = |block: u128| -> [u32; 4] {
        std::array::from_fn(|index| u32::try_from(block >> (32 * index) & 0xffff_ffff).unwrap())
    };
    let (a, b, c, d, e, f) = (words(1), words(2), words(3), words(4), words(5), words(6));
    let mut cases: Vec<(VectorCryptoKind, Vec<VectorConst>, u32, VectorConst)> = vec![
        (
            VectorCryptoKind::AesExpand128,
            vec![register(&[&a, &b]), byte(3)],
            32,
            register(&[&aes::expand128(a, 3), &aes::expand128(b, 3)]),
        ),
        (
            VectorCryptoKind::AesExpand256,
            vec![register(&[&a, &b]), register(&[&c, &d]), byte(6)],
            32,
            register(&[&aes::expand256(a, c, 6), &aes::expand256(b, d, 6)]),
        ),
        (
            VectorCryptoKind::Sm4Expand4,
            vec![register(&[&a, &b]), byte(5)],
            32,
            register(&[
                &sm4::Transform::expand4(a, 5).unwrap(),
                &sm4::Transform::expand4(b, 5).unwrap(),
            ]),
        ),
        (
            VectorCryptoKind::Sm4Rounds,
            vec![register(&[&a, &b]), register(&[&c])],
            32,
            register(&[
                &sm4::Transform::Cipher.rounds4(a, c).unwrap(),
                &sm4::Transform::Cipher.rounds4(b, c).unwrap(),
            ]),
        ),
        (
            VectorCryptoKind::AesEncrypt,
            vec![register(&[&a, &b]), register(&[&c])],
            32,
            register(&[
                &unblock(aes::round(false, false, false, block(a), block(c)).unwrap()),
                &unblock(aes::round(false, false, false, block(b), block(c)).unwrap()),
            ]),
        ),
        (
            VectorCryptoKind::GhashMultiply,
            vec![register(&[&a, &b]), register(&[&c, &d])],
            32,
            register(&[
                &unblock(ghash::multiply(block(a), block(c))),
                &unblock(ghash::multiply(block(b), block(d))),
            ]),
        ),
        (
            VectorCryptoKind::GhashMultiplyAdd,
            vec![
                register(&[&a, &b]),
                register(&[&c, &d]),
                register(&[&e, &f]),
            ],
            32,
            register(&[
                &unblock(ghash::multiply_add(block(a), block(c), block(e))),
                &unblock(ghash::multiply_add(block(b), block(d), block(f))),
            ]),
        ),
        (
            VectorCryptoKind::Sm3Expand8,
            vec![
                register(&[&eight(1), &eight(2)]),
                register(&[&eight(3), &eight(4)]),
            ],
            32,
            register(&[
                &sm3::riscv_sm3_expand8(eight(1), eight(3)),
                &sm3::riscv_sm3_expand8(eight(2), eight(4)),
            ]),
        ),
        (
            VectorCryptoKind::Sm3Compress2,
            vec![
                register(&[&eight(1), &eight(2)]),
                register(&[&eight(3), &eight(4)]),
                byte(9),
            ],
            32,
            register(&[
                &sm3::riscv_sm3_compress2(eight(1), eight(3), 9),
                &sm3::riscv_sm3_compress2(eight(2), eight(4), 9),
            ]),
        ),
    ];
    cases.push((
        VectorCryptoKind::Sha2Schedule4,
        vec![
            register(&[&a, &b]),
            register(&[&c, &d]),
            register(&[&e, &f]),
        ],
        32,
        register(&[&sha::sha2_schedule4(a, c, e), &sha::sha2_schedule4(b, d, f)]),
    ));
    for (kind, pair) in [
        (VectorCryptoKind::Sha2Compress2Low, [0, 1]),
        (VectorCryptoKind::Sha2Compress2High, [2, 3]),
    ] {
        let wk = |words: [u32; 4]| [words[pair[0]], words[pair[1]]];
        cases.push((
            kind,
            vec![
                register(&[&a, &b]),
                register(&[&c, &d]),
                register(&[&e, &f]),
            ],
            32,
            register(&[
                &sha::sha2_rounds2(a, c, wk(e)),
                &sha::sha2_rounds2(b, d, wk(f)),
            ]),
        ));
        let wk = |words: [u64; 4]| [words[pair[0]], words[pair[1]]];
        cases.push((
            kind,
            vec![
                doubles(&[wide(1), wide(2)]),
                doubles(&[wide(3), wide(4)]),
                doubles(&[wide(5), wide(6)]),
            ],
            64,
            doubles(&[
                sha::sha2_rounds2(wide(1), wide(3), wk(wide(5))),
                sha::sha2_rounds2(wide(2), wide(4), wk(wide(6))),
            ]),
        ));
    }
    for (kind, inputs, element_bits, expected) in cases {
        assert_eq!(
            kind.evaluate(&inputs, element_bits),
            Some(vec![expected]),
            "{kind:?} of {element_bits}-bit elements"
        );
    }
}

/// The AES and GF(2^8) steps compute what the hardware does. The x86 rounds,
/// `aesimc` and `aeskeygenassist` are those AES-NI gives on a Broadwell; the
/// AArch64 rounds, AddRoundKey then the substitution and the row shift, are
/// `aesenclast` and `aesdeclast` of the keyed state under a zero key there,
/// and `aesmc` is the inverse `aesimc` undoes. A 256-bit register is two
/// blocks, each its own. `gf2p8mulb` gives FIPS 197's products `{57}{83} =
/// {c1}` and `{53}{ca} = {01}`, `gf2p8affineinvqb` by the AES matrix and
/// `0x63` the AES substitution, and `gf2p8affineqb` by the identity matrix
/// each byte plus the constant.
#[test]
fn the_aes_and_gf2p8_steps_compute_what_the_hardware_does() {
    let block = |high: u128, low: u128| VectorConst::from_lanes(128, [high << 64 | low]).unwrap();
    let state = block(0x0011_2233_4455_6677, 0x8899_aabb_ccdd_eeff);
    let key = block(0x0f1e_2d3c_4b5a_6978, 0x8796_a5b4_c3d2_e1f0);
    let fips = block(0x2b7e_1516_28ae_d2a6, 0xabf7_1588_09cf_4f3c);
    let byte = |value: u8| VectorConst::from_lanes(8, [u128::from(value)]).unwrap();
    let one = |kind: VectorCryptoKind, inputs: &[VectorConst]| {
        kind.evaluate(inputs, 0)
            .map(|results| results.into_iter().next().unwrap())
    };
    for (kind, inputs, high, low) in [
        (
            VectorCryptoKind::AesEncrypt,
            vec![state.clone(), key.clone()],
            0xa11f_8c2c_8ef2_3d22,
            0x858d_ffe7_6f2e_c8ac,
        ),
        (
            VectorCryptoKind::AesEncryptLast,
            vec![state.clone(), key.clone()],
            0x14f0_05ff_8f9b_fa8d,
            0xcc14_965e_a02e_4de6,
        ),
        (
            VectorCryptoKind::AesDecrypt,
            vec![state.clone(), key.clone()],
            0x4914_497f_1bdd_e4d9,
            0x9fd9_a6e2_f0da_ec55,
        ),
        (
            VectorCryptoKind::AesDecryptLast,
            vec![state.clone(), key.clone()],
            0x28e7_fe5a_1993_0b7a,
            0x0175_3c4a_543f_758d,
        ),
        (
            VectorCryptoKind::AesInvMixColumns,
            vec![state.clone()],
            0xeebb_cc99_aaff_88dd,
            0x6633_4411_2277_0055,
        ),
        (
            VectorCryptoKind::AesKeygenAssist,
            vec![key.clone(), byte(0x01)],
            0xeb76_72d9_7672_d8eb,
            0x8d17_9007_1790_068d,
        ),
        (
            VectorCryptoKind::AesKeygenAssist,
            vec![key.clone(), byte(0x1b)],
            0xeb76_72c3_7672_d8eb,
            0x8d17_901d_1790_068d,
        ),
        (
            VectorCryptoKind::AesKeygenAssist,
            vec![key.clone(), byte(0x36)],
            0xeb76_72ee_7672_d8eb,
            0x8d17_9030_1790_068d,
        ),
        (
            VectorCryptoKind::AesKeygenAssist,
            vec![key.clone(), byte(0xff)],
            0xeb76_7227_7672_d8eb,
            0x8d17_90f9_1790_068d,
        ),
        (
            VectorCryptoKind::AesEncryptRound,
            vec![state.clone(), fips.clone()],
            0x509f_323f_26c9_9a3e,
            0xa6a8_8dc3_f10f_082e,
        ),
        (
            VectorCryptoKind::AesDecryptRound,
            vec![state.clone(), fips.clone()],
            0x0745_c6c2_0b39_f451,
            0xb806_f166_3263_b233,
        ),
    ] {
        assert_eq!(one(kind, &inputs), Some(block(high, low)), "{kind:?}");
    }
    let mixed = one(
        VectorCryptoKind::AesMixColumns,
        std::slice::from_ref(&state),
    )
    .unwrap();
    assert_eq!(
        one(VectorCryptoKind::AesInvMixColumns, &[mixed]),
        Some(state.clone())
    );
    // Two blocks: each encrypted by its own key.
    let pair = |first: &VectorConst, second: &VectorConst| {
        VectorConst::from_lanes(128, [first.lane(0), second.lane(0)]).unwrap()
    };
    let encrypted = one(
        VectorCryptoKind::AesEncrypt,
        &[pair(&state, &fips), pair(&key, &key)],
    )
    .unwrap();
    assert_eq!(
        encrypted.lane(0),
        one(VectorCryptoKind::AesEncrypt, &[state.clone(), key.clone()])
            .unwrap()
            .lane(0)
    );
    assert_eq!(
        encrypted.lane(1),
        one(VectorCryptoKind::AesEncrypt, &[fips.clone(), key.clone()])
            .unwrap()
            .lane(0)
    );
    // GF(2^8).
    let bytes = |values: [u8; 16]| VectorConst::from_lanes(8, values.map(u128::from)).unwrap();
    let mut left = [0u8; 16];
    let mut right = [0u8; 16];
    left[0] = 0x57;
    right[0] = 0x83;
    left[5] = 0x53;
    right[5] = 0xca;
    let mut product = [0u8; 16];
    product[0] = 0xc1;
    product[5] = 0x01;
    assert_eq!(
        one(VectorCryptoKind::Gf2p8Mul, &[bytes(left), bytes(right)]),
        Some(bytes(product))
    );
    let values: [u8; 16] =
        std::array::from_fn(|index| (index as u8).wrapping_mul(29).wrapping_add(3));
    let matrix =
        |word: u64| VectorConst::from_lanes(64, [u128::from(word), u128::from(word)]).unwrap();
    assert_eq!(
        one(
            VectorCryptoKind::Gf2p8AffineInv,
            &[bytes(values), matrix(0xf1e3_c78f_1f3e_7cf8), byte(0x63)]
        ),
        Some(bytes(values.map(crate::ir::aes::substitute)))
    );
    assert_eq!(
        one(
            VectorCryptoKind::Gf2p8Affine,
            &[bytes(values), matrix(0x0102_0408_1020_4080), byte(0x5a)]
        ),
        Some(bytes(values.map(|value| value ^ 0x5a)))
    );
    assert_eq!(
        VectorCryptoKind::AesEncryptKeyLocker.evaluate(&[state, key], 0),
        None
    );
}

/// An SVE pattern selects the elements Arm's `DecodePredCount` does of a
/// count of seven: the largest power of two, a fixed count only where seven
/// reach it, the largest multiples of four and three, all, and none for a
/// reserved field; every field encoding reads back as the field it is.
#[test]
fn an_element_pattern_selects_what_decode_pred_count_does() {
    let count = 7;
    for (pattern, selected) in [
        (ElementPattern::Pow2, 4),
        (ElementPattern::Fixed(7), 7),
        (ElementPattern::Fixed(8), 0),
        (ElementPattern::Mul4, 4),
        (ElementPattern::Mul3, 6),
        (ElementPattern::All, 7),
        (ElementPattern::Reserved(20), 0),
    ] {
        assert_eq!(pattern.select(count), selected, "{pattern}");
    }
    assert_eq!(ElementPattern::Pow2.select(0), 0);
    for bits in 0..32u8 {
        assert_eq!(ElementPattern::from_bits(bits).bits(), Some(bits));
    }
    assert_eq!(ElementPattern::from_bits(9), ElementPattern::Fixed(16));
    assert_eq!(ElementPattern::from_bits(13), ElementPattern::Fixed(256));
    assert_eq!(ElementPattern::Fixed(17).bits(), None);
}

/// The MPX bound checks are conditional traps on an unsigned comparison of the
/// bound with the checked value, in that order: `bndcl` fires when the lower
/// bound exceeds the value and `bndcu`/`bndcn` when the upper bound is below
/// it, so a pass that decides the comparison decides whether the fault is
/// raised. The bound-table load binds its bound register and the store only
/// reads and writes memory; both may fault.
#[test]
fn the_bound_checks_trap_on_the_unsigned_side_of_their_bound() {
    assert_eq!(TrapOp::BoundsLower.condition(), Some((CmpKind::Gt, true)));
    assert_eq!(TrapOp::BoundsUpper.condition(), Some((CmpKind::Lt, true)));
    assert!(TrapOp::BoundsLower.is_conditional());
    assert!(!TrapOp::BoundsCheck.is_conditional());
    assert_eq!(TrapOp::BoundsLower.name(), "trap.bounds.lower");
    assert_eq!(TrapOp::BoundsUpper.name(), "trap.bounds.upper");
    let load = MachineStateOp::BoundTableLoad;
    let store = MachineStateOp::BoundTableStore;
    assert!(load.writes_destination_operand());
    assert!(!store.writes_destination_operand());
    assert_eq!(
        SystemOpKind::MachineState(load).effects().kind,
        SsaEffectKind::Read
    );
    assert_eq!(
        SystemOpKind::MachineState(store).effects().kind,
        SsaEffectKind::ReadWrite
    );
    assert!(SystemOpKind::MachineState(load).effects().may_throw);
}

/// **A qualified access writes its space before its address, and a flat one
/// writes nothing.** The address space and the permission are each spelled
/// once, so `fs:` and a user access stay apart in the IR text.
#[test]
fn a_qualified_access_writes_its_space() {
    let (dest, addr, value) = (
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
        SsaVarId::from_index(3),
    );
    let load = |space| {
        SsaOp::<MockTarget>::LoadIndirect {
            dest,
            addr,
            value_type: MockType::I32,
            access: space,
        }
        .to_string()
    };
    assert_eq!(
        load(AccessQualifiers::FLAT),
        format!("{dest} = ldind.i32 {addr}")
    );
    assert_eq!(
        load(AccessQualifiers::new(
            NonZeroU16::new(1),
            AccessPermission::Unprivileged
        )),
        format!("{dest} = ldind.i32 addrspace(1) perm(user) {addr}")
    );
    let streaming = AccessQualifiers::FLAT.with_locality(Locality::NonTemporal);
    assert!(!streaming.is_flat());
    assert_eq!(
        load(streaming),
        format!("{dest} = ldind.i32 nontemporal {addr}")
    );
    let swap = SsaOp::<MockTarget>::AtomicLockRmw {
        dest,
        addr,
        value,
        op: AtomicRmwOp::Xchg,
        ordering: AtomicOrdering::Release,
        width: AtomicAccessWidth::Bits64,
        access: AccessQualifiers::new(None, AccessPermission::ShadowStack),
    }
    .to_string();
    assert!(swap.contains(&format!("perm(shadow) {addr}")), "{swap}");
}

/// A conversion's text says how it reads its operand and whether it checks
/// the result: a zero-extension and a sign-extension to one type are two
/// operations, and printed alike they read as one (`conv.u64` of a signed
/// value looked like a zero-extension of it).
#[test]
fn a_conversion_prints_its_extension_and_its_check() {
    let (dest, operand) = (SsaVarId::from_index(1), SsaVarId::from_index(0));
    let int_conv = |overflow_check: bool, unsigned: bool| -> String {
        SsaOp::<MockTarget>::IntConv {
            dest,
            operand,
            target: MockType::I64,
            overflow_check,
            unsigned,
        }
        .to_string()
    };
    assert_eq!(int_conv(false, false), "v1 = conv.i64 v0");
    assert_eq!(int_conv(false, true), "v1 = conv.i64.un v0");
    assert_eq!(int_conv(true, true), "v1 = conv.ovf.i64.un v0");
    let to_float = SsaOp::<MockTarget>::IntToFloat {
        dest,
        operand,
        target: MockType::F64,
        unsigned: true,
        rounding: None,
        environment: None,
    };
    assert_eq!(to_float.to_string(), "v1 = inttofloat.f64.un v0");
    let to_int = SsaOp::<MockTarget>::FloatToInt {
        dest,
        operand,
        target: MockType::I32,
        overflow_check: true,
        unsigned: false,
        source: None,
    };
    assert_eq!(to_int.to_string(), "v1 = floattoint.ovf.i32 v0");
}

/// A float raise defines its flags alone and reads what its operation reads:
/// the constructor makes the operation's definition the raise's, a rename of
/// an operand reaches into the operation, and the raise is pure.
#[test]
fn a_float_raise_defines_its_flags_and_reads_its_operations_operands() {
    let (left, right, sum, flags, renamed) = (
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
        SsaVarId::from_index(3),
        SsaVarId::from_index(9),
        SsaVarId::from_index(5),
    );
    let add: SsaOp<MockTarget> = SsaOp::Add {
        dest: sum,
        left,
        right,
        flags: None,
    };
    let mut raise = SsaOp::float_raise(
        flags,
        add,
        None,
        FloatExceptions::INEXACT.union(FloatExceptions::INVALID),
        FloatExceptionLayout::IEEE,
    );
    assert_eq!(raise.defs().collect::<Vec<_>>(), vec![flags]);
    assert_eq!(raise.uses(), vec![left, right]);
    assert_eq!(
        raise.to_string(),
        format!("{flags} = float.raise inexact@0 invalid@4 (add {left}, {right})")
    );
    assert!(raise.is_pure());
    assert!(!raise.may_throw());
    assert_eq!(raise.replace_uses(left, renamed), 1);
    assert_eq!(raise.uses(), vec![renamed, right]);
    assert!(
        !raise.replace_def(sum, renamed),
        "the operation's own definition is gone"
    );
    // Over the lanes a mask names, the raise reads the mask too.
    let lanes = SsaVarId::from_index(6);
    let masked = SsaOp::<MockTarget>::float_raise(
        flags,
        SsaOp::Add {
            dest: sum,
            left,
            right,
            flags: None,
        },
        Some(lanes),
        FloatExceptions::INEXACT,
        FloatExceptionLayout::IEEE,
    );
    assert_eq!(masked.uses(), vec![left, right, lanes]);
    assert_eq!(
        masked.to_string(),
        format!("{flags} = float.raise inexact@0 (add {left}, {right}) lanes={lanes}")
    );
}

/// A signalling compare is spelled as one and, like every float operation,
/// raises its exception as a flag rather than trapping.
#[test]
fn a_signalling_compare_is_spelled_and_does_not_trap() {
    let (dest, left, right) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
    );
    let less: SsaOp<MockTarget> = SsaOp::Clt {
        dest,
        left,
        right,
        unsigned: false,
        signaling: true,
    };
    assert_eq!(
        less.to_string(),
        format!("{dest} = clt.signaling {left}, {right}")
    );
    assert!(!less.may_throw());
    let flags: SsaOp<MockTarget> = SsaOp::FloatCompareFlags {
        flags: dest,
        left,
        right,
        signaling: true,
    };
    assert!(!flags.may_throw());
    assert!(flags.is_pure());
}

/// The operations a raise can be of are the float computations: arithmetic of
/// a float or of lanes that may be floats, a compare of floats, a conversion
/// from or to one, and a negation or absolute value computed as arithmetic.
/// An integer sum, a bitwise lane operation and a negation, which IEEE 754
/// calls non-computational, are not.
#[test]
fn the_operations_that_may_raise_are_the_float_computations() {
    let (dest, left, right) = (
        SsaVarId::from_index(0),
        SsaVarId::from_index(1),
        SsaVarId::from_index(2),
    );
    let float = |var: SsaVarId| Some(var != SsaVarId::from_index(9));
    let integer = |_: SsaVarId| Some(false);
    let add: SsaOp<MockTarget> = SsaOp::Add {
        dest,
        left,
        right,
        flags: None,
    };
    assert!(add.may_raise_float_exceptions(float));
    assert!(!add.may_raise_float_exceptions(integer));
    assert!(
        add.may_raise_float_exceptions(|_| None),
        "an unknown type may be a float"
    );
    let lanes = |kind: VectorBinaryKind, element_kind: VectorElementKind| -> SsaOp<MockTarget> {
        SsaOp::VectorBinary {
            dest,
            left,
            right,
            kind,
            element: VectorElement {
                kind: element_kind,
                ..VectorElement::default()
            },
            environment: None,
        }
    };
    assert!(
        lanes(VectorBinaryKind::Mul, VectorElementKind::Float).may_raise_float_exceptions(integer)
    );
    assert!(
        lanes(VectorBinaryKind::Mul, VectorElementKind::Unknown)
            .may_raise_float_exceptions(integer)
    );
    assert!(
        !lanes(VectorBinaryKind::Mul, VectorElementKind::Integer)
            .may_raise_float_exceptions(integer)
    );
    assert!(
        !lanes(VectorBinaryKind::Xor, VectorElementKind::Float).may_raise_float_exceptions(integer)
    );
    let negation: SsaOp<MockTarget> = SsaOp::VectorUnary {
        dest,
        value: left,
        kind: VectorUnaryKind::Neg,
        element: VectorElement {
            kind: VectorElementKind::Float,
            ..VectorElement::default()
        },
        environment: None,
    };
    assert!(!negation.may_raise_float_exceptions(float));
    for kind in [
        VectorUnaryKind::NegArithmetic,
        VectorUnaryKind::AbsArithmetic,
    ] {
        let arithmetic: SsaOp<MockTarget> = SsaOp::VectorUnary {
            dest,
            value: left,
            kind,
            element: VectorElement {
                kind: VectorElementKind::Float,
                ..VectorElement::default()
            },
            environment: None,
        };
        assert!(arithmetic.may_raise_float_exceptions(float), "{kind:?}");
    }
    let less: SsaOp<MockTarget> = SsaOp::Clt {
        dest,
        left,
        right,
        unsigned: false,
        signaling: true,
    };
    assert!(less.may_raise_float_exceptions(float));
    assert!(!less.may_raise_float_exceptions(integer));
    let copy: SsaOp<MockTarget> = SsaOp::Copy { dest, src: left };
    assert!(!copy.may_raise_float_exceptions(float));
}

/// One lane of an SVE float helper of `element_bits`-wide float lanes, its
/// operands each a one-lane vector, a `ftmad` immediate a scalar after them.
fn helper(
    kind: FpHelperKind,
    element_bits: u32,
    operands: &[u128],
    immediate: Option<u128>,
    rounding: Option<FloatRounding>,
) -> Option<u128> {
    let mut inputs: Vec<VectorConst> = operands
        .iter()
        .map(|operand| VectorConst::from_lanes(element_bits, [*operand]).unwrap())
        .collect();
    if let Some(immediate) = immediate {
        inputs.push(VectorConst::from_lanes(8, [immediate]).unwrap());
    }
    let element = VectorElement {
        rounding,
        ..VectorElement::float(element_bits)
    };
    Some(kind.evaluate_lanes(&inputs, element, None)?.lane(0))
}

/// **`frecpx` complements the exponent and `flogb` reads it.** The reciprocal
/// exponent of `2.0` is `1.0`, of `-0.5` `-4.0`, of a zero the largest finite
/// exponent and of an infinity zero; `flogb` of `8.0f` is 3, of `0.75` -1, of
/// a zero the least integer and of an infinity the greatest. A NaN's
/// reciprocal exponent and a subnormal's logarithm are the processor's
/// default-NaN and flush-to-zero modes' to decide, and fold nothing.
#[test]
fn the_reciprocal_exponent_and_the_exponent_are_the_exponent_fields() {
    let double = |value: f64| u128::from(value.to_bits());
    let recpx = |value: u128| helper(FpHelperKind::ReciprocalExponent, 64, &[value], None, None);
    assert_eq!(recpx(double(2.0)), Some(double(1.0)));
    assert_eq!(recpx(double(-0.5)), Some(double(-4.0)));
    assert_eq!(recpx(0), Some(0x7fe0_0000_0000_0000));
    assert_eq!(recpx(double(f64::INFINITY)), Some(0));
    assert_eq!(recpx(double(f64::NAN)), None);
    let logb =
        |bits: u32, value: u128| helper(FpHelperKind::ExtractExponent, bits, &[value], None, None);
    assert_eq!(logb(32, u128::from(8.0f32.to_bits())), Some(3));
    assert_eq!(logb(64, double(0.75)), Some(u128::from(u64::MAX)));
    assert_eq!(logb(64, 0), Some(1 << 63));
    assert_eq!(
        logb(64, double(f64::INFINITY)),
        Some(u128::from(i64::MAX.cast_unsigned()))
    );
    assert_eq!(logb(64, double(f64::NAN)), Some(1 << 63));
    assert_eq!(logb(16, 0x3c00), Some(0), "binary16 one");
    assert_eq!(logb(64, 1), None, "a subnormal");
}

/// **`fexpa` is `2^(i / 64)` under the exponent above the index.** Each
/// table entry is the fraction of `2^(i / 64)` (binary16 `2^(i / 32)`), as
/// computed in binary64 to within one unit in the last place, and the
/// midpoint is the square root of two exactly, in every format.
#[test]
fn the_exponential_accelerator_is_two_to_the_index() {
    let accelerate =
        |bits: u32, value: u128| helper(FpHelperKind::ExpAccelerate, bits, &[value], None, None);
    // Index 32 under the bias is the square root of two.
    assert_eq!(
        accelerate(64, 32 | 1023 << 6),
        Some(u128::from(2f64.sqrt().to_bits()))
    );
    assert_eq!(
        accelerate(32, 32 | 127 << 6),
        Some(u128::from(2f32.sqrt().to_bits()))
    );
    assert_eq!(accelerate(16, 16 | 15 << 5), Some(0x3da8));
    for (bits, index_bits, bias) in [(64u32, 6u32, 1023u128), (32, 6, 127), (16, 5, 15)] {
        let format = FloatFormat::of_width(bits).unwrap();
        let fraction = format.fraction_bits();
        let count = 1u32 << index_bits;
        for index in 0..count {
            let got = accelerate(bits, u128::from(index) | bias << index_bits).unwrap();
            let exact = 2f64.powf(f64::from(index) / f64::from(count));
            // The value's fraction, at the format's precision.
            let want = ((exact - 1.0) * 2f64.powi(i32::try_from(fraction).unwrap())).round();
            let got_fraction = got & ((1u128 << fraction) - 1);
            let distance = (want
                - f64::from(
                    u32::try_from(got_fraction >> fraction.saturating_sub(32)).unwrap_or(0),
                ))
            .abs();
            if fraction <= 32 {
                assert!(distance <= 1.0, "{bits}-bit index {index}: {got:#x}");
            } else {
                let got = f64::from_bits(u64::try_from(got).unwrap());
                assert!(
                    (got - exact).abs() <= exact * f64::EPSILON,
                    "index {index}: {got}"
                );
            }
            assert_eq!(got >> fraction, bias, "{bits}-bit index {index}'s exponent");
        }
    }
}

/// A return's target is an ordinary use, remapped with its value, while a
/// call's output keys follow the definitions they name.
#[test]
fn return_targets_are_uses_and_clobber_keys_follow_fresh_defs() {
    let input = SsaVarId::from_index(7);
    let changed = SsaVarId::from_index(11);
    let mut returned = SsaOp::<MockTarget>::Return {
        value: Some(input),
        state: Box::default(),
        results: Default::default(),
        transfer: Some(ReturnTransfer::through(input)),
    };
    assert_eq!(returned.uses(), vec![input, input]);
    assert_eq!(returned.replace_uses(input, changed), 2);
    assert_eq!(returned.uses(), vec![changed, changed]);
    assert!(returned.to_string().contains("[through "));
    let mut clobber = SsaOp::<MockTarget>::CallClobber {
        outputs: vec![input],
        machine: Some(
            vec![CallOutput {
                key: CallInputKey::new(17),
                index: 0,
            }]
            .into_boxed_slice(),
        ),
    };
    assert!(clobber.replace_def(input, changed));
    assert_eq!(clobber.defs().collect::<Vec<_>>(), vec![changed]);
    assert!(clobber.uses().is_empty());
    assert!(clobber.to_string().contains("17:#0"));
}

/// Large successor lists stop invoking the visitor as soon as it declines.
#[test]
fn successor_visitation_stops_before_the_remaining_switch_or_indirect_targets() {
    for indirect in [false, true] {
        let op: SsaOp<MockTarget> = if indirect {
            SsaOp::IndirectBranch {
                target: SsaVarId::from_index(0),
                resolved_targets: (0..4096).collect(),
                state: Box::default(),
            }
        } else {
            SsaOp::Switch {
                value: SsaVarId::from_index(0),
                targets: (0..4096).collect(),
                default: 5000,
            }
        };
        let mut visited = Vec::new();
        let result = op.try_for_each_successor(|target| {
            visited.push(target);
            if target == 64 {
                Err("cancelled")
            } else {
                Ok(())
            }
        });
        assert_eq!(visited, (0..=64).collect::<Vec<_>>());
        assert_eq!(result, Err("cancelled"));
        visited.clear();
        op.try_for_each_successor(|target| {
            visited.push(target);
            Ok::<(), ()>(())
        })
        .unwrap();
        let mut expected: Vec<_> = (0..4096).collect();
        if !indirect {
            expected.push(5000);
        }
        assert_eq!(visited, expected);
    }
}

/// Fallible and ordinary successor visitors preserve duplicate transfers and operand order.
#[test]
fn successor_visitors_preserve_each_control_shape_and_error_position() {
    let v = SsaVarId::from_index(0);
    let cases: Vec<(SsaOp<MockTarget>, Vec<usize>)> = vec![
        (SsaOp::Jump { target: 5 }, vec![5]),
        (SsaOp::Leave { target: 6 }, vec![6]),
        (
            SsaOp::Branch {
                condition: v,
                true_target: 7,
                false_target: 7,
            },
            vec![7, 7],
        ),
        (
            SsaOp::Switch {
                value: v,
                targets: vec![2, 1, 2],
                default: 4,
            },
            vec![2, 1, 2, 4],
        ),
        (
            SsaOp::IndirectBranch {
                target: v,
                resolved_targets: vec![3, 3, 1],
                state: Box::default(),
            },
            vec![3, 3, 1],
        ),
        (SsaOp::returns(None), vec![]),
    ];
    for (op, expected) in cases {
        let mut visited = Vec::new();
        op.for_each_successor(|target| visited.push(target));
        assert_eq!(visited, expected);
        for fail_at in 0..expected.len() {
            let mut visited = Vec::new();
            let result = op.try_for_each_successor(|target| {
                let at = visited.len();
                visited.push(target);
                if at == fail_at { Err(fail_at) } else { Ok(()) }
            });
            assert_eq!(result, Err(fail_at));
            assert_eq!(visited, expected.get(..=fail_at).unwrap());
        }
    }
}

/// **The trigonometric helpers select and accumulate.** `ftsmul` of `3.0`
/// is `9.0`, negative where the selector's bit 0 is set; `ftssel` is `1.0`
/// where bit 0 is set and the operand otherwise, negated where bit 1 is;
/// `ftmad` adds the sine coefficient its immediate names to `n * m` for a
/// positive `m` and the cosine one to `n * |m|` for a negative, rounded once
/// as the hardware's fused multiply-add rounds. An inexact product in an
/// unknown rounding mode, and a NaN whose sign the processor's alternate
/// handling decides, fold nothing.
#[test]
fn the_trigonometric_helpers_select_and_accumulate() {
    let double = |value: f64| u128::from(value.to_bits());
    let smul =
        |n: f64, m: u128| helper(FpHelperKind::TrigSelectMul, 64, &[double(n), m], None, None);
    assert_eq!(smul(3.0, 0), Some(double(9.0)));
    assert_eq!(smul(3.0, 1), Some(double(-9.0)));
    assert_eq!(smul(-3.0, 0), Some(double(9.0)));
    assert_eq!(smul(f64::NAN, 0), None);
    let ssel = |n: u128, m: u128| helper(FpHelperKind::TrigSelect, 64, &[n, m], None, None);
    assert_eq!(ssel(double(0.25), 0), Some(double(0.25)));
    assert_eq!(ssel(double(0.25), 1), Some(double(1.0)));
    assert_eq!(ssel(double(0.25), 2), Some(double(-0.25)));
    assert_eq!(ssel(double(0.25), 3), Some(double(-1.0)));
    assert_eq!(ssel(double(f64::NAN), 0), Some(double(f64::NAN)));
    assert_eq!(ssel(double(f64::NAN), 2), None);
    let mad = |n: f64, m: f64, index: u128, rounding| {
        helper(
            FpHelperKind::TrigMulAdd,
            64,
            &[double(n), double(m)],
            Some(index),
            rounding,
        )
    };
    // The first coefficient of each series is one; the cosine's second -1/2.
    assert_eq!(mad(0.5, 2.0, 0, None), Some(double(2.0)));
    assert_eq!(mad(0.5, -2.0, 0, None), Some(double(2.0)));
    assert_eq!(mad(0.5, -2.0, 1, None), Some(double(0.5)));
    // The sine's second, -1/6 rounded, makes the sum inexact.
    let sine = f64::from_bits(0xbfc5_5555_5555_5543);
    assert_eq!(mad(0.1, 0.7, 1, None), None);
    assert_eq!(
        mad(0.1, 0.7, 1, Some(FloatRounding::NearestEven)),
        Some(double(0.1f64.mul_add(0.7, sine)))
    );
    assert_ne!(
        mad(0.1, 0.7, 1, Some(FloatRounding::Up)),
        mad(0.1, 0.7, 1, Some(FloatRounding::Down)),
        "the sum is inexact"
    );
    assert_eq!(mad(f64::NAN, 1.0, 0, None), None);
    assert_eq!(mad(0.5, 2.0, 8, None), None, "an immediate past seven");
}

/// One lane of a float-controlled operation over lanes `bits` wide, each
/// operand a one-lane vector.
fn controlled(
    kind: ControlledKind,
    control: u8,
    bits: u32,
    operands: &[u128],
    environment: Option<FloatRounding>,
) -> Option<u128> {
    let inputs: Vec<VectorConst> = operands
        .iter()
        .map(|operand| VectorConst::from_lanes(bits, [*operand]).unwrap())
        .collect();
    Some(
        kind.evaluate_lanes(control, &inputs, VectorElement::float(bits), environment)?
            .lane(0),
    )
}

/// The bits of a binary64 value.
fn double(value: f64) -> u128 {
    u128::from(value.to_bits())
}

/// **`vrndscale` rounds to a multiple of `2^-M` and `vreduce` is what is
/// left.** By the control's rounding (nearest even, down, up, toward zero)
/// `2.5` is 2, 2, 3 and 2, and `-2.5` down is -3; at `M = 1` and `M = 2`,
/// `1.37` is 1.5 and 1.25; `-0.25` is -0 to nearest; an infinity is kept and
/// a NaN quieted; Xeon Phi's fixed-point round at 4 fraction bits takes
/// `1.03125` to 1 to nearest and 1.0625 up. A rounding the control leaves to
/// `MXCSR` folds only where every mode agrees, and a subnormal folds
/// nothing. `vreduce` of 2.75 is -0.25 to nearest, of a multiple of `2^-M`
/// +0 (and -0 rounding down), of an infinity +0. Worked from the Intel SDM's
/// `RoundToInteger` and `ReduceArgument`.
#[test]
fn a_scaled_round_is_a_multiple_of_its_scale_and_a_reduction_the_rest() {
    let round = |control: u8, x: f64| {
        controlled(ControlledKind::RoundScale, control, 64, &[double(x)], None)
    };
    assert_eq!(round(0x00, 2.5), Some(double(2.0)));
    assert_eq!(round(0x01, 2.5), Some(double(2.0)));
    assert_eq!(round(0x02, 2.5), Some(double(3.0)));
    assert_eq!(round(0x03, 2.5), Some(double(2.0)));
    assert_eq!(round(0x01, -2.5), Some(double(-3.0)));
    assert_eq!(round(0x10, 1.37), Some(double(1.5)));
    assert_eq!(round(0x20, 1.37), Some(double(1.25)));
    assert_eq!(round(0x00, -0.25), Some(double(-0.0)));
    assert_eq!(round(0x00, f64::INFINITY), Some(double(f64::INFINITY)));
    assert_eq!(
        controlled(
            ControlledKind::RoundScale,
            0,
            64,
            &[0x7ff0_0000_0000_0001],
            None
        ),
        Some(0x7ff8_0000_0000_0001)
    );
    assert_eq!(
        controlled(ControlledKind::RoundScale, 0, 64, &[1], None),
        None,
        "a subnormal"
    );
    // By MXCSR: 2.5 differs by mode, 2.0 does not.
    assert_eq!(round(0x04, 2.5), None);
    assert_eq!(round(0x04, 2.0), Some(double(2.0)));
    assert_eq!(
        controlled(
            ControlledKind::RoundScale,
            0x04,
            64,
            &[double(2.5)],
            Some(FloatRounding::Up)
        ),
        Some(double(3.0))
    );
    // A half: 2.5 to nearest even is 2.
    assert_eq!(
        controlled(ControlledKind::RoundScale, 0, 16, &[0x4100], None),
        Some(0x4000)
    );
    let fixed = |control: u8, x: f64| {
        controlled(
            ControlledKind::RoundFixedPoint,
            control,
            64,
            &[double(x)],
            None,
        )
    };
    assert_eq!(fixed(0x10, 1.03125), Some(double(1.0)));
    assert_eq!(fixed(0x12, 1.03125), Some(double(1.0625)));
    let reduce =
        |control: u8, x: f64| controlled(ControlledKind::Reduce, control, 64, &[double(x)], None);
    assert_eq!(reduce(0x00, 2.75), Some(double(-0.25)));
    assert_eq!(reduce(0x20, 2.75), Some(double(0.0)));
    assert_eq!(reduce(0x21, 2.75), Some(double(-0.0)));
    assert_eq!(reduce(0x00, f64::NEG_INFINITY), Some(0));
    // A scalar form keeps the other lanes of its first input.
    let upper = VectorConst::from_lanes(64, [double(7.0), double(8.0)]).unwrap();
    let data = VectorConst::from_lanes(64, [double(2.5), double(9.0)]).unwrap();
    let scalar = VectorElement {
        scalar: true,
        ..VectorElement::float(64)
    };
    assert_eq!(
        ControlledKind::RoundScale.evaluate_lanes(0, &[upper, data], scalar, None),
        VectorConst::from_lanes(64, [double(2.0), double(8.0)])
    );
}

/// **`vgetmant` normalizes the significand into the interval its control
/// names.** `10.0` is `1.25 * 2^3`: 1.25 in `[1, 2)`, 0.625 in `[1/2, 1)` and
/// in `[1/2, 2)` (the exponent 3 is odd), where `5.0`, of the even exponent
/// 2, is 1.25; `12.0` is 0.75 in `[3/4, 3/2)`
/// (its fraction's top bit is set). `-10.0` keeps its sign, is positive under
/// sign control 1 and the indefinite NaN under 2; `-0.0` is -1, an infinity 1.
#[test]
fn a_mantissa_is_normalized_into_the_interval_its_control_names() {
    let mant = |control: u8, x: f64| {
        controlled(ControlledKind::GetMantissa, control, 64, &[double(x)], None)
    };
    assert_eq!(mant(0b0000, 10.0), Some(double(1.25)));
    assert_eq!(mant(0b0010, 10.0), Some(double(0.625)));
    assert_eq!(mant(0b0001, 10.0), Some(double(0.625)));
    assert_eq!(
        mant(0b0001, 5.0),
        Some(double(1.25)),
        "the exponent 2 is even"
    );
    assert_eq!(mant(0b0011, 12.0), Some(double(0.75)));
    assert_eq!(mant(0b0011, 10.0), Some(double(1.25)));
    assert_eq!(mant(0b0000, -10.0), Some(double(-1.25)));
    assert_eq!(mant(0b0100, -10.0), Some(double(1.25)));
    assert_eq!(mant(0b1000, -10.0), Some(0xfff8_0000_0000_0000));
    assert_eq!(mant(0b0000, -0.0), Some(double(-1.0)));
    assert_eq!(mant(0b0000, f64::INFINITY), Some(double(1.0)));
}

/// **`vrange` selects by value or magnitude and states the sign its control
/// names.** Of 3 and -5: the minimum -5, the maximum 3, the one of smaller
/// magnitude 3 and of greater -5, with the first's sign 5 and with the sign
/// cleared or set 5 and -5; zeros of opposite signs give -0 as a minimum and
/// +0 as a maximum, as equal magnitudes give the negative and the positive
/// one; a quiet NaN gives the other operand, a signalling one itself quieted.
#[test]
fn a_range_selects_by_value_or_magnitude_with_the_sign_its_control_names() {
    let range = |control: u8, a: f64, b: f64| {
        controlled(
            ControlledKind::Range,
            control,
            64,
            &[double(a), double(b)],
            None,
        )
    };
    assert_eq!(range(0b0100, 3.0, -5.0), Some(double(-5.0)));
    assert_eq!(range(0b0101, 3.0, -5.0), Some(double(3.0)));
    assert_eq!(range(0b0110, 3.0, -5.0), Some(double(3.0)));
    assert_eq!(range(0b0111, 3.0, -5.0), Some(double(-5.0)));
    assert_eq!(range(0b0011, 3.0, -5.0), Some(double(5.0)));
    assert_eq!(range(0b1011, 3.0, -5.0), Some(double(5.0)));
    assert_eq!(range(0b1111, 3.0, -5.0), Some(double(-5.0)));
    assert_eq!(range(0b0100, 0.0, -0.0), Some(double(-0.0)));
    assert_eq!(range(0b0101, -0.0, 0.0), Some(double(0.0)));
    assert_eq!(range(0b0110, 2.0, -2.0), Some(double(-2.0)));
    assert_eq!(range(0b0111, -2.0, 2.0), Some(double(2.0)));
    assert_eq!(range(0b0100, f64::NAN, 2.0), Some(double(2.0)));
    assert_eq!(range(0b0100, 2.0, f64::NAN), Some(double(2.0)));
    assert_eq!(
        controlled(
            ControlledKind::Range,
            0b0100,
            64,
            &[0x7ff0_0000_0000_0001, double(2.0)],
            None
        ),
        Some(0x7ff8_0000_0000_0001)
    );
}

/// **`vfixupimm` replaces each lane by the response its class's nibble of
/// the table names.** +0 (token 2) under response 0xA is +1; a quiet NaN
/// (token 0) under 3 the indefinite NaN; 1.0 (token 3) under 0 keeps the
/// destination; -5 (token 6) under 0xD is pi/2; +inf (token 5) under 6 is
/// +inf; 7 (token 7) under 1 is itself; and the constants 1/2, 90 and the
/// largest finite value of each sign.
#[test]
fn a_fixup_replaces_a_lane_by_the_response_its_class_names() {
    let fixup = |destination: f64, source: f64, token: u32, response: u128| {
        controlled(
            ControlledKind::Fixup,
            0,
            64,
            &[double(destination), double(source), response << (4 * token)],
            None,
        )
    };
    assert_eq!(fixup(9.0, 0.0, 2, 0xA), Some(double(1.0)));
    assert_eq!(fixup(9.0, f64::NAN, 0, 0x3), Some(0xfff8_0000_0000_0000));
    assert_eq!(fixup(9.0, 1.0, 3, 0x0), Some(double(9.0)));
    assert_eq!(
        fixup(9.0, -5.0, 6, 0xD),
        Some(double(std::f64::consts::FRAC_PI_2))
    );
    assert_eq!(
        fixup(9.0, f64::INFINITY, 5, 0x6),
        Some(double(f64::INFINITY))
    );
    assert_eq!(fixup(9.0, 7.0, 7, 0x1), Some(double(7.0)));
    assert_eq!(fixup(9.0, 7.0, 7, 0xB), Some(double(0.5)));
    assert_eq!(fixup(9.0, 7.0, 7, 0xC), Some(double(90.0)));
    assert_eq!(fixup(9.0, 7.0, 7, 0xE), Some(double(f64::MAX)));
    assert_eq!(fixup(9.0, 7.0, 7, 0xF), Some(double(f64::MIN)));
    assert_eq!(
        controlled(
            ControlledKind::Fixup,
            0,
            32,
            &[0, u128::from(1.0f32.to_bits()), 0xd << 12],
            None
        ),
        Some(u128::from(std::f32::consts::FRAC_PI_2.to_bits()))
    );
}

/// **`vfpclass` sets a lane where its source is of a class the control
/// names.** +0 under bit 1 and not under bit 2, -0 under bit 2, a negative
/// number under bit 6, a quiet NaN under bit 0 and a signalling one under
/// bit 7, +inf under bit 3; a subnormal's class is the denormals-are-zero
/// mode's to decide.
#[test]
fn a_class_test_sets_the_lanes_of_the_classes_its_control_names() {
    let class =
        |control: u8, x: u128| controlled(ControlledKind::Classify, control, 64, &[x], None);
    let ones = u128::from(u64::MAX);
    assert_eq!(class(0x02, double(0.0)), Some(ones));
    assert_eq!(class(0x04, double(0.0)), Some(0));
    assert_eq!(class(0x04, double(-0.0)), Some(ones));
    assert_eq!(class(0x40, double(-3.0)), Some(ones));
    assert_eq!(class(0x40, double(3.0)), Some(0));
    assert_eq!(class(0x01, double(f64::NAN)), Some(ones));
    assert_eq!(class(0x80, 0x7ff0_0000_0000_0001), Some(ones));
    assert_eq!(class(0x01, 0x7ff0_0000_0000_0001), Some(0));
    assert_eq!(class(0x08, double(f64::INFINITY)), Some(ones));
    assert_eq!(class(0x20, 1), None);
}

/// Fallible visitors preserve the independent mutable walk and stop at every operand boundary.
#[test]
fn fallible_operands_preserve_every_variant_and_error_prefix() {
    let operations = all_sample_ops()
        .into_iter()
        .chain(visitor_battery().into_iter().map(|(op, _, _)| op));
    for mut op in operations {
        let mut expected = Vec::new();
        op.visit_operands_mut(|role, var| expected.push((role, *var)));
        let first_use = expected
            .iter()
            .position(|(role, _)| *role == OperandRole::Use)
            .unwrap_or(expected.len());
        assert!(
            expected
                .iter()
                .skip(first_use)
                .all(|(role, _)| *role == OperandRole::Use),
            "definitions follow a use in {op}"
        );
        for limit in 0..=expected.len() {
            let mut seen = Vec::new();
            let result = op.try_visit_operands(|role, var| {
                seen.push((role, var));
                if seen.len() > limit {
                    Err(limit)
                } else {
                    Ok(())
                }
            });
            assert_eq!(
                seen,
                expected.iter().take(limit + 1).copied().collect::<Vec<_>>(),
                "{op}"
            );
            assert_eq!(
                result,
                if limit < expected.len() {
                    Err(limit)
                } else {
                    Ok(())
                },
                "{op}"
            );
        }
        for definitions in [true, false] {
            let expected: Vec<_> = expected
                .iter()
                .filter_map(|(role, var)| {
                    ((*role != OperandRole::Use) == definitions).then_some(*var)
                })
                .collect();
            for limit in 0..=expected.len() {
                let mut seen = Vec::new();
                let mut visit = |var| {
                    seen.push(var);
                    if seen.len() > limit {
                        Err(limit)
                    } else {
                        Ok(())
                    }
                };
                let result = if definitions {
                    op.try_for_each_def(&mut visit)
                } else {
                    op.try_for_each_use(&mut visit)
                };
                assert_eq!(
                    seen,
                    expected.iter().take(limit + 1).copied().collect::<Vec<_>>(),
                    "{op}"
                );
                assert_eq!(
                    result,
                    if limit < expected.len() {
                        Err(limit)
                    } else {
                        Ok(())
                    },
                    "{op}"
                );
            }
        }
    }
}

/// Large native payloads retain definition order and stop without traversing their tails.
#[test]
fn fallible_boxed_operand_lists_preserve_large_definition_order() {
    let outputs: Vec<_> = (0..8192).map(SsaVarId::from_index).collect();
    let inputs: Vec<_> = (8192..16384).map(SsaVarId::from_index).collect();
    let op = SsaOp::<MockTarget>::VectorExtendInLane(Box::new(VectorExtendInLaneData {
        signed: false,
        source_bits: 8,
        element_bits: 32,
        outputs: outputs.clone(),
        inputs: inputs.clone(),
    }));
    assert_eq!(op.defs().collect::<Vec<_>>(), outputs);
    assert_eq!(op.uses(), inputs);
    assert_eq!(op.dest(), Some(SsaVarId::from_index(0)));
    assert_eq!(op.first_use(), Some(SsaVarId::from_index(8192)));
    let mut seen = 0;
    assert_eq!(
        op.try_visit_operands(|_, _| {
            seen += 1;
            if seen == 129 { Err(seen) } else { Ok(()) }
        }),
        Err(129)
    );
    assert_eq!(seen, 129);
}

/// Every operation has distinct canonical content and stops at every possible refusal point.
#[test]
fn canonical_content_visits_all_variants_and_propagates_every_refusal_prefix() {
    let ops = all_sample_ops();
    assert_eq!(
        ops.len(),
        176,
        "every call and function-pointer form is represented"
    );
    let variants = ops.iter().map(mem::discriminant).collect::<FxHashSet<_>>();
    assert_eq!(
        variants.len(),
        169,
        "every operation variant is represented"
    );
    let mut identities = FxHashSet::default();
    for op in ops {
        let mut complete = crate::ir::content::tests::Probe::new(usize::MAX);
        assert_eq!(op.write_content(&mut complete), Ok(()));
        assert!(
            identities.insert(complete.data.clone()),
            "content collision for {op}"
        );
        for limit in 0..complete.events {
            let mut partial = crate::ir::content::tests::Probe::new(limit);
            assert_eq!(
                op.write_content(&mut partial),
                Err(()),
                "missed refusal {limit} for {op}"
            );
            assert_eq!(partial.events, limit);
            assert_eq!(partial.after_refusal, 0, "visited after refusal for {op}");
            assert!(
                complete.data.starts_with(&partial.data),
                "wrong output prefix for {op}"
            );
        }
        if op.defs().next().is_some() || !op.uses().is_empty() {
            let remapped =
                op.remap_variables(|value| Some(SsaVarId::from_index(value.index() + 1024)));
            let mut changed = crate::ir::content::tests::Probe::new(usize::MAX);
            assert_eq!(remapped.write_content(&mut changed), Ok(()));
            assert_ne!(
                complete.data, changed.data,
                "SSA identities omitted for {op}"
            );
        }
    }
}

/// Builds a nested operation with distinct masks in its construction order.
fn raise_chain(depth: usize, masked: bool) -> SsaOp<MockTarget> {
    let v = SsaVarId::from_index;
    let mut operation = SsaOp::Add {
        dest: v(7),
        left: v(1),
        right: v(2),
        flags: Some(v(8)),
    };
    for index in 0..depth {
        operation = SsaOp::FloatRaise {
            dest: v(9),
            op: Box::new(operation),
            lanes: (masked && index % 3 == 0).then(|| v(index + 16)),
            raises: FloatExceptions::NONE,
            layout: FloatExceptionLayout::IEEE,
        };
    }
    operation
}

/// Drops a deep operation without recursive destruction, including before assertions.
fn drop_raise_chain(mut operation: SsaOp<MockTarget>) {
    loop {
        operation = match operation {
            SsaOp::FloatRaise { op, .. } => *op,
            _ => break,
        };
    }
}

/// Deep immutable and mutable walks retain one root definition and inner-first masks.
#[test]
fn deeply_nested_raise_operands_use_constant_stack_and_exact_order() {
    let depth = 40_000;
    for masked in [false, true] {
        let mut operation = raise_chain(depth, masked);
        let v = SsaVarId::from_index;
        let mut expected = vec![
            (OperandRole::Def, v(9)),
            (OperandRole::Use, v(1)),
            (OperandRole::Use, v(2)),
        ];
        if masked {
            expected.extend(
                (0..depth)
                    .filter(|index| index % 3 == 0)
                    .map(|index| (OperandRole::Use, v(index + 16))),
            );
        }
        let mut actual = Vec::new();
        operation.visit_operands(|role, var| actual.push((role, var)));
        let mut definitions = Vec::new();
        let definitions_result = operation.try_for_each_def(|var| {
            definitions.push(var);
            Ok::<(), ()>(())
        });
        let mut mutable = Vec::new();
        operation.visit_operands_mut(|role, var| {
            mutable.push((role, *var));
            *var = v(var.index() + 100_000);
        });
        let mut changed = Vec::new();
        operation.visit_operands(|role, var| changed.push((role, var)));
        let effect = operation.stack_effect();
        // Inspect every nested definition rather than relying on the public walk,
        // which deliberately suppresses those payload definitions.
        let mut node = &operation;
        let mut renamed = true;
        while let SsaOp::FloatRaise { dest, op, .. } = node {
            renamed &= *dest == v(100_009);
            node = op;
        }
        renamed &= matches!(node, SsaOp::Add { dest, flags: Some(flags), .. }
            if *dest == v(100_009) && *flags == v(100_009));
        drop_raise_chain(operation);
        assert_eq!(definitions_result, Ok(()));
        assert_eq!(definitions, vec![v(9)]);
        assert_eq!(actual, expected);
        assert_eq!(mutable, expected);
        assert_eq!(
            changed,
            expected
                .iter()
                .map(|(role, var)| (*role, v(var.index() + 100_000)))
                .collect::<Vec<_>>()
        );
        assert_eq!(effect, (2, 1));
        assert!(renamed);
    }
}

/// An operation formatter that refuses before exceeding its byte allowance.
struct RefusingOperationWriter {
    /// Bytes still accepted by the sink.
    remaining: usize,
    /// Whether the first refusal has occurred.
    refused: bool,
    /// Calls made after that refusal, which must remain zero.
    after_refusal: usize,
}

impl fmt::Write for RefusingOperationWriter {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.refused {
            self.after_refusal += 1;
            return Err(fmt::Error);
        }
        if text.len() > self.remaining {
            self.refused = true;
            return Err(fmt::Error);
        }
        self.remaining -= text.len();
        Ok(())
    }
}

/// Streaming preserves exact prefix elision and mask order, including mismatched definitions.
#[test]
fn nested_raise_display_preserves_exact_prefixes_and_mask_order() {
    let v = SsaVarId::from_index;
    for (inner, expected) in [
        (9, "v9 = float.raise (float.raise (v1) lanes=v3) lanes=v4"),
        (
            8,
            "v9 = float.raise (v8 = float.raise (v9 = v1) lanes=v3) lanes=v4",
        ),
    ] {
        let operation = SsaOp::<MockTarget>::FloatRaise {
            dest: v(9),
            lanes: Some(v(4)),
            raises: FloatExceptions::NONE,
            layout: FloatExceptionLayout::IEEE,
            op: Box::new(SsaOp::FloatRaise {
                dest: v(inner),
                lanes: Some(v(3)),
                raises: FloatExceptions::NONE,
                layout: FloatExceptionLayout::IEEE,
                op: Box::new(SsaOp::Copy {
                    dest: v(9),
                    src: v(1),
                }),
            }),
        };
        assert_eq!(operation.to_string(), expected);
        // Every output boundary includes the streamed leaf and closing suffix,
        // unlike the deep probe's early refusals within wrapper headers.
        for remaining in 0..=expected.len() {
            let mut writer = RefusingOperationWriter {
                remaining,
                refused: false,
                after_refusal: 0,
            };
            let result = write!(&mut writer, "{operation}");
            assert_eq!(
                result,
                if remaining < expected.len() {
                    Err(fmt::Error)
                } else {
                    Ok(())
                }
            );
            assert_eq!(writer.refused, remaining < expected.len());
            assert_eq!(writer.after_refusal, 0);
        }
    }
}

/// Deep formatting is linear in output size and stops immediately at sink refusal.
#[test]
fn deeply_nested_raise_display_streams_and_propagates_refusal() {
    let depth = 40_000;
    let operation = raise_chain(depth, true);
    let rendered = operation.to_string();
    let mut results = Vec::new();
    for remaining in [0, 1, 5, 20, 128, 1024] {
        let mut writer = RefusingOperationWriter {
            remaining,
            refused: false,
            after_refusal: 0,
        };
        results.push((
            write!(&mut writer, "{operation}"),
            writer.refused,
            writer.after_refusal,
        ));
    }
    drop_raise_chain(operation);
    assert_eq!(rendered.matches("float.raise").count(), depth);
    assert_eq!(rendered.matches('(').count(), depth);
    assert_eq!(rendered.matches(')').count(), depth);
    assert_eq!(rendered.matches(" lanes=").count(), depth.div_ceil(3));
    for (result, refused, after_refusal) in results {
        assert_eq!(result, Err(fmt::Error));
        assert!(refused);
        assert_eq!(after_refusal, 0);
    }
}

/// Precise no-memory locations refine every coarse effect class without deleting its control effect.
#[test]
fn memory_contract_queries_respect_explicit_no_memory() {
    for (kind, reads, writes) in [
        (SsaEffectKind::Pure, false, false),
        (SsaEffectKind::Read, true, false),
        (SsaEffectKind::Write, false, true),
        (SsaEffectKind::ReadWrite, true, true),
        (SsaEffectKind::Atomic, true, true),
        (SsaEffectKind::Call, true, true),
        (SsaEffectKind::Opaque, true, true),
        (SsaEffectKind::Fence, false, false),
        (SsaEffectKind::RuntimeDefined, false, false),
    ] {
        let ordinary = SsaEffects::new(kind, false);
        assert_eq!(ordinary.reads_memory(), reads);
        assert_eq!(ordinary.writes_memory(), writes);
        let isolated = ordinary.with_memory(MemoryEffectLocation::None);
        assert!(!isolated.reads_memory(), "{kind:?}");
        assert!(!isolated.writes_memory(), "{kind:?}");
        assert_eq!(isolated.kind, ordinary.kind);
        assert_eq!(isolated.control, ordinary.control);
    }
}

/// Explicit control-flag values touch no data memory and remain return dependencies.
#[test]
fn memory_contract_control_flags_have_no_data_memory_access() {
    for mask in [
        FlagsMask::DIRECTION,
        FlagsMask::ALIGN_CHECK,
        FlagsMask::INTERRUPT,
    ] {
        let op = SsaOp::<MockTarget>::InsertFlag {
            dest: SsaVarId::from_index(2),
            base: SsaVarId::from_index(0),
            bit: SsaVarId::from_index(1),
            mask,
        };
        let effects = op.effects();
        assert!(effects.is_pure());
        assert_eq!(effects.memory, MemoryEffectLocation::None);
        assert!(!effects.reads_memory());
        assert!(!effects.writes_memory());
        let returned = SsaOp::<MockTarget>::returns_with(None, vec![SsaVarId::from_index(2)]);
        assert_eq!(returned.uses(), vec![SsaVarId::from_index(2)]);
        assert!(!returned.effects().is_pure());
    }
}

/// Return-state visitation stops at the first refusal, retaining operand order.
#[test]
fn return_state_visitation_propagates_each_refusal() {
    let v = SsaVarId::from_index;
    let operation = SsaOp::<MockTarget>::returns_with(Some(v(0)), vec![v(1), v(2)]);
    let expected = [
        (OperandRole::Use, v(0)),
        (OperandRole::Use, v(1)),
        (OperandRole::Use, v(2)),
    ];
    for limit in 0..=expected.len() {
        let mut seen = Vec::new();
        let result = operation.try_visit_operands(|role, value| {
            if seen.len() == limit {
                return Err(());
            }
            seen.push((role, value));
            Ok(())
        });
        assert_eq!(seen, expected[..limit]);
        assert_eq!(
            result,
            if limit == expected.len() {
                Ok(())
            } else {
                Err(())
            }
        );
    }
}

/// A shuffle by an index vector reads its index after its inputs and names
/// it apart from them.
#[test]
fn an_index_shuffle_reads_and_names_its_index() {
    let [dest, left, right, index] = [0, 1, 2, 3].map(SsaVarId::from_index);
    let mut op = SsaOp::<MockTarget>::VectorShuffle {
        dest,
        left,
        right: Some(right),
        selector: VectorShuffleSelector::Index(index),
    };
    assert_eq!(op.uses(), vec![left, right, index]);
    assert_eq!(op.stack_effect(), (3, 1));
    assert_eq!(op.to_string(), "v0 = vshuffle v1, v2 by v3");
    let replacement = SsaVarId::from_index(4);
    assert_eq!(op.replace_uses(index, replacement), 1);
    assert_eq!(
        op,
        SsaOp::VectorShuffle {
            dest,
            left,
            right: Some(right),
            selector: VectorShuffleSelector::Index(replacement),
        }
    );
    let single = SsaOp::<MockTarget>::VectorShuffle {
        dest,
        left,
        right: None,
        selector: VectorShuffleSelector::Index(index),
    };
    assert_eq!(single.to_string(), "v0 = vshuffle v1 by v3");
    assert_eq!(single.stack_effect(), (2, 1));
}

/// **A read-check-write's check is Arm's `RCWCheck`.** Each case worked from
/// the 2026-09 shared pseudocode: a protected, valid 64-bit value (bits 52
/// and 0 set) changes only bits `RCWMASK_EL1` allows and keeps both bits; an
/// unprotected one keeps its protected bit and may change anything else; the
/// mask's bit 17 stands for bits 18 to 49 and its bit 0 for nothing; the soft
/// check of a valid value keeps bit 0 and changes only bits `RCWSMASK_EL1`
/// allows, the protected bit never; a 128-bit value's protected bit is 114
/// and its mask never allows the format's reserved bits.
#[test]
fn a_read_check_write_check_is_arms_rcwcheck() {
    let hard = RcwCheck {
        bits: 64,
        soft: false,
        protected: true,
    };
    let valid: u128 = (1 << 52) | 1;
    let allowed = 1u128 << 5;
    // Inside the mask: passes. Outside it: the check fails (Z), soft passes.
    assert_eq!(hard.status(valid, valid | (1 << 5), allowed, 0), 0b0010);
    assert_eq!(hard.status(valid, valid | (1 << 6), allowed, 0), 0b0110);
    // A protected value keeps its valid bit and its protected bit, whatever
    // the mask allows.
    assert_eq!(hard.status(valid, 1 << 52, u128::MAX, 0), 0b0110);
    assert_eq!(hard.status(valid, 1, u128::MAX, 0), 0b0110);
    // An unprotected value keeps its protected bit, and nothing else is
    // checked: no mask allows bit 6, yet it changes.
    assert_eq!(hard.status(1, 1 | (1 << 52), 0, 0), 0b0110);
    assert_eq!(hard.status(1, 1 | (1 << 6), 0, 0), 0b0010);
    // Without protection there is no read-check-write check at all.
    let unprotected = RcwCheck {
        protected: false,
        ..hard
    };
    assert_eq!(unprotected.status(valid, 0, 0, 0), 0b0010);
    // Bit 17 of the mask stands for bits 18 to 49; bit 0 for none.
    assert_eq!(hard.status(valid, valid | (1 << 30), 1 << 17, 0), 0b0010);
    assert_eq!(hard.status(valid, valid | (1 << 30), 1 << 30, 0), 0b0110);
    assert_eq!(hard.effective_mask(1, false), 0);
    // The soft check of a valid value: bit 0 kept, changes inside the soft
    // mask, whose protected bit is never allowed while protection is on.
    let soft = RcwCheck { soft: true, ..hard };
    let both = allowed | (1 << 52);
    assert_eq!(soft.status(valid, valid | (1 << 5), both, both), 0b0010);
    assert_eq!(soft.status(valid, valid | (1 << 5), both, 0), 0b0000);
    assert_eq!(soft.effective_mask(1 << 52, true), 0);
    assert_eq!(
        RcwCheck {
            protected: false,
            ..soft
        }
        .effective_mask(1 << 52, true),
        1 << 52
    );
    // An invalid unprotected value's soft check keeps only bit 0.
    assert_eq!(soft.status(0, 1, 0, 0), 0b0000);
    assert_eq!(soft.status(0, 1 << 6, 0, 0), 0b0010);
    // 128 bits: the protected bit is 114, and the reserved bits (56 to 90,
    // among others) stay out of the mask whatever the register holds.
    let wide = RcwCheck {
        bits: 128,
        soft: false,
        protected: true,
    };
    let valid: u128 = (1 << 114) | 1;
    assert_eq!(wide.protected_bit(), 114);
    assert_eq!(wide.status(valid, valid | (1 << 60), u128::MAX, 0), 0b0110);
    assert_eq!(wide.status(valid, valid | (1 << 100), u128::MAX, 0), 0b0010);
    assert_eq!(wide.status(valid, valid | (1 << 40), 1 << 16, 0), 0b0010);
}
