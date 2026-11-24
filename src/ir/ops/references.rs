//! The references into the host's metadata an operation names.
//!
//! Code generated from an operation writes these names (a field, a type
//! token, a call's method or signature), so whatever removes metadata asks
//! [`SsaOp::visit_references`] what the surviving code still names. The match
//! lists every operation: one that gains a reference cannot be forgotten here.

use crate::{
    ir::{
        ops::{Callee, SsaOp},
        value::ConstValue,
    },
    target::Target,
};

/// A reference into the host's metadata that an operation names.
#[derive(Debug)]
pub enum Reference<'a, T: Target> {
    /// A type, by its metadata reference.
    Type(&'a T::TypeRef),
    /// A method.
    Method(&'a T::MethodRef),
    /// A field.
    Field(&'a T::FieldRef),
    /// A standalone signature: an indirect call's.
    Signature(&'a T::SigRef),
    /// A symbol a constant names.
    Symbol(&'a T::SymbolRef),
    /// A host type an operand or result is stated in (a conversion's target,
    /// an element's or an access's type), which may name references of its
    /// own; the host reads them out of it.
    HostType(&'a T::Type),
}

impl<T: Target> SsaOp<T> {
    /// Calls `visit` with every reference into the host's metadata the
    /// operation names ([`Reference`]): a call's callee (its method, a
    /// constrained call's constraint type, an indirect call's signature), a
    /// constant's handle, symbol or array element type, a field, a type token,
    /// and every host type the operation states. A call's stated effects are
    /// facts about the code it runs, not names it writes, and are not
    /// visited.
    pub fn visit_references<'a>(&'a self, mut visit: impl FnMut(Reference<'a, T>)) {
        match self {
            Self::Const { value, .. } => match value {
                ConstValue::Type(type_ref) => visit(Reference::Type(type_ref)),
                ConstValue::MethodHandle(method) => visit(Reference::Method(method)),
                ConstValue::FieldHandle(field) => visit(Reference::Field(field)),
                ConstValue::Symbol(symbol) => visit(Reference::Symbol(symbol)),
                ConstValue::DecryptedArray(array) => {
                    visit(Reference::Type(&array.element_type_ref));
                }
                ConstValue::I8(_)
                | ConstValue::I16(_)
                | ConstValue::I32(_)
                | ConstValue::I64(_)
                | ConstValue::U8(_)
                | ConstValue::U16(_)
                | ConstValue::U32(_)
                | ConstValue::U64(_)
                | ConstValue::NativeInt(_)
                | ConstValue::NativeUInt(_)
                | ConstValue::Float(_)
                | ConstValue::Vector(_)
                | ConstValue::String(_)
                | ConstValue::DecryptedString(_)
                | ConstValue::Null
                | ConstValue::True
                | ConstValue::False => {}
            },
            Self::Call(call) => match &call.callee {
                Callee::Direct(method) | Callee::Virtual(method) => {
                    visit(Reference::Method(method));
                }
                Callee::Constrained {
                    method,
                    constraint_type,
                    ..
                } => {
                    visit(Reference::Method(method));
                    visit(Reference::Type(constraint_type));
                }
                Callee::Indirect { signature, .. } => visit(Reference::Signature(signature)),
            },
            Self::LoadFunctionPtr {
                method, constraint, ..
            } => {
                visit(Reference::Method(method));
                if let Some(constraint) = constraint {
                    visit(Reference::Type(constraint));
                }
            }
            Self::NewObj { ctor: method, .. } | Self::LoadVirtFunctionPtr { method, .. } => {
                visit(Reference::Method(method));
            }
            Self::LoadField { field, .. }
            | Self::StoreField { field, .. }
            | Self::LoadStaticField { field, .. }
            | Self::StoreStaticField { field, .. }
            | Self::LoadFieldAddr { field, .. }
            | Self::LoadStaticFieldAddr { field, .. } => visit(Reference::Field(field)),
            Self::LoadElementAddr {
                elem_type: type_ref,
                ..
            }
            | Self::NewArr {
                elem_type: type_ref,
                ..
            }
            | Self::CastClass {
                target_type: type_ref,
                ..
            }
            | Self::IsInst {
                target_type: type_ref,
                ..
            }
            | Self::Box {
                value_type: type_ref,
                ..
            }
            | Self::Unbox {
                value_type: type_ref,
                ..
            }
            | Self::UnboxAny {
                value_type: type_ref,
                ..
            }
            | Self::SizeOf {
                value_type: type_ref,
                ..
            }
            | Self::LoadToken {
                token: type_ref, ..
            }
            | Self::MakeTypedRef {
                value_type: type_ref,
                ..
            }
            | Self::TypedRefValue {
                value_type: type_ref,
                ..
            }
            | Self::InitObj {
                value_type: type_ref,
                ..
            }
            | Self::CopyObj {
                value_type: type_ref,
                ..
            }
            | Self::LoadObj {
                value_type: type_ref,
                ..
            }
            | Self::StoreObj {
                value_type: type_ref,
                ..
            } => visit(Reference::Type(type_ref)),
            Self::IntConv { target, .. }
            | Self::IntToPtr { target, .. }
            | Self::PtrToInt { target, .. }
            | Self::IntToFloat { target, .. }
            | Self::FloatToInt { target, .. }
            | Self::FloatConv { target, .. }
            | Self::Bitcast { target, .. }
            | Self::PtrAdd {
                result_type: target,
                ..
            }
            | Self::VectorLoad {
                vector_type: target,
                ..
            }
            | Self::VectorStore {
                vector_type: target,
                ..
            }
            | Self::VectorMaskedLoad {
                vector_type: target,
                ..
            }
            | Self::VectorMaskedStore {
                vector_type: target,
                ..
            }
            | Self::VectorBroadcastLoad {
                vector_type: target,
                ..
            }
            | Self::VectorGather {
                vector_type: target,
                ..
            }
            | Self::VectorFaultingLoad {
                vector_type: target,
                ..
            }
            | Self::VectorSegmentLoad {
                vector_type: target,
                ..
            }
            | Self::VectorScatter {
                vector_type: target,
                ..
            }
            | Self::VectorSegmentStore {
                vector_type: target,
                ..
            }
            | Self::VectorSplat {
                vector_type: target,
                ..
            }
            | Self::VectorCast {
                target_type: target,
                ..
            }
            | Self::VectorReinterpret {
                target_type: target,
                ..
            }
            | Self::VectorPack {
                vector_type: target,
                ..
            }
            | Self::VectorPackLoad {
                vector_type: target,
                ..
            }
            | Self::VectorPackStore {
                vector_type: target,
                ..
            }
            | Self::LoadElement {
                elem_type: target, ..
            }
            | Self::StoreElement {
                elem_type: target, ..
            }
            | Self::LoadIndirect {
                value_type: target, ..
            }
            | Self::StoreIndirect {
                value_type: target, ..
            }
            | Self::AtomicLoad {
                value_type: target, ..
            }
            | Self::AtomicStore {
                value_type: target, ..
            }
            | Self::AtomicStoreConditional {
                value_type: target, ..
            }
            | Self::VectorSlice {
                target_type: target,
                ..
            }
            | Self::VectorInsertSlice {
                target_type: target,
                ..
            }
            | Self::VectorConcat {
                target_type: target,
                ..
            } => visit(Reference::HostType(target)),
            Self::AtomicPairLoad {
                first_type,
                second_type,
                ..
            }
            | Self::AtomicPairStoreConditional {
                first_type,
                second_type,
                ..
            } => {
                visit(Reference::HostType(first_type));
                visit(Reference::HostType(second_type));
            }
            Self::Add { .. }
            | Self::AddOvf { .. }
            | Self::Sub { .. }
            | Self::SubOvf { .. }
            | Self::Mul { .. }
            | Self::MulOvf { .. }
            | Self::WideMul { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::FloatCompareFlags { .. }
            | Self::FloatRaise { .. }
            | Self::WideDiv { .. }
            | Self::Neg { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Not { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. }
            | Self::Ceq { .. }
            | Self::Clt { .. }
            | Self::Cgt { .. }
            | Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. }
            | Self::Select { .. }
            | Self::ReadFlags { .. }
            | Self::MergeFlags { .. }
            | Self::InsertFlag { .. }
            | Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. }
            | Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::BranchFlags { .. }
            | Self::Switch { .. }
            | Self::IndirectBranch { .. }
            | Self::Return { .. }
            | Self::ArrayLength { .. }
            | Self::ArgList { .. }
            | Self::ArgNext { .. }
            | Self::ArgStep { .. }
            | Self::TypedRefType { .. }
            | Self::LoadArg { .. }
            | Self::LoadLocal { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::Copy { .. }
            | Self::Pop { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::EndFinally
            | Self::EndFilter { .. }
            | Self::RuntimeValue { .. }
            | Self::InterruptReturn
            | Self::Unreachable
            | Self::Leave { .. }
            | Self::InitBlk { .. }
            | Self::MoveBlk { .. }
            | Self::CopyBlk { .. }
            | Self::Fence { .. }
            | Self::SystemOp(_)
            | Self::Compute(_)
            | Self::Contract(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorStructLoadReplicate(_)
            | Self::VectorSveCompute(_)
            | Self::CompareBlk(_)
            | Self::ScanBlk { .. }
            | Self::LoadBlk { .. }
            | Self::CallClobber { .. }
            | Self::AtomicExchange { .. }
            | Self::AtomicLockRmw { .. }
            | Self::AtomicCondRmw { .. }
            | Self::AtomicCmpXchg { .. }
            | Self::AtomicPairCmpXchg { .. }
            | Self::AtomicPairRmw { .. }
            | Self::AtomicCheckedRmw { .. }
            | Self::Nop
            | Self::Break(_)
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::FloatFunction(_)
            | Self::FpuControl(_)
            | Self::LocalAlloc { .. }
            | Self::Phi { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{ops::CallData, variable::SsaVarId},
        testing::MockTarget,
    };

    fn references(op: &SsaOp<MockTarget>) -> Vec<String> {
        let mut names = Vec::new();
        op.visit_references(|reference| names.push(format!("{reference:?}")));
        names
    }

    /// **An operation names what its code writes.** A constrained call names
    /// its method and its constraint type, an indirect call its signature, a
    /// handle constant what it holds, a typed reference its type, a
    /// constrained function pointer its method and type; arithmetic names
    /// nothing.
    #[test]
    fn an_operation_names_the_references_its_code_writes() {
        let v = SsaVarId::from_index;
        let constrained: SsaOp<MockTarget> = CallData::new(
            None,
            Callee::Constrained {
                method: 9,
                constraint_type: 3,
                instance: true,
            },
            vec![v(0)],
        )
        .into();
        assert_eq!(references(&constrained), ["Method(9)", "Type(3)"]);
        let indirect: SsaOp<MockTarget> = CallData::new(
            None,
            Callee::Indirect {
                fptr: v(0),
                signature: 5,
            },
            vec![],
        )
        .into();
        assert_eq!(references(&indirect), ["Signature(5)"]);
        let handle = SsaOp::<MockTarget>::Const {
            dest: v(0),
            value: ConstValue::FieldHandle(7),
        };
        assert_eq!(references(&handle), ["Field(7)"]);
        let typed_ref = SsaOp::<MockTarget>::MakeTypedRef {
            dest: v(0),
            address: v(1),
            value_type: 4,
        };
        assert_eq!(references(&typed_ref), ["Type(4)"]);
        let static_virtual = SsaOp::<MockTarget>::LoadFunctionPtr {
            dest: v(0),
            method: 9,
            constraint: Some(3),
        };
        assert_eq!(references(&static_virtual), ["Method(9)", "Type(3)"]);
        let add = SsaOp::<MockTarget>::Add {
            dest: v(0),
            left: v(1),
            right: v(2),
            flags: None,
        };
        assert!(references(&add).is_empty());
    }
}
