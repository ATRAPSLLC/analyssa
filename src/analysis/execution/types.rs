//! Numeric layouts of a body's variables, derived once and shared by every path.
//!
//! Every activation and path of one body asks the same layout questions of the
//! same variables and operations. [`TypeLayouts`] answers each once, lazily and
//! keyed by the variable or the
//! instruction position, which the immutable body fixes. Host metadata is
//! never cloned, hashed or compared, and type graphs no query needs are never
//! visited.

use std::{
    cell::{Cell, RefCell},
    fmt,
};

use rustc_hash::FxHashMap;

use crate::{
    analysis::{
        consts::{Destination, OperandLayout},
        execution::body::SourceBody,
    },
    ir::{ops::SsaOp, variable::SsaVarId, varstore::VarMap},
    pointer::PointerSize,
    target::{Target, TypeLayout},
};

/// Borrows the lazy layout query for one operation's explicit target type.
#[derive(Clone, Copy)]
pub struct OperationLayout<'a>(&'a dyn Fn() -> Option<TypeLayout>);

impl fmt::Debug for OperationLayout<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("OperationLayout(..)")
    }
}

impl<'a> OperationLayout<'a> {
    /// Binds one immutable instruction's target query for its evaluation.
    pub const fn new(query: &'a dyn Fn() -> Option<TypeLayout>) -> Self {
        Self(query)
    }

    /// Reads the target layout for the current instruction.
    pub fn read(self) -> Option<TypeLayout> {
        self.0()
    }
}

/// Holds numeric layouts by stable variable identity and instruction position.
///
/// The immutable body determines every key. Host metadata is never cloned,
/// hashed or compared, and unqueried type graphs are never inspected.
#[derive(Debug)]
pub struct TypeLayouts<'a, T: Target> {
    /// Body whose variable identities these slots describe.
    ssa: SourceBody<'a, T>,
    /// Machine width fixing native integer representations in this cache.
    ptr_size: PointerSize,
    /// Lazily populated layouts, shared across paths and calls to this body.
    layouts: RefCell<VarMap<TypeLayout>>,
    /// Lazily populated explicit target layouts for instructions in this body.
    targets: RefCell<FxHashMap<(usize, usize), TypeLayout>>,
    /// Whether a required host type has no supported numeric interpretation.
    unsupported: Cell<bool>,
}

impl<'a, T: Target> TypeLayouts<'a, T> {
    /// Retains a body's storage independently of the explorer's body table.
    pub fn of_body(ssa: SourceBody<'a, T>, ptr_size: PointerSize) -> Self {
        let layouts = VarMap::new(ssa.var_id_bound());
        Self {
            ssa,
            ptr_size,
            layouts: RefCell::new(layouts),
            targets: RefCell::new(FxHashMap::default()),
            unsupported: Cell::new(false),
        }
    }

    /// Returns whether a required type query has failed.
    pub fn is_unsupported(&self) -> bool {
        self.unsupported.get()
    }

    /// Projects a borrowed host type and retains unavailable semantics for the executor.
    fn project_type(&self, ty: &T::Type) -> Option<TypeLayout> {
        let layout = T::type_layout(ty, self.ptr_size);
        if layout.is_none() {
            self.unsupported.set(true);
        }
        layout
    }

    /// Reads or projects one instruction target without inspecting unrelated types.
    ///
    #[inline]
    pub fn operation(&self, block: usize, instruction: usize) -> Option<TypeLayout> {
        let key = (block, instruction);
        if let Some(layout) = self.targets.borrow().get(&key).copied() {
            return Some(layout);
        }
        self.project_operation(block, instruction)
    }

    /// Projects and caches a fresh instruction target.
    #[cold]
    fn project_operation(&self, block: usize, instruction: usize) -> Option<TypeLayout> {
        let key = (block, instruction);

        let op = self.ssa.block(block)?.instructions().get(instruction)?.op();
        let layout = self.project_type(op.stated_type()?)?;

        self.targets.borrow_mut().insert(key, layout);
        Some(layout)
    }

    /// Reads or projects one variable by its stable identity.
    ///
    #[inline]
    pub fn get(&self, variable: SsaVarId) -> Option<TypeLayout> {
        if let Some(layout) = self.layouts.borrow().get(variable).copied() {
            return Some(layout);
        }
        self.project_variable(variable)
    }

    /// Projects and caches a variable without inspecting unrelated types.
    #[cold]
    fn project_variable(&self, variable: SsaVarId) -> Option<TypeLayout> {
        let ty = self.ssa.variable(variable)?.var_type();
        let layout = self.project_type(ty)?;
        // A minimum vector shape does not provide a runtime value extent.
        if layout.is_runtime_sized() {
            self.unsupported.set(true);
            return None;
        }
        self.layouts.borrow_mut().insert(variable, layout);
        Some(layout)
    }

    /// Reads the destination's declared facts without unrestricted host getters.
    pub fn destination(&self, op: &SsaOp<T>) -> Destination {
        op.dest()
            .and_then(|var| self.get(var))
            .map_or_else(Destination::default, Destination::from_type_layout)
    }

    /// Reads an input's numeric interpretation from this body's cache.
    pub fn operand(&self, variable: SsaVarId) -> OperandLayout {
        self.get(variable)
            .map_or_else(OperandLayout::default, OperandLayout::from_type_layout)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        hash::{Hash, Hasher},
        rc::Rc,
        sync::Arc,
    };

    use super::*;
    use crate::{
        analysis::{
            Machine, SsaVerifier, VerifyLevel,
            context::AnalysisContext,
            execution::{
                Address, Completion, ExecutionHost, ExtentSource, ObjectExtents, ObjectKind,
                StaticExecution, StorageObject, Value,
            },
        },
        ir::{
            exception::HandlerKind,
            float::FloatFormat,
            function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
            ops::CmpKind,
            value::ConstValue,
        },
        target::{BooleanStorage, Endianness},
        testing::{MockTarget, MockType},
    };

    #[derive(Debug, Default)]
    struct Probes {
        inspected: Cell<usize>,
        forbid_raw: Cell<bool>,
        /// The target states no layout for any type.
        declined: Cell<bool>,
    }

    #[derive(Debug)]
    struct ProbedType {
        kind: MockType,
        probes: Rc<Probes>,
    }

    impl ProbedType {
        fn raw(&self) {
            assert!(
                !self.probes.forbid_raw.get(),
                "unrestricted type metadata access"
            );
        }
    }
    impl Clone for ProbedType {
        fn clone(&self) -> Self {
            self.raw();
            Self {
                kind: self.kind,
                probes: Rc::clone(&self.probes),
            }
        }
    }
    impl PartialEq for ProbedType {
        fn eq(&self, other: &Self) -> bool {
            self.raw();
            other.raw();
            self.kind == other.kind
        }
    }
    impl Eq for ProbedType {}
    impl Hash for ProbedType {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.raw();
            self.kind.hash(state);
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct ProbedTarget;

    impl Target for ProbedTarget {
        type TypeRef = u32;
        type MethodRef = u32;
        type FieldRef = u32;
        type SigRef = u32;
        type SymbolRef = u32;
        type ExceptionKind = ();
        type Type = ProbedType;
        type OriginalInstruction = ();
        type LocalSignature = ();
        type VariableFacts = ();
        type InstructionFacts = ();
        type FunctionFacts = ();
        type Capability = ();
        fn ptr_bytes(&self) -> u32 {
            8
        }
        fn synthetic_instruction() {}
        fn unknown_type() -> ProbedType {
            ProbedType {
                kind: MockType::Unknown,
                probes: Rc::default(),
            }
        }
        fn instruction_mnemonic(_: &()) -> &'static str {
            "fixture"
        }
        fn instruction_rva(_: &()) -> u64 {
            0
        }
        fn handler_kind(_: &()) -> HandlerKind {
            HandlerKind::Catch
        }
        fn is_boolean(ty: &ProbedType) -> bool {
            ty.raw();
            MockTarget::is_boolean(&ty.kind)
        }
        fn boolean_storage(ty: &ProbedType) -> Option<BooleanStorage> {
            ty.raw();
            MockTarget::boolean_storage(&ty.kind)
        }
        fn is_integer(ty: &ProbedType) -> bool {
            ty.raw();
            MockTarget::is_integer(&ty.kind)
        }
        fn is_floating(ty: &ProbedType) -> bool {
            ty.raw();
            MockTarget::is_floating(&ty.kind)
        }
        fn is_signed(ty: &ProbedType) -> bool {
            ty.raw();
            MockTarget::is_signed(&ty.kind)
        }
        fn is_pointer(ty: &ProbedType) -> bool {
            ty.raw();
            MockTarget::is_pointer(&ty.kind)
        }
        fn is_reference(ty: &ProbedType) -> bool {
            ty.raw();
            MockTarget::is_reference(&ty.kind)
        }
        fn is_unknown(ty: &ProbedType) -> bool {
            ty.raw();
            MockTarget::is_unknown(&ty.kind)
        }
        fn bit_width(ty: &ProbedType) -> Option<u32> {
            ty.raw();
            MockTarget::bit_width(&ty.kind)
        }
        fn float_format(ty: &ProbedType) -> Option<FloatFormat> {
            ty.raw();
            MockTarget::float_format(&ty.kind)
        }
        fn type_layout(ty: &ProbedType, ptr_size: PointerSize) -> Option<TypeLayout> {
            ty.probes.inspected.set(ty.probes.inspected.get() + 1);
            (!ty.probes.declined.get()).then(|| TypeLayout::of::<MockTarget>(&ty.kind, ptr_size))
        }
    }

    fn fixture(kind: MockType, probes: &Rc<Probes>) -> (SsaFunction<ProbedTarget>, SsaVarId) {
        let mut builder = SsaFunctionBuilder::<ProbedTarget>::new(1, 0);
        let mut variable = None;
        builder
            .in_block(0, |block| {
                let value = match kind {
                    MockType::U64 => ConstValue::U64(1),
                    MockType::Ptr => ConstValue::NativeUInt(1),
                    _ => ConstValue::I32(1),
                };
                let v = block.const_value(
                    SsaDefSpec::tmp(ProbedType {
                        kind,
                        probes: Rc::clone(probes),
                    }),
                    value,
                )?;
                variable = Some(v);
                block.ret(Some(v))
            })
            .unwrap();
        let ssa = builder.finish().unwrap();
        assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        (ssa, variable.unwrap())
    }

    /// Cache creation is lazy, and repeated reads never re-enter host type inspection.
    #[test]
    fn variable_layout_cache_is_lazy_and_reuses_its_layouts() {
        let probes = Rc::new(Probes::default());
        let (ssa, variable) = fixture(MockType::I32, &probes);
        probes.forbid_raw.set(true);

        let layouts = TypeLayouts::of_body(SourceBody::Borrowed(&ssa), PointerSize::Bit64);
        assert_eq!(probes.inspected.get(), 0);
        for _ in 0..128 {
            assert_eq!(layouts.get(variable).unwrap().bit_width(), Some(32));
        }
        assert_eq!(probes.inspected.get(), 1);
    }

    /// Equal local variable ids do not share type facts across bodies or machine widths.
    #[test]
    fn variable_layout_cache_preserves_body_and_machine_scope() {
        let probes = Rc::new(Probes::default());
        let (first, a) = fixture(MockType::I32, &probes);
        let (second, b) = fixture(MockType::U64, &probes);
        let (pointers, p) = fixture(MockType::Ptr, &probes);
        probes.forbid_raw.set(true);

        let left = TypeLayouts::of_body(SourceBody::Borrowed(&first), PointerSize::Bit64);
        let right = TypeLayouts::of_body(SourceBody::Borrowed(&second), PointerSize::Bit64);
        assert_eq!(a, b);
        assert_eq!(left.get(a).unwrap().bit_width(), Some(32));
        assert_eq!(right.get(b).unwrap().bit_width(), Some(64));
        for width in [PointerSize::Bit32, PointerSize::Bit64] {
            let cache = TypeLayouts::of_body(SourceBody::Borrowed(&pointers), width);
            assert_eq!(
                cache
                    .get(p)
                    .unwrap()
                    .integer_conversion_type()
                    .unwrap()
                    .bits(),
                width.bits()
            );
        }
        assert_eq!(probes.inspected.get(), 4);
    }

    /// A verified function executes without raw type getters, clones, equality or hashing.
    #[test]
    fn execution_variable_queries_use_only_projected_layouts() {
        let probes = Rc::new(Probes::default());
        let (ssa, _) = fixture(MockType::I32, &probes);
        probes.forbid_raw.set(true);
        let machine = Machine::new(PointerSize::Bit64, None, Default::default());
        let context = AnalysisContext::new(machine);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert_eq!(facts.completion(), Completion::Complete);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(1))
        );
        assert_eq!(probes.inspected.get(), 1);
    }

    /// A type the target states no layout for stops the execution instead of
    /// running it with a guessed width.
    #[test]
    fn a_type_without_a_layout_stops_the_execution() {
        let probes = Rc::new(Probes::default());
        let (ssa, _) = fixture(MockType::I32, &probes);
        probes.declined.set(true);
        let machine = Machine::new(PointerSize::Bit64, None, Default::default());
        let context = AnalysisContext::new(machine);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert_eq!(facts.completion(), Completion::UnsupportedType);
        assert!(probes.inspected.get() > 0);
    }

    /// Floating branch comparisons use cached type facts rather than raw metadata.
    #[test]
    fn execution_branch_queries_use_only_projected_layouts() {
        let probes = Rc::new(Probes::default());
        let ty = |kind| ProbedType {
            kind,
            probes: Rc::clone(&probes),
        };
        let mut builder = SsaFunctionBuilder::<ProbedTarget>::new(1, 0);
        builder
            .in_block(0, |block| {
                let first = block.const_value(
                    SsaDefSpec::tmp(ty(MockType::F64)),
                    ConstValue::U64(1.0f64.to_bits()),
                )?;
                let second = block.const_value(
                    SsaDefSpec::tmp(ty(MockType::F64)),
                    ConstValue::U64(2.0f64.to_bits()),
                )?;
                block.branch_cmp(first, second, CmpKind::Lt, false, 1, 2)
            })
            .unwrap();
        for (index, value) in [(1, 1), (2, 2)] {
            builder
                .in_block(index, |block| {
                    let result = block
                        .const_value(SsaDefSpec::tmp(ty(MockType::I32)), ConstValue::I32(value))?;
                    block.ret(Some(result))
                })
                .unwrap();
        }
        let ssa = builder.finish().unwrap();
        assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        probes.forbid_raw.set(true);
        let machine = Machine::new(PointerSize::Bit64, None, Default::default());
        let context = AnalysisContext::new(machine);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert_eq!(facts.completion(), Completion::Complete);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::I32(1))
        );
        assert_eq!(probes.inspected.get(), 3);
    }

    fn target_fixture(probes: &Rc<Probes>) -> SsaFunction<ProbedTarget> {
        let ty = |kind| ProbedType {
            kind,
            probes: Rc::clone(probes),
        };
        let mut builder = SsaFunctionBuilder::<ProbedTarget>::new(1, 0);
        builder
            .in_block(0, |block| {
                let mut input =
                    block.const_value(SsaDefSpec::tmp(ty(MockType::I32)), ConstValue::I32(7))?;
                for kind in [MockType::I64, MockType::U32] {
                    input = block.emit_def(SsaDefSpec::tmp(ty(kind)), |dest| SsaOp::IntConv {
                        dest,
                        operand: input,
                        target: ty(kind),
                        unsigned: false,
                        overflow_check: false,
                    })?;
                }
                block.ret(Some(input))
            })
            .unwrap();
        let ssa = builder.finish().unwrap();
        assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        ssa
    }

    /// Target caches inspect only requested fields and keep distinct instruction positions.
    #[test]
    fn operation_layout_cache_is_lazy_and_preserves_instruction_scope() {
        let probes = Rc::new(Probes::default());
        let ssa = target_fixture(&probes);
        probes.forbid_raw.set(true);

        let cache = TypeLayouts::of_body(SourceBody::Borrowed(&ssa), PointerSize::Bit64);
        assert_eq!(probes.inspected.get(), 0);
        assert!(cache.operation(0, 0).is_none());
        assert_eq!(probes.inspected.get(), 0);
        for _ in 0..128 {
            assert_eq!(cache.operation(0, 1).unwrap().bit_width(), Some(64));
            assert_eq!(cache.operation(0, 2).unwrap().bit_width(), Some(32));
        }
        assert_eq!(probes.inspected.get(), 2);
    }

    /// Shared kernels consume the scoped operation query without any raw host access.
    #[test]
    fn execution_kernel_targets_use_only_projected_layouts() {
        let probes = Rc::new(Probes::default());
        let ssa = target_fixture(&probes);
        probes.forbid_raw.set(true);
        let context =
            AnalysisContext::new(Machine::new(PointerSize::Bit64, None, Default::default()));
        let facts = StaticExecution::new(&ssa, &context).run();
        assert_eq!(facts.completion(), Completion::Complete);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::U32(7))
        );
        assert_eq!(probes.inspected.get(), 5);
    }

    /// Memory footprints, proofs and transfers share one target inspection per instruction.
    #[test]
    fn execution_memory_proofs_share_cached_target_layouts() {
        for (kind, initial) in [
            (MockType::U32, ConstValue::U32(0x1234_5678)),
            (MockType::Bool, ConstValue::True),
            (MockType::Bool, ConstValue::False),
        ] {
            let probes = Rc::new(Probes::default());
            let ty = |kind| ProbedType {
                kind,
                probes: Rc::clone(&probes),
            };
            let mut builder = SsaFunctionBuilder::<ProbedTarget>::new(1, 0);
            builder
                .in_block(0, |block| {
                    let addr = block.const_value(
                        SsaDefSpec::tmp(ty(MockType::Ptr)),
                        ConstValue::NativeUInt(0x1000),
                    )?;
                    let input = block.const_value(SsaDefSpec::tmp(ty(kind)), initial.clone())?;
                    block.store_indirect(addr, input, ty(kind))?;
                    let loaded = block.load_indirect(SsaDefSpec::tmp(ty(kind)), addr, ty(kind))?;
                    block.ret(Some(loaded))
                })
                .unwrap();
            let ssa = builder.finish().unwrap();
            assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
            probes.forbid_raw.set(true);
            let context = AnalysisContext::new(Machine::new(
                PointerSize::Bit64,
                Some(Endianness::Little),
                Default::default(),
            ));
            // With objects stated, every access is judged against the object
            // its pointer was formed into, from the same cached layouts.
            for stated in [false, true] {
                let inspected = probes.inspected.get();
                let execution = StaticExecution::new(&ssa, &context);
                let facts = if stated {
                    execution.with_host(&ObjectAt).run()
                } else {
                    execution.run()
                };
                assert_eq!(facts.completion(), Completion::Complete, "{stated}");
                assert_eq!(facts.returned().and_then(Value::constant), Some(&initial));
                // Each run projects the four layouts it uses into its own
                // cache, and inspects no other type.
                assert_eq!(probes.inspected.get() - inspected, 4, "{stated}");
            }
        }
    }

    /// States one eight-byte object at `0x1000`.
    #[derive(Debug)]
    struct ObjectAt;

    impl ObjectExtents for ObjectAt {
        fn object(&self, address: Address) -> Option<StorageObject> {
            let object = StorageObject {
                start: 0x1000,
                bytes: 8,
                kind: ObjectKind::Global,
                source: ExtentSource::Symbol,
            };
            object.reach(address).is_some().then_some(object)
        }
    }

    impl ExecutionHost<ProbedTarget> for ObjectAt {
        fn object_extents(
            &self,
            _root: &SsaFunction<ProbedTarget>,
            _method: Option<&u32>,
        ) -> Option<Arc<dyn ObjectExtents>> {
            Some(Arc::new(Self))
        }
    }

    /// Cached Boolean classification reaches numeric conversion without raw metadata access.
    #[test]
    fn cached_boolean_operands_keep_positive_conversion_semantics() {
        let probes = Rc::new(Probes::default());
        let ty = |kind| ProbedType {
            kind,
            probes: Rc::clone(&probes),
        };
        let mut builder = SsaFunctionBuilder::<ProbedTarget>::new(1, 0);
        builder
            .in_block(0, |block| {
                let input =
                    block.const_value(SsaDefSpec::tmp(ty(MockType::Bool)), ConstValue::True)?;
                let output =
                    block.emit_def(SsaDefSpec::tmp(ty(MockType::I64)), |dest| SsaOp::IntConv {
                        dest,
                        operand: input,
                        target: ty(MockType::I64),
                        unsigned: false,
                        overflow_check: true,
                    })?;
                block.ret(Some(output))
            })
            .unwrap();
        let ssa = builder.finish().unwrap();
        assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
        probes.forbid_raw.set(true);
        let context =
            AnalysisContext::new(Machine::new(PointerSize::Bit64, None, Default::default()));
        let facts = StaticExecution::new(&ssa, &context).run();
        assert_eq!(facts.completion(), Completion::Complete);
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&ConstValue::I64(1))
        );
        assert_eq!(probes.inspected.get(), 3);
    }
}
