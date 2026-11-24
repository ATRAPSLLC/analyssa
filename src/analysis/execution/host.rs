//! What a host tells static execution about the functions a program calls.
//!
//! One function cannot say what another does, so a call is where execution
//! asks the host. There are three answers, tried in order:
//!
//! 1. **A body.** The host hands over the callee's SSA and says which argument
//!    each of its parameters receives, and the call is executed in place, on
//!    the caller's actual arguments and in the caller's memory. The callee's
//!    stack pointer is the caller's at the call moved by what the call itself
//!    pushes, so arguments the caller left on the stack are where the callee
//!    reads them.
//! 2. **A model.** A function the host knows by what it does rather than by a
//!    body - an import, a runtime helper - as a [`CallModel`].
//!    [`LibraryModel`] covers the C library's memory, string and allocation
//!    functions, and a host maps its imports onto it or writes its own.
//! 3. **An address.** For a call through a pointer the execution knows, the
//!    host names the function at that address, and the call is answered as a
//!    direct one.
//!
//! A virtual call names a method its receiver may override, so the named
//! method is not necessarily the one that runs. Execution answers it only
//! when the host resolves the receiver to an implementation
//! ([`ExecutionHost::dispatch`]), and then as a direct call to that.
//! Constrained calls use [`ExecutionHost::dispatch_constrained`] to select an
//! implementation or adapter by the constraint type while retaining the
//! original argument shape, including an instance receiver's address.
//!
//! A [`crate::ir::CallData::tail`] transfer releases its caller's private stack
//! before any body or model runs. The callee inherits the enclosing return
//! continuation and occupies the same call depth. ABI binding still reads the
//! caller's actual operand values; a pointer into retired caller storage does
//! not keep its contents alive. Root SSA observations remain attached to the
//! original entry activation rather than to whichever callee occupies depth one.
//!
//! A call with no body or model answer honors a complete purity promise on
//! the call instruction. It preserves memory and allocation lifetimes, while
//! an unknown return can retain stack-pointer provenance. Other unanswered
//! calls may write any memory the execution cannot keep private.

use std::{fmt, ops::Range, sync::Arc};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    analysis::{
        Machine,
        budget::Exhaustion,
        consts::{Destination, OperandLayout},
        execution::{
            ArgumentBindings, BranchContinuation, BranchSite, CallValues, CodeMemory, DataWrites,
            EnteredCode, EntrySite, InstructionSources, MachineEffects, MemoryWriteGuard,
            ModelCall, ProtectionEncoding, RealignedExtent, ReturnBinding, ReturnContract,
            StackCongruence,
            access::Access,
            allocation::{
                Allocation, Conditions, Deallocation, Delivery, Duplication, FreeResult, Operand,
                Pages, Reallocation,
            },
            arguments::PhysicalArgument,
            facts::StorageObject,
            inputs::InputIndex,
            memory::{AllocatorFamily, BlockSize, Memory, MemoryRefusal, Through},
            provenance::Pointer,
            register,
            scalar::Scalar,
            thrown::Thrown,
            uses::UseTrace,
            value::{Address, Value},
        },
    },
    ir::{
        AccessMode, CallAccess, CallInputKey, CallLocation, CallMemory, PointeeExtent,
        function::SsaFunction,
        instruction::SsaInstruction,
        integer::{IntegerBits, IntegerType},
        ops::SsaOp,
        value::ConstValue,
        variable::{SsaVarId, VariableOrigin},
    },
    pointer::PointerSize,
    target::{Endianness, Target, TypeLayout},
};

/// One call, as the host is asked about it: the calling function and where
/// in it the call stands.
pub struct CallSite<'a, T: Target> {
    /// The calling function.
    pub caller: &'a SsaFunction<T>,
    /// The method the calling function is, when known: the method a call
    /// entered it as, or the one the execution was started with
    /// ([`StaticExecution::with_method`](crate::analysis::execution::StaticExecution::with_method)).
    pub caller_method: Option<&'a T::MethodRef>,
    /// The block holding the call.
    pub block: usize,
    /// The call's index in the block.
    pub instruction: usize,
}

// By hand: a derive would ask `T` to be `Copy`, and only the reference is
// copied.
impl<T: Target> Clone for CallSite<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Target> Copy for CallSite<'_, T> {}

/// Where a callee's stack pointer starts, relative to its caller's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallStack {
    /// The caller's value holding its stack pointer at the call, or `None`
    /// for the pointer the caller's activation was entered with: a function
    /// that never names its stack pointer has no value of it, and the depth
    /// its host measured places the stack all the same.
    pub stack_pointer: Option<SsaVarId>,
    /// What the call adds to it before the callee runs: minus the size of a
    /// return address a call instruction pushes, zero where the return
    /// address goes to a register.
    pub adjustment: i64,
}

/// A measured call frame available while constructing native input bindings.
struct StackBindings<'a> {
    /// Caller memory, including stores retained by the lifted body.
    memory: &'a mut Memory,
    /// The callee's actual entry stack pointer in that memory.
    entry: Address,
    /// Target layout and byte order.
    access: Access<'a>,
}

/// What the explorer binds one call's parameters from: its arguments, the
/// implicit inputs it captured with their keyed lookup, and the caller's
/// memory with the callee's entry stack pointer.
pub struct CallBinding<'a, T: Target> {
    /// Actual values at this call, borrowed until callee entry.
    pub arguments: &'a [Value<T>],
    /// The separately captured implicit values and their keyed lookup, when
    /// the call has any.
    pub implicit: Option<(&'a [Value<T>], &'a InputIndex)>,
    /// Caller memory and the measured callee entry stack pointer,
    /// and the target layout.
    pub stack: Option<(&'a mut Memory, Option<Address>, Access<'a>)>,
}

/// How the executor lends [`ParameterBindings`] to a host.
///
/// The host-facing binds are inherent; building the bindings from the
/// explorer's call state and attaching its use obligations and physical
/// carriers stay with the executor.
pub trait ExecutorParameters<'a, T: Target> {
    /// Binds a call from what the explorer lends it. Only a measured shared
    /// stack is attached, never an invented callee frame.
    fn of_call(binding: CallBinding<'a, T>) -> Self;

    /// Tracks physical entry accesses in every active call obligation.
    #[must_use]
    fn with_uses(self, uses: Option<&'a UseTrace>) -> Self;

    /// Attaches the carriers belonging to the actual logical argument view.
    #[must_use]
    fn with_physical(self, physical: Option<&'a [PhysicalArgument<T>]>) -> Self;
}

impl<'a, T: Target> ExecutorParameters<'a, T> for ParameterBindings<'a, T> {
    fn of_call(binding: CallBinding<'a, T>) -> Self {
        let (implicit, input_index) = binding
            .implicit
            .map_or((&[][..], None), |(values, index)| (values, Some(index)));
        Self {
            arguments: binding.arguments,
            physical: None,
            uses: None,
            implicit,
            input_index,
            values: Vec::new(),
            exhaustion: None,
            stack: binding.stack.and_then(|(memory, entry, access)| {
                entry.map(|entry| StackBindings {
                    memory,
                    entry,
                    access,
                })
            }),
        }
    }

    fn with_uses(mut self, uses: Option<&'a UseTrace>) -> Self {
        self.uses = uses;
        self
    }

    fn with_physical(mut self, physical: Option<&'a [PhysicalArgument<T>]>) -> Self {
        self.physical = physical;
        self
    }
}

/// The construction of one callee's actual input values.
///
/// An [`ExecutionHost::parameters`] implementation visits the callee's inputs
/// and calls [`Self::bind`] for each justified argument pairing or
/// [`Self::bind_implicit`] for a host-defined hidden input. Missing arguments
/// leave the input unknown.
///
/// The executor owns construction and publication. A memory refusal or
/// configured staging exhaustion stops the invocation and discards every
/// binding; ignoring a returned `false` cannot publish a partial frame.
///
/// # Examples
///
/// A call whose sole operand carries the callee's second declared parameter
/// binds that input explicitly.
///
/// ```rust
/// use analyssa::{
///     analysis::{context::AnalysisContext, execution::{
///         CallSite, ExecutionHost, ParameterBindings, StaticExecution,
///     }},
///     ir::{Callee, ConstValue, function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
///         variable::{DefSite, VariableOrigin}},
///     testing::{MACHINE, MockTarget, MockType},
/// };
///
/// struct Host { callee: SsaFunction<MockTarget> }
/// impl ExecutionHost<MockTarget> for Host {
///     fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
///         (*method == 5).then_some(&self.callee)
///     }
///     fn parameters(&self, _: CallSite<'_, MockTarget>, _: &u32,
///         body: &SsaFunction<MockTarget>, bindings: &mut ParameterBindings<'_, MockTarget>) {
///         for variable in body.variables() {
///             if variable.version() == 0 && variable.origin() == VariableOrigin::Argument(1) {
///                 bindings.bind(variable.id(), 0);
///             }
///         }
///     }
/// }
/// let mut callee = SsaFunctionBuilder::<MockTarget>::new(2, 0);
/// let input = callee.function_mut().create_variable(
///     VariableOrigin::Argument(1), 0, DefSite::entry(), MockType::I64);
/// callee.in_block(0, |block| block.ret(Some(input)))?;
/// let host = Host { callee: callee.finish()? };
/// let mut caller = SsaFunctionBuilder::<MockTarget>::new(0, 0);
/// caller.in_block(0, |block| {
///     let value = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(42))?;
///     let answer = block.call(SsaDefSpec::tmp(MockType::I64), Callee::Direct(5), vec![value])?;
///     block.ret(Some(answer))
/// })?;
/// let caller = caller.finish()?;
/// let context = AnalysisContext::new(MACHINE);
/// let facts = StaticExecution::new(&caller, &context).with_host(&host).run();
/// assert!(facts.is_complete());
/// assert_eq!(facts.returned().and_then(|value| value.constant()), Some(&ConstValue::I64(42)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ParameterBindings<'a, T: Target> {
    /// Actual enclosing and newly entered call obligations, shared across paths.
    uses: Option<&'a UseTrace>,
    /// Physical carriers supplied by one explicit logical argument projection.
    physical: Option<&'a [PhysicalArgument<T>]>,
    /// Actual values at this call, borrowed until callee entry.
    arguments: &'a [Value<T>],
    /// Captured hidden values, disjoint from the ordinary argument prefix.
    implicit: &'a [Value<T>],
    /// Host-defined semantic slots in the implicit slice.
    input_index: Option<&'a InputIndex>,
    /// Input copies in the host's binding order.
    values: Vec<(SsaVarId, Value<T>)>,
    /// Configured staging exhaustion retained until the executor rejects the invocation.
    exhaustion: Option<Exhaustion>,
    /// Measured shared stack, absent when the host supplied no valid mapping.
    stack: Option<StackBindings<'a>>,
}

impl<'a, T: Target> ParameterBindings<'a, T> {
    /// Binds a native entry read from its actual physical carrier before logical narrowing.
    ///
    /// The host proves that this position and parameter share a physical register.
    /// The logical model view may be narrower; it cannot erase established upper
    /// carrier bits or invent them. Missing or unsupported carriers remain unknown.
    pub fn bind_physical_register_view(
        &mut self,
        parameter: SsaVarId,
        argument: usize,
        target: &T::Type,
        machine: Machine,
    ) {
        if let Some(physical) = self.physical.and_then(|values| values.get(argument)) {
            self.values.push((
                parameter,
                register::read(&physical.value, physical.layout, target, machine),
            ));
        }
    }

    /// Returns whether this call has a measured caller-to-callee stack mapping.
    ///
    /// A detached callee stack supplies no knowledge of caller memory. Hosts
    /// may still bind independently justified ordinary or implicit values.
    #[must_use]
    pub const fn has_stack(&self) -> bool {
        self.stack.is_some()
    }

    /// Stages an actual argument's bytes at an ABI-proven callee-stack offset.
    ///
    /// `offset` is relative to the callee's entry stack pointer. `layout` states
    /// the actual stored representation, including width; it is not a nominal
    /// parameter declaration. This reconstructs scalar-promoted argument
    /// stores while retaining other bytes already present in caller memory.
    /// Stage every argument before calling [`Self::bind_stack`].
    ///
    /// The host must justify the location and the value's validity at this
    /// call. An absent argument or shared mapping stages nothing. Returns
    /// `false` on a memory refusal or configured storage exhaustion. Ignoring
    /// failure cannot publish the invocation's facts or resume stack access.
    #[must_use]
    pub fn stage_stack(&mut self, argument: usize, offset: i64, layout: TypeLayout) -> bool {
        if self.exhaustion.is_some() {
            return false;
        }
        let (Some(stack), Some(value)) = (self.stack.as_mut(), self.arguments.get(argument)) else {
            if let Some(uses) = &self.uses {
                uses.unresolved();
            }
            return true;
        };
        let Some(address) = stack.entry.moved(offset, stack.access.ptr_size()) else {
            stack.memory.refuse(MemoryRefusal::AddressRange);
            return false;
        };
        if let Err(exhaustion) =
            stack
                .access
                .store_layout(stack.memory, &Value::Addr(address), value, layout, None)
        {
            self.exhaustion = Some(exhaustion);
            return false;
        }
        if let Some(uses) = &self.uses {
            if let Some(storage) = layout.storage_layout(stack.access.ptr_size()) {
                uses.access(address, storage.bytes());
            } else {
                uses.unresolved();
            }
        }
        stack.memory.refusal().is_none()
    }

    /// Binds the value read from an ABI-proven callee-stack location.
    ///
    /// Reads `ty`'s exact storage layout from the callee's entry pointer plus
    /// `offset`, using the target's byte order. Wider, narrower and overlapping
    /// argument stores therefore supply bytes rather than becoming wrongly
    /// typed whole parameter values. Missing mapping, layout or contents keep
    /// the input unknown. Ordinary and implicit binding order is preserved.
    ///
    /// The host must justify this input's entry location and actual read type.
    /// Returns `false` on a memory refusal or retained staging exhaustion.
    /// The executor rejects the invocation even if the host ignores failure.
    #[must_use]
    pub fn bind_stack(&mut self, parameter: SsaVarId, offset: i64, ty: &T::Type) -> bool {
        if self.exhaustion.is_some() {
            return false;
        }
        let value = match self.stack.as_mut() {
            Some(stack) => {
                let Some(address) = stack.entry.moved(offset, stack.access.ptr_size()) else {
                    stack.memory.refuse(MemoryRefusal::AddressRange);
                    return false;
                };
                match stack.access.load_with_footprint(
                    stack.memory,
                    &Value::Addr(address),
                    ty,
                    None,
                ) {
                    Ok(read) => {
                        if let Some(uses) = &self.uses {
                            if let Some((address, bytes)) = read.footprint {
                                uses.access(address, bytes);
                            } else {
                                uses.unresolved();
                            }
                        }
                        read.value
                    }
                    Err(reason) => {
                        stack.memory.refuse(reason);
                        return false;
                    }
                }
            }
            None => {
                if let Some(uses) = &self.uses {
                    uses.unresolved();
                }
                Value::Unknown
            }
        };
        self.values.push((parameter, value));
        true
    }

    /// Copies actual argument `argument` into callee SSA input `parameter`.
    ///
    /// The host supplies a valid input id in the body it returned and a pairing
    /// justified by the call's ABI or declared signature. Repeated bindings are
    /// applied in emission order, with the last one taking precedence.
    /// An absent argument adds no binding, preserving the callee's unknown
    /// input.
    pub fn bind(&mut self, parameter: SsaVarId, argument: usize) {
        if let Some(value) = self.arguments.get(argument) {
            self.values.push((parameter, value.clone()));
        }
    }

    /// Binds the low integer bits of a host-justified physical register argument.
    ///
    /// The host must establish that `parameter` reads the same physical register
    /// as `argument`, with `source` describing the caller's representation and
    /// `target` the callee's. Equal or narrower scalar integer reads use the
    /// shared typed conversion kernel. Wider reads, absent layouts and noninteger
    /// representations add no binding; this method never invents upper bits.
    /// Partial integer facts and possible private-stack provenance follow the
    /// same conversion rules as an executed integer conversion. An absent
    /// argument adds no binding. Repeated bindings use emission order.
    pub fn bind_integer_register(
        &mut self,
        parameter: SsaVarId,
        argument: usize,
        source: OperandLayout,
        target: &T::Type,
        machine: Machine,
    ) {
        let Some(destination) =
            T::type_layout(target, machine.ptr_size).map(Destination::from_type_layout)
        else {
            return;
        };
        let Some((source_type, target_type)) =
            source.scalar_integer().zip(destination.scalar_integer())
        else {
            return;
        };
        if target_type.bits() > source_type.bits() {
            return;
        }
        let Some(value) = self.arguments.get(argument) else {
            return;
        };
        let conversion = SsaOp::IntConv {
            dest: parameter,
            operand: parameter,
            target: target.clone(),
            unsigned: true,
            overflow_check: false,
        };
        let converted = Scalar::new(machine).evaluate_with_operand_layouts(
            &conversion,
            destination,
            |_| value.clone(),
            |_| source,
        );
        self.values.push((parameter, converted));
    }

    /// Binds a host-justified physical register through its actual typed view.
    ///
    /// The host must prove that `argument` and `parameter` name the same physical
    /// register. `source` describes the caller's carrier and `target` describes
    /// the callee's read. Equal or narrower integers retain their low bits;
    /// vector reshapes and low floating lanes use the shared typed evaluator.
    /// This operation never widens a carrier or numerically promotes a float.
    ///
    /// Missing layouts and unsupported views bind an unknown value while
    /// retaining possible private-stack provenance. An absent argument adds no
    /// binding. Repeated bindings use emission order.
    pub fn bind_register_view(
        &mut self,
        parameter: SsaVarId,
        argument: usize,
        source: Option<TypeLayout>,
        target: &T::Type,
        machine: Machine,
    ) {
        if let Some(value) = self.arguments.get(argument) {
            let view = register::read(value, source, target, machine);
            self.values.push((parameter, view));
        }
    }

    /// Copies semantic input `key` into a justified callee SSA input.
    ///
    /// Keys use the host's contract, independently of ordinary argument indices.
    /// The last occurrence of a repeated key supplies the value. A missing key
    /// leaves the input unknown. The host must establish that this
    /// key denotes the input under the callee's ABI or runtime contract.
    ///
    pub fn bind_implicit(&mut self, parameter: SsaVarId, key: CallInputKey) {
        if let Some(value) = self
            .input_index
            .and_then(|index| index.get(key))
            .and_then(|index| self.implicit.get(index))
        {
            self.values.push((parameter, value.clone()));
        }
    }

    /// Binds a keyed physical register through its actual typed view.
    ///
    /// The host must prove that `key` carries the register read by `parameter`.
    /// The key resolves in the implicit input domain, independently of ordinary
    /// argument positions, with the last occurrence taking precedence. `source`
    /// and `target` describe the caller's carrier and the callee's read.
    /// Conversion and unsupported-view provenance follow [`Self::bind_register_view`].
    /// An absent key adds no binding.
    pub fn bind_implicit_register_view(
        &mut self,
        parameter: SsaVarId,
        key: CallInputKey,
        source: Option<TypeLayout>,
        target: &T::Type,
        machine: Machine,
    ) {
        if let Some(value) = self
            .input_index
            .and_then(|index| index.get(key))
            .and_then(|index| self.implicit.get(index))
        {
            let view = register::read(value, source, target, machine);
            self.values.push((parameter, view));
        }
    }

    /// Returns configured staging exhaustion before any frame is published.
    pub const fn exhaustion(&self) -> Option<Exhaustion> {
        self.exhaustion
    }

    /// The values bound, in the host's binding order.
    pub fn finish(self) -> Vec<(SsaVarId, Value<T>)> {
        self.values
    }
}

/// The objects of storage outside the heap one root's run addresses, as its
/// host states them ([`ExecutionHost::object_extents`]).
///
/// The table is the host's own, handed to the run's memory for the run's
/// lifetime, so it holds what it answers rather than borrowing it.
pub trait ObjectExtents: fmt::Debug + Send + Sync {
    /// The object `address` lies in, where the host states one, with where
    /// its extent comes from.
    fn object(&self, address: Address) -> Option<StorageObject>;
}

/// What a host knows about the functions a program calls.
///
/// Every method has a default that knows nothing, so a host implements only
/// what it can answer; `()` knows nothing at all.
pub trait ExecutionHost<T: Target> {
    /// How `body` returns, where it is native code that returns through an
    /// address its caller leaves it: on the stack, or in a link register.
    ///
    /// The executor gives the address to the body when it enters it and
    /// checks each return goes back through it: a body that overwrote the
    /// stack cell, or returns through another value than the link register
    /// held on entry, cannot resume its caller ([`MemoryRefusal::ReturnControl`]).
    /// `None` for a body that returns semantically, with nothing to check.
    ///
    /// # Errors
    /// Returns a refusal when the body is native but its return cannot be stated.
    fn body_return(
        &self,
        _body: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
    ) -> Result<Option<ReturnContract>, MemoryRefusal> {
        Ok(None)
    }

    /// Returns the actual post-call code address established by this instruction.
    ///
    /// Semantic invocations return `None` and receive a formal calling input;
    /// a physical caller must establish its actual instruction continuation.
    /// No numeric address is invented for a formal input.
    ///
    /// # Errors
    /// Returns a refusal when retained call instructions cannot establish the target.
    fn call_return_address(
        &self,
        _site: CallSite<'_, T>,
    ) -> Result<Option<Value<T>>, MemoryRefusal> {
        Ok(None)
    }

    /// Returns the values `body` is entered with as the root activation.
    ///
    /// Each names one of the body's entry definitions and the value a
    /// contract that holds at every entry gives it: the address a calling
    /// convention enters a function with in a register, or a register the
    /// loader sets once for the whole program. Nothing a particular caller
    /// passes is among them. A callee receives these from its caller's state
    /// instead. The executor refuses a value that names no entry definition,
    /// or one named twice. The default attests none.
    fn root_entry_values(
        &self,
        _body: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
    ) -> Vec<(SsaVarId, Value<T>)> {
        Vec::new()
    }

    /// The registers the selected model leaves its result in, once at its
    /// call: what the call's keyed outputs read. The default states none,
    /// and every keyed output of the call is unknown; the call's own result
    /// keeps its separately checked return binding.
    fn model_machine_effects(
        &self,
        _site: CallSite<'_, T>,
        _method: &T::MethodRef,
        _model: &dyn CallModel<T>,
    ) -> MachineEffects {
        MachineEffects::default()
    }

    /// Returns the body's declared machine interpretation, when the host supplies one.
    ///
    /// The executor requires the effective pointer width, byte order, floating-
    /// point rules and integer division rules to match its analysis context
    /// before interpreting roots or callees. An absent byte order uses the
    /// same image fallback as the context. The default admits semantic IR
    /// supplied under that context's machine contract.
    fn body_machine(
        &self,
        _body: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
    ) -> Option<Machine> {
        None
    }

    /// Declares known physical low bits of this body's entry stack pointer
    /// when it is entered as the root.
    ///
    /// The default supplies no contract. Root execution treats a supplied fact
    /// as explicit entry evidence, translating it to the existing stack root.
    /// A call never reads it: the callee runs on the stack pointer its caller
    /// passes, with the low bits the path established for that root, which a
    /// declaration neither strengthens nor contradicts. A compiler that sees
    /// every caller of a local function may enter it below its convention's
    /// alignment. This grants no memory extent, lifetime, readability or
    /// writability.
    fn entry_stack_congruence(
        &self,
        _body: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
    ) -> Option<StackCongruence> {
        None
    }

    /// Maps a call's entry stack relative to its caller's activation entry.
    ///
    /// The default supplies no mapping. Execution uses this measured relation
    /// only when [`Self::call_stack`] supplies no SSA pointer mapping. This
    /// preserves entry coordinates when an optimizer removes an unused stack
    /// variable; it does not invent an SSA input, physical base or ABI residue.
    /// The offset includes any return-address push or tail-transfer adjustment
    /// exactly once. It is relative to the current activation, including its
    /// inherited entry offset, rather than the root activation's origin.
    /// An unresolved explicitly supplied SSA mapping does not fall back here.
    fn call_stack_from_entry(&self, _site: CallSite<'_, T>) -> Option<i64> {
        None
    }

    /// Returns whether this body's instruction semantics remain valid for the whole run.
    ///
    /// The default admits semantic IR independent of mutable instruction storage.
    /// A host executing decoded code must prove its original instruction bytes
    /// remain valid across all calls and possible writes, including future
    /// instructions of an active activation. A frozen body map is not that proof.
    /// The executor checks roots and entered callees before interpreting them.
    fn code_is_stable(&self, _body: &SsaFunction<T>, _method: Option<&T::MethodRef>) -> bool {
        true
    }

    /// Admits a body's source against the current path's known write history.
    ///
    /// The default uses [`Self::code_is_stable`]. Conditional hosts additionally
    /// check known writes, compatible overlapping snapshots and original effect
    /// coverage before recording source entry. History remains path-local.
    fn admit_code_source(
        &self,
        body: &SsaFunction<T>,
        method: Option<&T::MethodRef>,
        _sources: &mut InstructionSources,
    ) -> bool {
        self.code_is_stable(body, method)
    }

    /// Retires the running body's active source dependency before return or tail transfer.
    ///
    /// The default leaves whole-run source predicates unchanged. Hosts using
    /// active source versions retire exactly one activation here; a suspended
    /// ordinary caller remains active. Tail models retire their caller before
    /// model effects, and normal body returns retire before resuming the caller.
    /// Returns `false` when the source lifecycle cannot be preserved, withholding
    /// the execution's facts.
    fn retire_code_source(
        &self,
        _body: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
        _sources: &mut InstructionSources,
    ) -> bool {
        true
    }

    /// Returns the immutable write policy used throughout this root execution.
    ///
    /// The default adds no constraint. Guards may track writes to instruction
    /// sources independently of image data permissions. The executor forwards
    /// the same guard to body stores, models, transfers and ABI staging.
    fn write_guard(&self) -> Option<&dyn MemoryWriteGuard> {
        None
    }

    /// Returns whether call effects that execution leaves unevaluated are admissible.
    ///
    /// Execution asks before an unanswered call's conservative fallback, and
    /// before a call stated never to return skips its body where a handler
    /// may catch its raise or where its callee's code is no body that admits
    /// ([`Self::admit_code_source`]): termination alone is no proof of what
    /// a handler sees, and a statement read in other code proves nothing of
    /// this code. The default permits ordinary conservative value execution.
    /// An audit requiring evaluated effects returns `false`; execution then
    /// ends with [`crate::analysis::execution::Completion::UnevaluatedEffects`]
    /// and withholds every observation.
    fn admit_unevaluated_call(&self, _site: CallSite<'_, T>) -> bool {
        true
    }

    /// The body of `method`, to execute a call to it in place.
    ///
    /// # Arguments
    ///
    /// * `method` - The method a call names.
    fn body(&self, _method: &T::MethodRef) -> Option<&SsaFunction<T>> {
        None
    }

    /// Returns an owned semantic body selected from the current reaching memory.
    ///
    /// This lookup runs before argument projection, observation, or tail retirement.
    /// A returned body has priority over [`Self::body`] and remains immutable while
    /// entered; later calls may select another allocation for another code version.
    /// The host retains exact source, machine, and method identities and ordinary
    /// source admission still applies. `Ok(None)` falls back to [`Self::body`].
    ///
    /// # Errors
    ///
    /// Returns a refusal from a required code read. Execution withholds the trial
    /// rather than entering a stale fallback body after that refusal.
    fn body_snapshot(
        &self,
        _method: &T::MethodRef,
        _memory: CodeMemory<'_>,
    ) -> Result<Option<Arc<SsaFunction<T>>>, MemoryRefusal> {
        Ok(None)
    }

    /// Resolves an actual unresolved indirect branch into another immutable source body.
    ///
    /// The executor lends the reaching target and memory, then owns validation,
    /// value transfer and source retirement. This is a source discovery boundary,
    /// not a call or a mutable operation callback. Returning `None` leaves the
    /// branch unresolved. Bindings must retain every needed machine component;
    /// missing state remains unknown rather than being reconstructed by a rerun.
    ///
    /// # Errors
    /// Returns a required fetch or source refusal, which withholds the trial.
    fn branch_continuation(
        &self,
        _site: BranchSite<'_, T>,
        _target: &Value<T>,
        _memory: CodeMemory<'_>,
    ) -> Result<Option<BranchContinuation<T>>, MemoryRefusal> {
        Ok(None)
    }

    /// The key of the register the call at `site` receives its result in,
    /// where its convention returns one in a register: the value a callee
    /// leaves there is the call's result, whatever the callee's own return
    /// states. The default states none.
    fn call_result_register(&self, _site: CallSite<'_, T>) -> Option<CallInputKey> {
        None
    }

    /// Whether `next` belongs to the source instruction `written` was lowered
    /// from: an activation whose code `written` changed finishes that source
    /// instruction before it goes on in a lift of the code as it is now. The
    /// default, a host whose instructions are their own source, answers no.
    fn same_source(&self, _written: &SsaInstruction<T>, _next: &SsaInstruction<T>) -> bool {
        false
    }

    /// Whether a write numbered after `since` changed code the activation
    /// running `site`'s body runs after `site` before its block transfers
    /// ([`InstructionSources::changed_within`]): the rest of the instruction
    /// the block is in, and the code the block holds after it. The executor
    /// asks after an instruction whose write changed entered code, and when
    /// a call returns to a caller after one, at the call. Code the activation
    /// reaches only by a transfer is the business of
    /// [`Self::entered_code_change`], asked as control gets there; bytes that
    /// already ran and that nothing enters again need no new lift. The
    /// default, a host with no code that can change, answers no.
    fn code_changed(
        &self,
        _site: BranchSite<'_, T>,
        _sources: &InstructionSources,
        _since: u64,
    ) -> bool {
        false
    }

    /// How the activation running `site`'s body goes on after `site`, the
    /// instruction whose write changed its code or the call during which a
    /// callee did, where [`Self::code_changed`] said it changed: in a lift of
    /// what the code holds now, entered with the machine state the body
    /// states at `site`. Returning `None` leaves it nowhere to go on, and the
    /// trial ends as running code that changed under it.
    ///
    /// # Errors
    /// Returns a required fetch or source refusal, which withholds the trial.
    fn resume_continuation(
        &self,
        _site: BranchSite<'_, T>,
        _memory: CodeMemory<'_>,
    ) -> Result<Option<BranchContinuation<T>>, MemoryRefusal> {
        Ok(None)
    }

    /// Whether control entering `site.to` runs code a write numbered after
    /// `since` changed, and if so the host block
    /// ([`BlockProvenance`](crate::ir::block::BlockProvenance)) the activation
    /// goes on from in a lift of what the code holds now: the first control
    /// runs through on the way. The executor asks at every transfer within
    /// an activation that a write since its lift may have changed. The
    /// default, a host with no code that can change, answers unchanged.
    fn entered_code_change(
        &self,
        _site: EntrySite<'_, T>,
        _sources: &InstructionSources,
        _since: u64,
    ) -> EnteredCode {
        EnteredCode::Unchanged
    }

    /// How the activation running `site`'s body goes on where
    /// [`Self::entered_code_change`] named `host`: in a lift of what the code
    /// holds now, from where a run of `host` begins, entered with the machine
    /// state the body states there
    /// ([`SsaFunction::host_entry_state`](crate::ir::function::SsaFunction::host_entry_state)).
    /// Returning `None` leaves it nowhere to go on, and the trial ends as
    /// running code that changed under it.
    ///
    /// # Errors
    /// Returns a required fetch or source refusal, which withholds the trial.
    fn entry_continuation(
        &self,
        _site: EntrySite<'_, T>,
        _host: u64,
        _memory: CodeMemory<'_>,
    ) -> Result<Option<BranchContinuation<T>>, MemoryRefusal> {
        Ok(None)
    }

    /// Confirms that an executed local instruction fence synchronizes preceding stores.
    ///
    /// The executor asks only about a typed instruction-synchronization
    /// barrier. The host proves its actual machine semantics and local
    /// instruction-fetch domain. Barriers that order subsequent fetches
    /// without making modified bytes visible do not establish this guarantee.
    fn synchronizes_instruction_fetch(
        &self,
        _body: &SsaFunction<T>,
        _instruction: &SsaInstruction<T>,
    ) -> bool {
        false
    }

    /// Selects an admitted body transfer for the owner's bounded control observations.
    ///
    /// Called after the target actually enters, and only when the owner
    /// requests transfer observations. `computed` says whether the target
    /// came from a value the execution computed - an indirect call, a
    /// computed branch - rather than one the instruction names. The default
    /// selects none.
    fn observes_code_transfer(
        &self,
        _from: Option<&T::MethodRef>,
        _to: &T::MethodRef,
        _computed: bool,
    ) -> bool {
        false
    }

    /// Returns numeric ranges proven disjoint from the image, stack and allocations.
    ///
    /// The ranges are scoped to this execution root. They establish address
    /// identity only. Memory contents and store permissions use their own
    /// reaching contracts.
    fn disjoint_numeric_ranges(&self) -> Vec<Range<u64>> {
        Vec::new()
    }

    /// Returns the exact physical bits of an address in a live allocation.
    ///
    /// The executor checks that the address is inside the reaching allocation
    /// before calling this method. The host establishes the physical mapping
    /// for this execution root and allocation identity. The default supplies
    /// no concrete representation for relative allocation addresses.
    fn heap_address_bits(&self, _address: Address) -> Option<u64> {
        None
    }

    /// Returns the objects of storage a run of `root` addresses outside the
    /// heap, each with where its extent comes from: image objects (a sized
    /// symbol, a recovered data object) for symbol addresses, and objects of
    /// `root`'s own frame (its recovered slots) for stack addresses,
    /// measured from its entry stack. A heap allocation's extent is the
    /// executor's own. No object is stated by default.
    ///
    /// An access past an object's extent, judged against the object the
    /// pointer was formed into or else the one the access lands in, is a
    /// finding of its extent's source ([`FindingKind::PastExtent`]), and the
    /// run goes on. An exact extent ([`ExtentSource::exact`]) also bounds what
    /// an observation reports behind a pointer ([`ByteBound::Object`]).
    ///
    /// [`ByteBound::Object`]: crate::analysis::execution::ByteBound::Object
    /// [`ExtentSource::exact`]: crate::analysis::execution::ExtentSource::exact
    /// [`FindingKind::PastExtent`]: crate::analysis::execution::FindingKind::PastExtent
    fn object_extents(
        &self,
        _root: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
    ) -> Option<Arc<dyn ObjectExtents>> {
        None
    }

    /// Returns storage proven writable throughout one activation, relative to its entry stack.
    ///
    /// The half-open range names accessible bytes in this function's frame, not
    /// merely offsets used by its instructions. The promise includes backing
    /// and ordinary write permission until the activation returns or tail-calls.
    /// It must remain valid across its calls. Unknown frame size, guard pages,
    /// conditional allocation or remapping require `None` or a smaller proven
    /// range. No stack range is assumed by default.
    fn stack_extent(
        &self,
        _body: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
    ) -> Option<Range<i64>> {
        None
    }

    /// Returns storage proven writable throughout one activation below the
    /// base a mask realigned its stack pointer to ([`RealignedExtent`]).
    ///
    /// The promise is [`Self::stack_extent`]'s, for the one realignment the
    /// extent names. No range is assumed by default.
    fn realigned_stack_extent(
        &self,
        _body: &SsaFunction<T>,
        _method: Option<&T::MethodRef>,
    ) -> Option<RealignedExtent> {
        None
    }

    /// Binds one logical ABI argument view before observation or transfer.
    ///
    /// The values of ordinary and staged SSA inputs are already captured. Hosts
    /// may select their checked representations or declare actual reaching stack
    /// words through `bindings`. Callee parameters, the selected model and pre-call
    /// observations consume the same list. This callback runs before tail-frame
    /// retirement. The default preserves independently supplied semantic operands.
    /// `model` is the actual selected implementation; no second lookup is needed.
    /// Runs only when a body or model was selected. Unanswered calls retain the
    /// source operand domain used by their stated effect contracts.
    fn arguments(
        &self,
        _site: CallSite<'_, T>,
        _method: Option<&T::MethodRef>,
        _body: Option<&SsaFunction<T>>,
        _model: Option<&dyn CallModel<T>>,
        _bindings: &mut ArgumentBindings<'_, T>,
    ) {
    }

    /// Binds actual call arguments to the callee's SSA inputs.
    ///
    /// The default binds version-zero [`VariableOrigin::Argument`] inputs to
    /// their declared argument indices. A host whose call lists contain only
    /// selected arguments matches their recorded register or stack locations
    /// instead of assuming positional identity.
    ///
    /// Emit each pairing through [`ParameterBindings::bind`] or
    /// [`ParameterBindings::bind_implicit`]. The default does not bind implicit
    /// inputs; their keys and callee mapping belong to the host's contract.
    /// Returning without a pairing leaves that input unknown, which is the
    /// conservative answer when the ABI or call provenance is unavailable.
    /// Such inputs can still alias private storage whose pointer the call
    /// supplied in either domain; declining a pairing does not prove disjointness.
    ///
    /// # Arguments
    ///
    /// * `site` - The original caller and source-bearing call position.
    /// * `method` - The resolved method being entered.
    /// * `body` - The exact body returned by [`Self::body`].
    /// * `bindings` - The actual argument values, and where to bind them.
    fn parameters(
        &self,
        _site: CallSite<'_, T>,
        _method: &T::MethodRef,
        body: &SsaFunction<T>,
        bindings: &mut ParameterBindings<'_, T>,
    ) {
        for variable in body.variables() {
            if variable.version() == 0
                && let VariableOrigin::Argument(index) = variable.origin()
            {
                bindings.bind(variable.id(), usize::from(index));
            }
        }
    }

    /// Returns the unchanged-result contract for the selected body.
    ///
    /// The default follows semantic IR's call/result contract. Native hosts
    /// must attest physical identity and covered bits before admitting a view.
    /// For a tail call this maps to the retired function's own return, including
    /// any result conversion the transfer skips. This answer must be stable for
    /// the same source call and selected implementation. Observation admission
    /// never changes this execution contract.
    fn body_return_binding(
        &self,
        _site: CallSite<'_, T>,
        _method: &T::MethodRef,
        _body: &SsaFunction<T>,
    ) -> ReturnBinding {
        ReturnBinding::Identity
    }

    /// Returns the unchanged-result contract for the selected model.
    ///
    /// `model` is the actual implementation selected after body lookup, without
    /// another host lookup. Native hosts attest its physical return contract;
    /// an unsupported mapping withholds its result while preserving its effects.
    /// The default follows the generic host's semantic call/result contract.
    fn model_return_binding(
        &self,
        _site: CallSite<'_, T>,
        _method: &T::MethodRef,
        _model: &dyn CallModel<T>,
    ) -> ReturnBinding {
        ReturnBinding::Identity
    }

    /// Where the callee of the call at `site` finds its stack pointer, or
    /// `None` when the host cannot say.
    ///
    /// When this returns `None`, execution also asks [`Self::call_stack_from_entry`]
    /// for a measured entry-relative relation. Without either mapping, the
    /// callee runs on a separate stack and incoming stack contents are unknown.
    /// An explicitly supplied but unresolved SSA pointer remains unresolved.
    ///
    /// # Arguments
    ///
    /// * `site` - The call.
    fn call_stack(&self, _site: CallSite<'_, T>) -> Option<CallStack> {
        None
    }

    /// What calling `method` does, for a function the host knows by its
    /// semantics rather than its body.
    ///
    /// # Arguments
    ///
    /// * `method` - The method a call names.
    fn model(&self, _method: &T::MethodRef) -> Option<&dyn CallModel<T>> {
        None
    }

    /// Resolves a direct call's source-qualified method in its actual reaching state.
    ///
    /// The default preserves the declared method. A recovering host can select a
    /// current code version before body lookup, ABI argument projection or tail
    /// retirement. The selected identity is used consistently by body/machine
    /// lookup, bindings, observations and return continuation handling. An
    /// unresolved required target returns `None`; this grants no source admission.
    fn method_for_call(
        &self,
        _site: CallSite<'_, T>,
        method: &T::MethodRef,
        _memory: CodeMemory<'_>,
    ) -> Option<T::MethodRef> {
        Some(method.clone())
    }

    /// Resolves a declared call using its already captured source and machine inputs.
    ///
    /// The default preserves [`Self::method_for_call`]. A native host can select
    /// a pointer captured before a later cell write without reloading the cell,
    /// projecting arguments, retiring stack storage, or executing a prefix again.
    fn method_for_call_with_inputs(
        &self,
        site: CallSite<'_, T>,
        method: &T::MethodRef,
        _inputs: CallValues<'_, T>,
        memory: CodeMemory<'_>,
    ) -> Option<T::MethodRef> {
        self.method_for_call(site, method, memory)
    }

    /// The method at `address` in the host's symbol space, for a call through
    /// a pointer the execution knows.
    ///
    /// # Arguments
    ///
    /// * `address` - Where the call goes.
    fn method_at(&self, _address: u64) -> Option<T::MethodRef> {
        None
    }

    /// Resolves a known indirect target with its actual address-space identity.
    ///
    /// The default delegates image addresses to [`Self::method_at`]. A recovering
    /// host may also resolve live stack or allocated code using `memory`, keeping
    /// object lifetime, representation, entry offset and mode in its method key.
    /// Unavailable or unsupported code returns `None`.
    fn method_at_address(&self, address: Address, _memory: CodeMemory<'_>) -> Option<T::MethodRef> {
        self.method_at(address.image_address()?)
    }

    /// Resolves an indirect call using its actual source site and reaching memory.
    ///
    /// The default preserves [`Self::method_at_address`]. A native recovering
    /// host can retain the caller's physical return requirement while selecting
    /// the code version, before arguments or return bindings are constructed.
    fn method_at_call(
        &self,
        _site: CallSite<'_, T>,
        address: Address,
        memory: CodeMemory<'_>,
    ) -> Option<T::MethodRef> {
        self.method_at_address(address, memory)
    }

    /// The implementation a virtual call to `method` runs on `receiver`, or
    /// `None` when the host cannot say, which leaves the call opaque.
    ///
    /// The named method is the answer only when nothing can override it: a
    /// final method, a sealed class, a receiver whose exact type the host
    /// knows. Entering the named body otherwise executes code the program may
    /// never run there.
    ///
    /// # Arguments
    ///
    /// * `method` - The method the call names.
    /// * `receiver` - The value of the call's receiver, its first argument.
    fn dispatch(&self, _method: &T::MethodRef, _receiver: &Value<T>) -> Option<T::MethodRef> {
        None
    }

    /// Resolves the implementation selected by a call's exact type constraint.
    ///
    /// An instance call supplies its receiver's address as `Some`; a static
    /// constrained call supplies `None`. The returned method or adapter must
    /// consume the original ordinary and implicit inputs with their stated
    /// semantics. If the language requires boxing or another adaptation, the
    /// selected body or model must perform it; an object method expecting a
    /// boxed receiver cannot consume the unmodified value address.
    ///
    /// Returns `None` when the implementation cannot be established, leaving
    /// the call opaque. The default supplies no resolution. This lookup is
    /// separate from [`Self::dispatch`], whose receiver is an object.
    ///
    /// # Arguments
    ///
    /// * `method` - The declared method, not an assumed implementation.
    /// * `constraint_type` - The type selecting the implementation.
    /// * `receiver` - The original instance receiver address, or `None` for a static call.
    fn dispatch_constrained(
        &self,
        _method: &T::MethodRef,
        _constraint_type: &T::TypeRef,
        _receiver: Option<&Value<T>>,
    ) -> Option<T::MethodRef> {
        None
    }
}

/// The host that knows nothing: every call is opaque.
impl<T: Target> ExecutionHost<T> for () {}

/// The semantics of a function a host knows by what it does.
///
/// A model runs no program code. [`ModelCall`] reads and writes memory,
/// allocates and frees, and states the call's result and control, but enters
/// no body. A function that calls back into the program (a comparator, an
/// enumeration callback, a handler it registers) therefore has no model: the
/// host leaves it to its body, which the execution runs as a call like any
/// other, or to an unknown call, whose effects cover whatever the callback
/// does.
///
/// A model's work is charged to the run: each byte its memory operations
/// span costs one step of
/// [`Budget::ExecutionSteps`](crate::analysis::budget::Budget::ExecutionSteps).
pub trait CallModel<T: Target> {
    /// Performs the call.
    ///
    /// A model that cannot say what the call does on these arguments - a copy
    /// of an unknown length - says so through [`ModelCall::clobber`], which is
    /// what a call with no model does.
    ///
    /// # Arguments
    ///
    /// * `call` - Ordinary arguments, keyed implicit inputs and the shared
    ///   memory the model acts on. An omitted or explicitly unknown normal
    ///   result can alias a private pointer from either input domain.
    fn call(&self, call: &mut ModelCall<'_, T>);

    /// Returns whether every possible byte access is reported through [`ModelCall`].
    ///
    /// This promise covers every alternative, including reads whose values are
    /// unused and failure paths. It permits actual access footprints
    /// to bound pre-call observations. The default is conservative: a model may
    /// establish useful values or unchanged memory while omitting reads.
    /// Unknown accesses must use [`ModelCall::clobber`] or another unresolved
    /// operation; an unreported access cannot establish a complete use extent.
    fn tracks_all_memory_accesses(&self) -> bool {
        false
    }

    /// Bounds the bytes used through one ordinary argument before this model runs.
    ///
    /// A returned length covers every use through that argument, including all
    /// alternatives. Zero establishes no pointee use. `None`
    /// supplies no bound; unknown or malformed inputs must not imply zero.
    /// The bound describes this selected model's uses, not an object's size,
    /// and establishes no purity, memory-effect or termination promise.
    ///
    /// This query must be pure. It receives the captured pre-call arguments and
    /// may not read or change memory. Execution asks only the selected model:
    /// an available body takes precedence. Existing object and reporting bounds
    /// still restrict the prefix, and embedded zero bytes remain ordinary data.
    ///
    /// # Arguments
    ///
    /// * `argument` - Ordinary argument whose pointee use is being bounded.
    /// * `arguments` - Actual ordinary argument values captured before effects.
    /// * `ptr_size` - Target pointer width used to interpret native counts.
    fn observation_bound(
        &self,
        _argument: usize,
        _arguments: &[Value<T>],
        _ptr_size: PointerSize,
    ) -> Option<usize> {
        None
    }

    /// Returns the number of alternatives required to cover the call's outcomes.
    ///
    /// The explorer evaluates each alternative from the same incoming state.
    /// This includes allocation failure rather than assuming every allocation
    /// returns a non-null object. Zero alternatives decline the model.
    fn alternatives(&self) -> usize {
        1
    }

    /// Executes one alternative, numbered from zero to [`Self::alternatives`].
    ///
    /// The default model has one outcome, implemented by [`Self::call`].
    fn call_alternative(&self, _alternative: usize, call: &mut ModelCall<'_, T>) {
        self.call(call);
    }

    /// Whether the call's whole effect is what it does to memory.
    ///
    /// A loop whose calls are all such can be replaced by the memory it leaves
    /// behind. One that allocates, frees, or does anything outside memory
    /// cannot, and the default says so. This contract covers every alternative
    /// and is inspected before the model is invoked.
    fn only_memory(&self) -> bool {
        false
    }
}

/// The control outcome of one modelled call alternative.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ModelControl {
    /// The call returns normally.
    #[default]
    Return,
    /// Control never returns, for example because the process exits or loops.
    NoReturn,
    /// The call raises an unhandled exception or trap.
    Trap,
    /// The model cannot establish the control outcome for this alternative.
    Unsupported,
    /// A configured operation extent was exceeded before its memory effects were applied.
    Exhausted(Exhaustion),
}

/// The C library's memory, string and allocation functions, and the platform
/// functions that do the same.
///
/// Each variant names the arguments it reads by position, so one model covers
/// every spelling of a function: `malloc(n)` allocates the size argument `0`
/// holds and `HeapAlloc(heap, flags, n)` the one `2` holds
/// ([`Allocation`]).
///
/// Copy, fill and string-writing models retain a known destination as their
/// normal return value even when unknown inputs require forgetting the bytes.
/// String length scans retain no byte buffer; string copies retain each byte
/// and their terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LibraryModel {
    /// `memcpy(dest, src, n)` and `memmove`: copies `n` bytes, returns `dest`.
    Copy,
    /// `memset(dest, byte, n)`: fills `n` bytes, returns `dest`.
    Fill,
    /// `strlen(s)`: returns the length of a string.
    Length,
    /// `strcpy(dest, src)`: copies a string and its terminator, returns `dest`.
    CopyString,
    /// `strcat(dest, src)`: appends a string, returns `dest`.
    AppendString,
    /// An allocator: `malloc`, `calloc`, `aligned_alloc`, `posix_memalign`,
    /// `HeapAlloc`, `operator new`.
    Allocate(Allocation),
    /// A reallocator: `realloc`, `reallocarray`, `HeapReAlloc`.
    Reallocate(Reallocation),
    /// A string duplicator: `strdup`, `strndup`.
    Duplicate(Duplication),
    /// A deallocator: `free`, `operator delete`, `HeapFree`, `LocalFree`.
    Free(Deallocation),
    /// A function that preserves memory and object lifetimes, such as a clock.
    ///
    /// Its result is unknown. This does not model `free` or other operations
    /// that end lifetimes, and does not authorize deleting the call.
    NoWrites,
    /// A function that writes data but no code: `getcwd`, `fprintf`, `stat`.
    ///
    /// It writes through the pointers it is handed but those it only reads,
    /// and data the program let escape, and may end the lifetime of what it
    /// reaches ([`DataWrites::write`]). It writes no instruction, so every
    /// body a run entered stays the code it was checked to be. Its result is
    /// unknown.
    WritesData(DataWrites),
    /// A function that returns one of its arguments and writes nothing:
    /// `LocalLock(h)` and `GlobalLock(h)` of a fixed block return the block.
    Identity {
        /// Which argument it returns.
        argument: usize,
    },
    /// A function that never returns normally, such as process termination.
    NoReturn,
    /// `__cxa_throw(object, type_info, destructor)`: puts `object` in flight
    /// as an exception of the type `type_info` names, which the runtime
    /// destroys with `destructor` once handled, and raises it
    /// ([`Exceptions`](crate::analysis::execution::Exceptions)).
    Throw {
        /// Which argument holds the exception object.
        object: usize,
        /// Which argument holds the address of its `type_info`.
        type_info: usize,
        /// Which argument holds its destructor, or null.
        destructor: usize,
    },
    /// `__cxa_rethrow()`: raises again the exception the innermost handler
    /// caught; with none, ends the program (`std::terminate`).
    Rethrow,
    /// `_Unwind_Resume(exception)`: resumes the raise of the exception in
    /// flight, as a cleanup's landing pad does once it has run.
    Resume,
    /// `__cxa_begin_catch(exception)`: takes the exception in flight into the
    /// handler and returns its object.
    BeginCatch,
    /// `__cxa_end_catch()`: the innermost handler is done with its exception,
    /// which the runtime destroys and frees, a block of `family`. One whose
    /// destructor is not a known null would run program code, which a model
    /// cannot, so its call is unmodeled.
    EndCatch {
        /// The family of the exception's block.
        family: AllocatorFamily,
    },
    /// A heap's creation: `HeapCreate(options, initial, maximum)` returns a
    /// handle, a block of `family`, that names the heap the blocks allocated
    /// with it belong to; null on failure.
    CreateHeap {
        /// The family of the handle's block.
        family: AllocatorFamily,
    },
    /// A heap's destruction: `HeapDestroy(heap)` ends the handle and every
    /// block allocated in its heap, and returns a `BOOL`.
    DestroyHeap {
        /// Where the handle is read.
        handle: Operand,
        /// The family of the handle's block.
        family: AllocatorFamily,
    },
    /// A host-attested protection routine changing the access of a byte
    /// extent: `mprotect(address, size, protection)` and
    /// `VirtualProtect(address, size, protection, previous)`, and
    /// `NtProtectVirtualMemory(process, &address, &size, protection, previous)`.
    ///
    /// The access installed is the protection argument read in `encoding`; a
    /// protection that is no constant, or that `encoding` decodes to nothing
    /// the model can hold, leaves the call unmodeled. A successful call
    /// writes the access the extent held before, as `encoding` spells it, to
    /// the 32-bit cell `previous` points at. The routine preserves readable
    /// backing and byte contents, including on failure, and changes nothing
    /// when it fails.
    ProtectCode {
        /// Where the extent's base is read; a cell is updated on success.
        address: Operand,
        /// Where the extent's byte count is read; a cell is updated on success.
        size: Operand,
        /// Which argument holds the access to install.
        protection: usize,
        /// How the protection argument spells an access.
        encoding: ProtectionEncoding,
        /// What the routine returns on success and on failure.
        result: ProtectionResult,
        /// Which argument points at the cell receiving the previous access,
        /// where the routine reports one.
        previous: Option<usize>,
        /// What a call must state for either outcome to be modeled.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Conditions::is_none")
        )]
        requires: Conditions,
        /// The exact pages the extent covers, rounded outwards on success.
        ///
        /// Rounding requires known physical address bits and an exact page
        /// size; otherwise the successful alternative is unmodeled.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        pages: Option<Pages>,
    },
}

/// What one argument or the result of a modeled routine holds, as the model
/// reads or makes it: the kind a host with no declared signature passes or
/// receives it as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelValue {
    /// No value: a result only, of a routine that returns nothing.
    Nothing,
    /// A pointer.
    Pointer,
    /// An unsigned count at the pointer width (`size_t`).
    Size,
    /// A signed 32-bit integer: a fill byte's `int`, a protection's or a
    /// flags `DWORD`, a status.
    Int32,
    /// A signed integer at the pointer width: a kernel's result (`long`).
    Offset,
}

/// What a protection routine returns: a signed 32-bit integer from a library
/// routine, a signed one at the pointer width from a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProtectionResult {
    /// Zero on success and minus one on failure, as `mprotect` returns.
    ZeroOrMinusOne,
    /// One on success and zero on failure, a Windows `BOOL`, as
    /// `VirtualProtect` returns.
    OneOrZero,
    /// Zero on success and the given failing signed 32-bit `NTSTATUS`.
    NtStatus {
        /// The status a failed call returns.
        failure: i32,
    },
    /// Zero on success, and on failure a negated `errno` (from -4095 to
    /// -1) at the pointer width, as a Linux system call returns.
    ZeroOrNegatedErrno,
}

impl ProtectionResult {
    /// The value the routine returns, on success or on failure, at
    /// `pointer`'s width where it returns at the pointer width: a failed
    /// system call's negated `errno` is no one number, only one whose bits
    /// from 12 up are all set.
    #[must_use]
    pub fn returned<T: Target>(self, succeeded: bool, pointer: PointerSize) -> Value<T> {
        let int = |value| Value::Const(ConstValue::I32(value));
        match (self, succeeded) {
            (Self::ZeroOrMinusOne | Self::NtStatus { .. }, true) | (Self::OneOrZero, false) => {
                int(0)
            }
            (Self::ZeroOrMinusOne, false) => int(-1),
            (Self::OneOrZero, true) => int(1),
            (Self::NtStatus { failure }, false) => int(failure),
            (Self::ZeroOrNegatedErrno, true) => Value::Const(ConstValue::native(pointer, 0)),
            (Self::ZeroOrNegatedErrno, false) => {
                let width = pointer.bits();
                let all = IntegerType::new(width, false).map_or(0, IntegerType::mask);
                IntegerBits::new(width, 0, all & !0xfff)
                    .map_or(Value::Unknown, |bits| Value::partial_integer(bits, false))
            }
        }
    }

    /// The kind of value the routine returns.
    #[must_use]
    pub const fn kind(self) -> ModelValue {
        match self {
            Self::ZeroOrNegatedErrno => ModelValue::Offset,
            Self::ZeroOrMinusOne | Self::OneOrZero | Self::NtStatus { .. } => ModelValue::Int32,
        }
    }
}

impl<T: Target> CallModel<T> for LibraryModel {
    fn call(&self, call: &mut ModelCall<'_, T>) {
        match *self {
            Self::Copy => call.copy_arguments(),
            Self::Fill => call.fill_arguments(),
            Self::Length => {
                let Some(string) = call.pointer(0) else {
                    call.trace.taint();
                    return;
                };
                if let Some(length) = call
                    .string_length(string, call.access.string_limit())
                    .and_then(|length| i64::try_from(length).ok())
                {
                    call.return_value(Value::Const(ConstValue::native(call.ptr_size(), length)));
                }
            }
            Self::CopyString | Self::AppendString => {
                call.copy_string(*self == Self::AppendString);
            }
            Self::Allocate(allocation) => allocation.allocate(call),
            Self::Reallocate(reallocation) => reallocation.reallocate(call),
            Self::Duplicate(duplication) => duplication.duplicate(call),
            Self::Free(deallocation) => deallocation.free(call),
            Self::NoWrites => {}
            Self::WritesData(writes) => writes.write(call),
            Self::Identity { argument } => match call.argument(argument).cloned() {
                Some(value) => call.return_value(value),
                None => call.set_control(ModelControl::Unsupported),
            },
            Self::NoReturn => call.set_control(ModelControl::NoReturn),
            Self::Throw {
                object,
                type_info,
                destructor,
            } => {
                let thrown = match (call.argument(object), call.argument(destructor)) {
                    (Some(object), Some(destructor)) => {
                        Thrown::new(object.clone(), call.address(type_info), destructor.clone())
                    }
                    _ => {
                        call.set_control(ModelControl::Unsupported);
                        return;
                    }
                };
                call.exceptions_mut().throw(thrown);
                call.set_control(ModelControl::Trap);
            }
            Self::Rethrow => {
                let control = if call.exceptions_mut().rethrow() {
                    ModelControl::Trap
                } else {
                    ModelControl::NoReturn
                };
                call.set_control(control);
            }
            Self::Resume => {
                let control = if call.exceptions().in_flight().is_some() {
                    ModelControl::Trap
                } else {
                    ModelControl::Unsupported
                };
                call.set_control(control);
            }
            Self::BeginCatch => {
                let caught = call
                    .exceptions_mut()
                    .begin_catch()
                    .map(|thrown| thrown.object.clone());
                match caught {
                    Some(object) => call.return_value(object),
                    None => call.set_control(ModelControl::Unsupported),
                }
            }
            Self::EndCatch { family } => match call.exceptions_mut().end_catch() {
                Some(thrown) if thrown.destroys_nothing() => {
                    if let Value::Addr(object) = thrown.object {
                        let _ = call.free(object, Some(family));
                    }
                }
                _ => call.set_control(ModelControl::Unsupported),
            },
            Self::CreateHeap { family } => {
                let handle = call.allocate_block(BlockSize::UNSTATED, false, Some(family));
                call.return_value(Value::Addr(handle));
            }
            Self::DestroyHeap { handle, family } => {
                let named = handle.address(call);
                call.destroy_heap(named);
                if let Some(named) = named {
                    let _ = call.free(named, Some(family));
                }
                call.return_value(Value::Const(ConstValue::I32(1)));
            }
            Self::ProtectCode { .. } => self.protect(call, true),
        }
    }

    fn tracks_all_memory_accesses(&self) -> bool {
        matches!(
            self,
            Self::Copy
                | Self::Fill
                | Self::Length
                | Self::CopyString
                | Self::AppendString
                | Self::Identity { .. }
                | Self::ProtectCode { .. }
        )
    }

    fn observation_bound(
        &self,
        argument: usize,
        arguments: &[Value<T>],
        ptr_size: PointerSize,
    ) -> Option<usize> {
        if !matches!(self, Self::Copy | Self::Fill) || arguments.len() != 3 {
            return None;
        }
        let bytes = usize::try_from(arguments.get(2)?.native_count(ptr_size)?).ok()?;
        match (self, argument) {
            (Self::Copy, 0 | 1) | (Self::Fill, 0) => Some(bytes),
            (_, 0..=2) => Some(0),
            _ => None,
        }
    }

    fn alternatives(&self) -> usize {
        match self {
            Self::Allocate(_)
            | Self::Reallocate(_)
            | Self::Duplicate(_)
            | Self::CreateHeap { .. }
            | Self::ProtectCode { .. } => 2,
            _ => 1,
        }
    }

    fn call_alternative(&self, alternative: usize, call: &mut ModelCall<'_, T>) {
        match (self, alternative) {
            (Self::Allocate(allocation), 1) => allocation.fail(call),
            (Self::Reallocate(reallocation), 1) => reallocation.fail(call),
            (Self::Duplicate(_) | Self::CreateHeap { .. }, 1) => {
                call.return_value(Value::Const(ConstValue::native(call.ptr_size(), 0)));
            }
            (Self::ProtectCode { .. }, 1) => self.protect(call, false),
            _ => self.call(call),
        }
    }

    fn only_memory(&self) -> bool {
        matches!(
            self,
            Self::Copy | Self::Fill | Self::Length | Self::CopyString | Self::AppendString
        )
    }
}

impl LibraryModel {
    /// The memory a call to the routine touches, and how: the bytes its
    /// pointer arguments reach, what it ends, and the routine's own state it
    /// may set (an allocator's `errno`); nothing else. A pointer of
    /// `pointer`'s width is what an allocator hands back through an out
    /// argument.
    ///
    /// What the model says the routine does, read as accesses: a block copy
    /// or a fill replaces exactly the bytes its size argument counts, so a
    /// store to them before it is dead; a string's bytes run to its
    /// terminator, which no extent states, so a string written may leave
    /// bytes past it as they were; ending an object (`free`, the old block
    /// of a `realloc`) changes what its bytes hold without saying to what. A
    /// routine that writes nothing ([`Self::NoWrites`]) may still read
    /// anything the program let escape (`getenv` reads the environment), and
    /// one that never returns has no footprint a caller sees past it.
    #[must_use]
    pub fn memory<T: Target>(&self, pointer: PointerSize) -> CallMemory<T> {
        let pointee = |arg: usize, extent: PointeeExtent, access: AccessMode| CallAccess {
            location: CallLocation::Pointee {
                arg: u32::try_from(arg).unwrap_or(u32::MAX),
                extent,
            },
            access,
        };
        let counted =
            |arg: usize| PointeeExtent::InArgument(u32::try_from(arg).unwrap_or(u32::MAX));
        let escaped = |access: AccessMode| CallAccess {
            location: CallLocation::Escaped,
            access,
        };
        let word = PointeeExtent::Bytes(u64::try_from(pointer.bytes()).unwrap_or(u64::MAX));
        // The pointer-width cell an operand read through points at.
        let cell = |operand: Operand, access: AccessMode| match operand {
            Operand::Argument(_) => None,
            Operand::Through { through } => Some(pointee(through, word, access)),
        };
        let required = |requires: Conditions| {
            requires
                .iter()
                .filter_map(|condition| cell(condition.operand, AccessMode::Read))
                .collect::<Vec<_>>()
        };
        let accesses = match *self {
            Self::Copy => vec![
                pointee(0, counted(2), AccessMode::Write),
                pointee(1, counted(2), AccessMode::Read),
            ],
            Self::Fill => vec![pointee(0, counted(2), AccessMode::Write)],
            Self::Length => vec![pointee(0, PointeeExtent::Unbounded, AccessMode::Read)],
            Self::CopyString => vec![
                pointee(0, PointeeExtent::Unbounded, AccessMode::MayWrite),
                pointee(1, PointeeExtent::Unbounded, AccessMode::Read),
            ],
            Self::AppendString => vec![
                pointee(0, PointeeExtent::Unbounded, AccessMode::Read),
                pointee(0, PointeeExtent::Unbounded, AccessMode::MayWrite),
                pointee(1, PointeeExtent::Unbounded, AccessMode::Read),
            ],
            Self::Allocate(Allocation {
                delivery,
                size,
                requires,
                protection,
                ..
            }) => {
                let mut accesses = vec![escaped(AccessMode::MayWrite)];
                if let Delivery::Out { out, .. } = delivery {
                    accesses.push(pointee(out, word, AccessMode::MayWrite));
                }
                accesses.extend(cell(size, AccessMode::Read));
                accesses.extend(cell(size, AccessMode::MayWrite));
                accesses.extend(
                    protection.and_then(|protection| cell(protection.operand, AccessMode::Read)),
                );
                accesses.extend(required(requires));
                accesses
            }
            Self::Reallocate(Reallocation { pointer: old, .. }) => vec![
                pointee(old, PointeeExtent::Unbounded, AccessMode::Read),
                pointee(old, PointeeExtent::Unbounded, AccessMode::MayWrite),
                escaped(AccessMode::MayWrite),
            ],
            Self::Duplicate(Duplication { source, limit, .. }) => vec![
                pointee(
                    source,
                    limit.map_or(PointeeExtent::Unbounded, counted),
                    AccessMode::Read,
                ),
                escaped(AccessMode::MayWrite),
            ],
            Self::Free(Deallocation {
                pointer: object,
                requires,
                whole,
                released,
                ..
            }) => {
                let mut accesses = match object {
                    // The object behind a cell is one no argument names.
                    Operand::Argument(index) => {
                        vec![pointee(
                            index,
                            PointeeExtent::Unbounded,
                            AccessMode::MayWrite,
                        )]
                    }
                    Operand::Through { .. } => vec![escaped(AccessMode::MayWrite)],
                };
                accesses.extend(cell(object, AccessMode::Read));
                accesses.extend(whole.and_then(|(operand, _)| cell(operand, AccessMode::Read)));
                accesses.extend(released.and_then(|operand| cell(operand, AccessMode::MayWrite)));
                accesses.extend(required(requires));
                accesses
            }
            Self::NoWrites => vec![escaped(AccessMode::Read)],
            // What its pointers reach of what the program let escape is all
            // the program can name; a pointer into the frame escaped with it.
            Self::WritesData(_) => vec![escaped(AccessMode::Read), escaped(AccessMode::MayWrite)],
            Self::Identity { .. } => Vec::new(),
            // The handle is a fresh block, as an allocator's is.
            Self::CreateHeap { .. } => vec![escaped(AccessMode::MayWrite)],
            // Every block of the heap ends, wherever it escaped to.
            Self::DestroyHeap { handle, .. } => {
                let mut accesses = vec![escaped(AccessMode::MayWrite)];
                accesses.extend(cell(handle, AccessMode::Read));
                accesses
            }
            Self::ProtectCode {
                address,
                size,
                previous,
                requires,
                ..
            } => {
                let mut accesses = required(requires);
                for operand in [address, size] {
                    accesses.extend(cell(operand, AccessMode::Read));
                    accesses.extend(cell(operand, AccessMode::MayWrite));
                }
                accesses
                    .extend(previous.map(|index| {
                        pointee(index, PointeeExtent::Bytes(4), AccessMode::MayWrite)
                    }));
                accesses
            }
            Self::NoReturn => return CallMemory::Unknown,
            // The runtime's own state is all a throw, a resumed unwind or a
            // handler's taking of its exception writes: the exception's
            // header before its object, and the thread's chain of them.
            Self::Throw { .. } | Self::Rethrow | Self::Resume | Self::BeginCatch => Vec::new(),
            // The handled exception ends, wherever its object escaped.
            Self::EndCatch { .. } => vec![escaped(AccessMode::MayWrite)],
        };
        CallMemory::Only(accesses)
    }

    /// The arguments the model reads, by position, where it names every
    /// position up to the last one it reads; `None` where it skips one (as
    /// `HeapAlloc`'s heap), whose kind only a signature states, or reads
    /// none it names.
    #[must_use]
    pub fn arguments(&self) -> Option<Vec<ModelValue>> {
        // An operand read through a cell is a pointer argument, whatever the
        // cell holds.
        let operand = |operand: Operand, kind: ModelValue| match operand {
            Operand::Argument(index) => (index, kind),
            Operand::Through { through } => (through, ModelValue::Pointer),
        };
        let named: Vec<(usize, ModelValue)> = match *self {
            Self::Copy => vec![
                (0, ModelValue::Pointer),
                (1, ModelValue::Pointer),
                (2, ModelValue::Size),
            ],
            Self::Fill => vec![
                (0, ModelValue::Pointer),
                (1, ModelValue::Int32),
                (2, ModelValue::Size),
            ],
            Self::Length => vec![(0, ModelValue::Pointer)],
            Self::CopyString | Self::AppendString => {
                vec![(0, ModelValue::Pointer), (1, ModelValue::Pointer)]
            }
            Self::Allocate(allocation) => {
                let mut named = vec![operand(allocation.size, ModelValue::Size)];
                named.extend(allocation.count.map(|index| (index, ModelValue::Size)));
                named.extend(allocation.alignment.map(|index| (index, ModelValue::Size)));
                named.extend(
                    allocation
                        .flags
                        .map(|flags| (flags.argument, ModelValue::Int32)),
                );
                if let Delivery::Out { out, .. } = allocation.delivery {
                    named.push((out, ModelValue::Pointer));
                }
                named.extend(
                    allocation
                        .protection
                        .map(|protection| operand(protection.operand, ModelValue::Int32)),
                );
                named
            }
            Self::Reallocate(reallocation) => {
                let mut named = vec![
                    (reallocation.pointer, ModelValue::Pointer),
                    (reallocation.size, ModelValue::Size),
                ];
                named.extend(reallocation.count.map(|index| (index, ModelValue::Size)));
                named.extend(
                    reallocation
                        .flags
                        .map(|flags| (flags.argument, ModelValue::Int32)),
                );
                named
            }
            Self::Duplicate(duplication) => {
                let mut named = vec![(duplication.source, ModelValue::Pointer)];
                named.extend(duplication.limit.map(|index| (index, ModelValue::Size)));
                named
            }
            Self::Free(deallocation) => {
                let mut named = vec![operand(deallocation.pointer, ModelValue::Pointer)];
                named.extend(deallocation.size.map(|index| (index, ModelValue::Size)));
                named.extend(
                    deallocation
                        .whole
                        .map(|(released, _)| operand(released, ModelValue::Size)),
                );
                named.extend(
                    deallocation
                        .released
                        .map(|released| operand(released, ModelValue::Size)),
                );
                named
            }
            Self::Identity { argument } => vec![(argument, ModelValue::Pointer)],
            Self::DestroyHeap { handle, .. } => vec![operand(handle, ModelValue::Pointer)],
            Self::CreateHeap { .. } => return None,
            Self::NoWrites | Self::WritesData(_) | Self::NoReturn => return None,
            Self::Throw {
                object,
                type_info,
                destructor,
            } => vec![
                (object, ModelValue::Pointer),
                (type_info, ModelValue::Pointer),
                (destructor, ModelValue::Pointer),
            ],
            Self::Resume | Self::BeginCatch => vec![(0, ModelValue::Pointer)],
            Self::Rethrow | Self::EndCatch { .. } => return None,
            Self::ProtectCode {
                address,
                size,
                protection,
                previous,
                ..
            } => {
                let mut named = vec![
                    operand(address, ModelValue::Pointer),
                    operand(size, ModelValue::Size),
                    (protection, ModelValue::Int32),
                ];
                named.extend(previous.map(|index| (index, ModelValue::Pointer)));
                named
            }
        };
        let mut arguments: Vec<Option<ModelValue>> = Vec::new();
        for (index, kind) in named {
            if arguments.len() <= index {
                arguments.resize(index.checked_add(1)?, None);
            }
            *arguments.get_mut(index)? = Some(kind);
        }
        arguments.into_iter().collect()
    }

    /// Whether the model states what becomes of the function the routine
    /// takes at argument `parameter`. A model runs no program code, so a
    /// routine that calls a function it is handed back runs its model only
    /// where the model accounts for that function: a throw's destructor is
    /// not called by the throw but by whatever finishes with the exception,
    /// and finishing ([`Self::EndCatch`]) declines an exception that has one.
    /// A routine that writes data keeps the functions its statement names
    /// without calling them ([`DataWrites::kept`]). No other model accounts
    /// for a callback.
    #[must_use]
    pub const fn accounts_for_callback(&self, parameter: usize) -> bool {
        match self {
            Self::Throw { destructor, .. } => *destructor == parameter,
            Self::WritesData(writes) => writes.kept.contains(parameter),
            Self::Copy
            | Self::Fill
            | Self::CopyString
            | Self::AppendString
            | Self::Identity { .. }
            | Self::Length
            | Self::Allocate(_)
            | Self::Reallocate(_)
            | Self::Duplicate(_)
            | Self::Free(_)
            | Self::NoWrites
            | Self::CreateHeap { .. }
            | Self::DestroyHeap { .. }
            | Self::NoReturn
            | Self::ProtectCode { .. }
            | Self::Rethrow
            | Self::Resume
            | Self::BeginCatch
            | Self::EndCatch { .. } => false,
        }
    }

    /// The argument, by position, the routine returns when it returns
    /// normally: the destination a copy, a fill or a string copy or append
    /// writes, which `memcpy(p, s, n)` hands back as `p` is. `None` where the
    /// result is a value of its own, as an allocation's object or a
    /// length is.
    #[must_use]
    pub const fn returned_argument(&self) -> Option<usize> {
        match self {
            Self::Copy | Self::Fill | Self::CopyString | Self::AppendString => Some(0),
            Self::Identity { argument } => Some(*argument),
            Self::Length
            | Self::Allocate(_)
            | Self::Reallocate(_)
            | Self::Duplicate(_)
            | Self::Free(_)
            | Self::NoWrites
            | Self::WritesData(_)
            | Self::CreateHeap { .. }
            | Self::DestroyHeap { .. }
            | Self::NoReturn
            | Self::ProtectCode { .. }
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::Resume
            | Self::BeginCatch
            | Self::EndCatch { .. } => None,
        }
    }

    /// What the routine returns, where the model states it.
    #[must_use]
    pub const fn returns(&self) -> Option<ModelValue> {
        match self {
            Self::Copy
            | Self::Fill
            | Self::CopyString
            | Self::AppendString
            | Self::Reallocate(_)
            | Self::Duplicate(_)
            | Self::Identity { .. }
            | Self::CreateHeap { .. } => Some(ModelValue::Pointer),
            Self::DestroyHeap { .. } => Some(ModelValue::Int32),
            Self::Allocate(Allocation {
                delivery: Delivery::Out { .. },
                ..
            }) => Some(ModelValue::Int32),
            Self::ProtectCode { result, .. } => Some(result.kind()),
            Self::Allocate(_) => Some(ModelValue::Pointer),
            Self::Length => Some(ModelValue::Size),
            Self::Free(Deallocation { result, .. }) => Some(match result {
                FreeResult::Nothing => ModelValue::Nothing,
                FreeResult::One | FreeResult::Zero => ModelValue::Int32,
                FreeResult::Null => ModelValue::Pointer,
            }),
            Self::BeginCatch => Some(ModelValue::Pointer),
            Self::EndCatch { .. } => Some(ModelValue::Nothing),
            Self::NoWrites
            | Self::WritesData(_)
            | Self::NoReturn
            | Self::Throw { .. }
            | Self::Rethrow
            | Self::Resume => None,
        }
    }

    /// Executes a protection outcome, updating the extent's cells and
    /// reporting its previous access only on success.
    fn protect<T: Target>(self, call: &mut ModelCall<'_, T>, succeeded: bool) {
        let Self::ProtectCode {
            address,
            size,
            protection,
            encoding,
            result,
            previous,
            requires,
            pages,
        } = self
        else {
            return;
        };
        if !requires.hold(call) {
            call.set_control(ModelControl::Unsupported);
            return;
        }
        if !succeeded {
            call.return_value(result.returned(false, call.ptr_size()));
            return;
        }
        let range = call.protection_range(address, size, pages);
        let installed = call
            .count(protection)
            .and_then(|value| u64::try_from(value).ok())
            .and_then(|value| encoding.decode(value));
        let report = match previous.map(|index| call.pointer(index)) {
            Some(None) => {
                call.set_control(ModelControl::Unsupported);
                return;
            }
            Some(Some(cell)) => Some(cell),
            None => None,
        };
        let (Some((base, len)), Some(installed)) = (range, installed) else {
            call.set_control(ModelControl::Unsupported);
            return;
        };
        let Some(held) = call.protect_code(base, len, installed) else {
            return;
        };
        if let Some(cell) = address.cell(call) {
            call.write_address(cell, base);
        }
        if let Some(cell) = size.cell(call) {
            call.write_count(cell, u64::try_from(len).unwrap_or(u64::MAX));
        }
        if let Some(cell) = report {
            match (held, call.access.byte_order()) {
                (Some(held), Some(order)) => {
                    let value = u32::try_from(encoding.encode(held)).unwrap_or(u32::MAX);
                    let bytes = match order {
                        Endianness::Little => value.to_le_bytes(),
                        Endianness::Big => value.to_be_bytes(),
                    };
                    call.write(cell, &bytes);
                }
                _ => call.write_unknown(cell, 4),
            }
        }
        call.return_value(result.returned(true, call.ptr_size()));
    }
}

impl<T: Target> ModelCall<'_, T> {
    /// Reads the protection extent, rounding to exact pages when the
    /// physical address is known and refusing empty or wrapping ranges.
    fn protection_range(
        &mut self,
        address: Operand,
        size: Operand,
        pages: Option<Pages>,
    ) -> Option<(Address, usize)> {
        let base = address.address(self)?;
        let len = size.count(self).or_else(|| {
            let Operand::Argument(index) = size else {
                return None;
            };
            let value = self.argument(index)?.address()?;
            self.access.physical_address_bits(value).map(u128::from)
        })?;
        let len = usize::try_from(len).ok()?;
        let Some(pages) = pages else {
            return Some((base, len));
        };
        if !pages.exact || !pages.size.is_power_of_two() || len == 0 {
            return None;
        }
        let physical = self.access.physical_address_bits(base)?;
        let mask = pages.size.checked_sub(1)?;
        let below = physical & mask;
        let base = base.moved(i64::try_from(below).ok()?.checked_neg()?, self.ptr_size())?;
        let len = u64::try_from(len)
            .ok()?
            .checked_add(below)?
            .checked_add(mask)?
            & !mask;
        let len = usize::try_from(len).ok()?;
        let physical_end = u128::from(physical & !mask).checked_add(len as u128)?;
        let address_limit = u128::from(self.ptr_size().mask_unsigned(u64::MAX)).checked_add(1)?;
        (physical_end <= address_limit && base.linear_prefix_len(len) == len).then_some((base, len))
    }
    /// Reads a native byte count, retaining explicit exhaustion for guarded writes.
    fn checked_size(&mut self, index: usize) -> Option<usize> {
        let raw = self.count(index)?;
        let Ok(len) = usize::try_from(raw) else {
            if self.access.has_write_guard() {
                self.state.memory_mut().refuse(MemoryRefusal::AddressRange);
            }
            return None;
        };
        if self.access.has_write_guard() {
            self.check_write_extent(len).then_some(len)
        } else {
            (len <= self.access.buffer_limit()).then_some(len)
        }
    }
    /// Records the exact destination returned on every modeled normal return.
    ///
    /// Unknown source bytes or sizes affect memory knowledge independently of
    /// this pointer. A missing destination establishes no exact return value.
    fn returned_destination(&mut self) -> Option<Pointer> {
        let destination = self.pointer(0)?;
        self.return_value(Value::Addr(destination.address()));
        Some(destination)
    }
    /// Copies source bytes as observed before the self, including unknown bytes.
    ///
    /// Snapshot semantics implement `memmove` and every defined `memcpy` self.
    /// Overlapping `memcpy` arguments have no defined C result; this model does
    /// not use that case to infer a forward-copy result.
    fn copy_arguments(&mut self) {
        let destination = self.returned_destination();
        let Some(len) = self.checked_size(2) else {
            self.clobber();
            return;
        };
        if len == 0 {
            return;
        }
        let Some(dest) = destination else {
            self.clobber();
            return;
        };
        self.copy(dest, 1, len);
    }
    /// Fills the destination, retaining its normal return when size or bytes are unknown.
    fn fill_arguments(&mut self) {
        let destination = self.returned_destination();
        let Some(len) = self.checked_size(2) else {
            self.clobber();
            return;
        };
        if len == 0 {
            return;
        }
        let Some(dest) = destination else {
            self.clobber();
            return;
        };
        match self.integer(1) {
            Some(byte) => self.fill(dest, byte.to_le_bytes()[0], len),
            None => self.write_unknown(dest, len),
        }
    }
    /// Checks the first byte a normal string-copy return must write, then
    /// forgets the unknown tail: the rest of the object the destination
    /// points into ([`ModelCall::forget_from`]).
    fn unknown_string(&mut self, destination: Pointer) {
        self.through(Through {
            reads: None,
            writes: destination.object(),
        });
        if self
            .access
            .preflight_block_write(
                self.state.memory_mut(),
                &Value::<T>::Addr(destination.address()),
                1,
                1,
                Some(false),
                false,
            )
            .is_err()
        {
            return;
        }
        self.forget_from(destination, false);
    }
    /// Copies or appends a bounded string and its terminator.
    ///
    /// Append offsets use a scan without retained storage. An unknown
    /// source, or a destination whose end nobody knows, forgets the rest of
    /// the destination's object ([`ModelCall::forget_from`]).
    fn copy_string(&mut self, append: bool) {
        let Some(dest) = self.returned_destination() else {
            self.clobber();
            return;
        };
        let at = if append {
            // Nobody knows where the string ends: the append writes
            // somewhere from the destination to the end of its object.
            let Some(existing) = self.string_length(dest, self.access.string_limit()) else {
                self.forget_from(dest, false);
                return;
            };
            let Some(at) = i64::try_from(existing)
                .ok()
                .and_then(|length| dest.moved(length, self.ptr_size()))
            else {
                self.state.memory_mut().refuse(MemoryRefusal::AddressRange);
                self.trace.taint();
                return;
            };
            at
        } else {
            dest
        };
        let Some(src) = self.pointer(1) else {
            self.unknown_string(at);
            return;
        };
        match self.read_string(src, self.access.string_limit()) {
            Some(mut string) => {
                string.push(0);
                self.write(at, &string);
            }
            None => self.unknown_string(at),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, error::Error, process, sync::Mutex};

    use super::*;
    use crate::{
        analysis::{
            budget::{Budget, WorkBudget},
            context::AnalysisContext,
            execution::{
                AllocatedProtection, AllocationFlags, AllocatorFamily, ArgumentSet, Completion,
                Condition, ExtentSource, FindingKind, Findings, InstructionPermissions, ObjectKind,
                Pages, StaticExecution, Test,
                access::Access,
                frame::Frame,
                memory::{Loaded, WriteSite},
                state::State,
                trace::Trace,
                value::{Address, Root},
            },
            image::{ImageData, ImageRegion, Mutability},
        },
        ir::{
            Callee, SsaDefSpec, SsaFunctionBuilder,
            float::{FloatBits, FloatFormat},
            integer::IntegerBits,
            lanes::VectorConst,
            variable::{DefSite, VariableRole},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// Native integer register reads preserve low bits and the callee's signed type.
    #[test]
    fn parameter_binding_integer_register_reads_exact_low_bits() {
        let arguments = [Value::<MockTarget>::Const(ConstValue::U64(
            0x1234_5678_ffff_ffff,
        ))];
        let source = OperandLayout::from_type_layout(TypeLayout::of::<MockTarget>(
            &MockType::U64,
            MACHINE.ptr_size,
        ));
        for (target, expected) in [
            (MockType::U8, ConstValue::U8(255)),
            (MockType::I8, ConstValue::I8(-1)),
            (MockType::U32, ConstValue::U32(u32::MAX)),
            (MockType::I32, ConstValue::I32(-1)),
            (MockType::U64, ConstValue::U64(0x1234_5678_ffff_ffff)),
        ] {
            let mut bindings = ParameterBindings::of_call(CallBinding {
                arguments: &arguments,
                implicit: None,
                stack: None,
            });
            let parameter = SsaVarId::from_index(0);
            bindings.bind_integer_register(parameter, 0, source, &target, MACHINE);
            assert_eq!(
                bindings.finish(),
                vec![(parameter, Value::Const(expected))],
                "{target:?}"
            );
        }
    }

    /// Wider, absent and noninteger inputs establish no invented register bits.
    #[test]
    fn parameter_binding_integer_register_requires_complete_layouts() {
        let arguments = [Value::<MockTarget>::Const(ConstValue::U8(7))];
        for (source, target, argument) in [
            (MockType::U8, MockType::U64, 0),
            (MockType::U8, MockType::F64, 0),
            (MockType::F64, MockType::U8, 0),
            (MockType::Unknown, MockType::U8, 0),
            (MockType::U8, MockType::U8, 1),
        ] {
            let mut bindings = ParameterBindings::of_call(CallBinding {
                arguments: &arguments,
                implicit: None,
                stack: None,
            });
            let source = OperandLayout::from_type_layout(TypeLayout::of::<MockTarget>(
                &source,
                MACHINE.ptr_size,
            ));
            bindings.bind_integer_register(
                SsaVarId::from_index(0),
                argument,
                source,
                &target,
                MACHINE,
            );
            assert!(bindings.finish().is_empty());
        }
    }

    /// Narrow register reads retain unknown bits and possible private-stack aliases.
    #[test]
    fn parameter_binding_integer_register_keeps_partial_stack_provenance() {
        let partial = IntegerBits::new(64, 0xff00, 2).unwrap();
        let arguments = [
            Value::<MockTarget>::partial_integer(partial, true),
            Value::Addr(Address::new(Root::Stack, -16, MACHINE.ptr_size)),
        ];
        let source = OperandLayout::from_type_layout(TypeLayout::of::<MockTarget>(
            &MockType::U64,
            MACHINE.ptr_size,
        ));
        let mut bindings = ParameterBindings::of_call(CallBinding {
            arguments: &arguments,
            implicit: None,
            stack: None,
        });
        bindings.bind_integer_register(SsaVarId::from_index(0), 0, source, &MockType::U8, MACHINE);
        bindings.bind_integer_register(SsaVarId::from_index(1), 1, source, &MockType::U32, MACHINE);
        let values = bindings.finish();
        assert_eq!(values.len(), 2);
        let partial = &values.first().unwrap().1;
        assert_eq!(partial.integer_bits(), IntegerBits::new(8, 0, 2));
        assert!(partial.may_reference_stack());
        let address = &values.get(1).unwrap().1;
        assert!(address.may_reference_stack());
        assert!(address.constant().is_none());
    }

    /// Supported register views preserve values in the destination representation.
    #[test]
    fn parameter_binding_register_view_preserves_supported_representations() {
        let vector = VectorConst::from_lanes(32, [1, 2, 3, 4]).unwrap();
        for (source, target, value, expected) in [
            (
                MockType::U64,
                MockType::I8,
                Value::<MockTarget>::Const(ConstValue::U64(255)),
                Value::Const(ConstValue::I8(-1)),
            ),
            (
                MockType::F64,
                MockType::F64,
                Value::Const(ConstValue::Float(
                    FloatBits::new(FloatFormat::Double, 0x3ff0_0000_0000_0000).unwrap(),
                )),
                Value::Const(ConstValue::Float(
                    FloatBits::new(FloatFormat::Double, 0x3ff0_0000_0000_0000).unwrap(),
                )),
            ),
            (
                MockType::V4I32,
                MockType::V2F64,
                Value::Const(ConstValue::Vector(vector.clone())),
                Value::Const(ConstValue::Vector(vector.resized(64, 2).unwrap())),
            ),
            (
                MockType::Ptr,
                MockType::U64,
                Value::Addr(Address::new(Root::Stack, -16, MACHINE.ptr_size)),
                Value::Addr(Address::new(Root::Stack, -16, MACHINE.ptr_size)),
            ),
            (
                MockType::U64,
                MockType::Ptr,
                Value::Addr(Address::new(Root::Stack, -16, MACHINE.ptr_size)),
                Value::Addr(Address::new(Root::Stack, -16, MACHINE.ptr_size)),
            ),
        ] {
            let arguments = [value];
            let mut bindings = ParameterBindings::of_call(CallBinding {
                arguments: &arguments,
                implicit: None,
                stack: None,
            });
            let parameter = SsaVarId::from_index(0);
            bindings.bind_register_view(
                parameter,
                0,
                MockTarget::type_layout(&source, MACHINE.ptr_size),
                &target,
                MACHINE,
            );
            assert_eq!(
                bindings.finish(),
                vec![(parameter, expected)],
                "{source:?} -> {target:?}"
            );
        }
    }

    /// Unsupported views discard exact bits without discarding a possible stack escape.
    #[test]
    fn parameter_binding_register_view_refuses_unsupported_bits() {
        for (source, target) in [
            (None, MockType::U64),
            (Some(MockType::Unknown), MockType::U64),
            (Some(MockType::U8), MockType::U64),
            (Some(MockType::F64), MockType::F80),
            (Some(MockType::F64), MockType::BF16),
            (Some(MockType::F64), MockType::V2F64),
            (Some(MockType::V4I32), MockType::Unknown),
        ] {
            for value in [
                Value::<MockTarget>::Unknown,
                Value::Const(ConstValue::U64(7)),
                Value::StackDerived,
                Value::Addr(Address::new(Root::Stack, -16, MACHINE.ptr_size)),
            ] {
                let expected = value.join(&Value::Unknown);
                let arguments = [value];
                let mut bindings = ParameterBindings::of_call(CallBinding {
                    arguments: &arguments,
                    implicit: None,
                    stack: None,
                });
                let parameter = SsaVarId::from_index(0);
                bindings.bind_register_view(
                    parameter,
                    0,
                    source.and_then(|ty| MockTarget::type_layout(&ty, MACHINE.ptr_size)),
                    &target,
                    MACHINE,
                );
                assert_eq!(
                    bindings.finish(),
                    vec![(parameter, expected)],
                    "{source:?} -> {target:?}"
                );
            }
        }
        let arguments = [];
        let mut bindings = ParameterBindings::<MockTarget>::of_call(CallBinding {
            arguments: &arguments,
            implicit: None,
            stack: None,
        });
        bindings.bind_register_view(SsaVarId::from_index(0), 0, None, &MockType::U64, MACHINE);
        assert!(bindings.finish().is_empty());
    }

    /// Typed keyed inputs stay disjoint from ordinary positions and retain last-key precedence.
    #[test]
    fn parameter_binding_implicit_register_view_uses_semantic_keys() {
        let key = CallInputKey::new(0);
        let missing = CallInputKey::new(1);
        let ordinary = [Value::<MockTarget>::Const(ConstValue::U64(99))];
        let implicit = [
            Value::<MockTarget>::Const(ConstValue::U64(0x1234_5678_0000_0007)),
            Value::Const(ConstValue::U64(0x1234_5678_0000_0008)),
        ];
        for indexed in [false, true] {
            let index = if indexed {
                InputIndex::Indexed(BTreeMap::from([(key, 1)]))
            } else {
                InputIndex::Inline([(key, 0), (key, 1)].into_iter().collect())
            };
            let mut bindings = ParameterBindings::of_call(CallBinding {
                arguments: &ordinary,
                implicit: Some((&implicit, &index)),
                stack: None,
            });
            let parameter = SsaVarId::from_index(0);
            let source = MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size);
            bindings.bind_implicit_register_view(parameter, key, source, &MockType::U32, MACHINE);
            bindings.bind_implicit_register_view(
                SsaVarId::from_index(1),
                missing,
                source,
                &MockType::U32,
                MACHINE,
            );
            assert_eq!(
                bindings.finish(),
                vec![(parameter, Value::Const(ConstValue::U32(8)))]
            );
        }
    }

    /// Unsupported keyed views retain stack provenance without copying incompatible exact values.
    #[test]
    fn parameter_binding_implicit_register_view_preserves_stack_provenance() {
        let key = CallInputKey::new(0);
        let index = InputIndex::Inline([(key, 0)].into_iter().collect());
        for source in [
            None,
            MockTarget::type_layout(&MockType::Ptr, MACHINE.ptr_size),
        ] {
            let implicit = [Value::<MockTarget>::Addr(Address::new(
                Root::Stack,
                -16,
                MACHINE.ptr_size,
            ))];
            let mut bindings = ParameterBindings::of_call(CallBinding {
                arguments: &[],
                implicit: Some((&implicit, &index)),
                stack: None,
            });
            let parameter = SsaVarId::from_index(0);
            bindings.bind_implicit_register_view(parameter, key, source, &MockType::F64, MACHINE);
            assert_eq!(bindings.finish(), vec![(parameter, Value::StackDerived)]);
        }
    }

    /// Mixed-width stores are read as bytes in each target byte order.
    #[test]
    fn parameter_binding_stack_layouts_preserve_overlapping_bytes() {
        for (order, expected) in [
            (Endianness::Little, 0x4433_aa11),
            (Endianness::Big, 0x44aa_2211),
        ] {
            let mut machine = MACHINE;
            machine.byte_order = Some(order);
            let context = AnalysisContext::new(machine);
            let arguments = [
                Value::<MockTarget>::Const(ConstValue::U32(0x4433_2211)),
                Value::Const(ConstValue::U8(0xaa)),
            ];
            let mut memory = Memory::default();
            let entry = Address::new(Root::Stack, -32, PointerSize::Bit64);
            let mut bindings = ParameterBindings::of_call(CallBinding {
                arguments: &arguments,
                implicit: None,
                stack: Some((&mut memory, Some(entry), Access::new(&context))),
            });
            assert!(bindings.has_stack());
            assert!(bindings.stage_stack(
                0,
                0,
                TypeLayout::of::<MockTarget>(&MockType::I32, PointerSize::Bit64)
            ));
            assert!(bindings.stage_stack(
                1,
                1,
                TypeLayout::of::<MockTarget>(&MockType::I8, PointerSize::Bit64)
            ));
            assert!(bindings.bind_stack(SsaVarId::from_index(0), 0, &MockType::U32));
            assert!(bindings.bind_stack(SsaVarId::from_index(1), 1, &MockType::U8));
            assert_eq!(
                bindings.finish(),
                vec![
                    (
                        SsaVarId::from_index(0),
                        Value::Const(ConstValue::U32(expected))
                    ),
                    (SsaVarId::from_index(1), Value::Const(ConstValue::U8(0xaa))),
                ]
            );
        }
    }

    /// A detached frame cannot read caller memory or invent a stack input.
    #[test]
    fn parameter_binding_stack_requires_a_measured_mapping() {
        let arguments = [Value::<MockTarget>::Const(ConstValue::I64(42))];
        let mut bindings = ParameterBindings::of_call(CallBinding {
            arguments: &arguments,
            implicit: None,
            stack: None,
        });
        assert!(!bindings.has_stack());
        assert!(bindings.stage_stack(
            0,
            0,
            TypeLayout::of::<MockTarget>(&MockType::I64, PointerSize::Bit64)
        ));
        assert!(bindings.bind_stack(SsaVarId::from_index(0), 0, &MockType::I64));
        assert_eq!(
            bindings.finish(),
            vec![(SsaVarId::from_index(0), Value::Unknown)]
        );
    }

    /// A host cannot publish an unrelated constant after ignoring a stack-span refusal.
    #[test]
    fn parameter_binding_stack_memory_refusal_withholds_the_invocation()
    -> Result<(), Box<dyn Error>> {
        /// Deliberately ignores a memory refusal during ABI placement.
        struct IgnoresRefusal(SsaFunction<MockTarget>);
        impl ExecutionHost<MockTarget> for IgnoresRefusal {
            fn body(&self, _: &u32) -> Option<&SsaFunction<MockTarget>> {
                Some(&self.0)
            }
            fn call_stack(&self, site: CallSite<'_, MockTarget>) -> Option<CallStack> {
                Some(CallStack {
                    stack_pointer: Some(site.caller.entry_stack_pointer()?),
                    adjustment: 0,
                })
            }
            fn parameters(
                &self,
                _: CallSite<'_, MockTarget>,
                _: &u32,
                _: &SsaFunction<MockTarget>,
                bindings: &mut ParameterBindings<'_, MockTarget>,
            ) {
                let _ = bindings.stage_stack(
                    0,
                    i64::MAX,
                    TypeLayout::of::<MockTarget>(&MockType::I64, PointerSize::Bit64),
                );
            }
        }
        let mut callee = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        callee.in_block(0, |block| {
            let answer = block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(7))?;
            block.ret(Some(answer))
        })?;
        let host = IgnoresRefusal(callee.finish()?);
        let mut caller = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let stack = caller.function_mut().create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        caller
            .function_mut()
            .variable_mut(stack)
            .ok_or("missing stack")?
            .set_role(VariableRole::EntryStackPointer);
        caller.in_block(0, |block| {
            let argument =
                block.const_value(SsaDefSpec::tmp(MockType::I64), ConstValue::I64(42))?;
            let answer = block.call(
                SsaDefSpec::tmp(MockType::I64),
                Callee::Direct(1),
                vec![argument],
            )?;
            block.ret(Some(answer))
        })?;
        let caller = caller.finish()?;
        let mut machine = MACHINE;
        machine.ptr_size = PointerSize::Bit128;
        let context = AnalysisContext::new(machine);
        let facts = StaticExecution::new(&caller, &context)
            .with_host(&host)
            .run();
        assert!(
            matches!(
                facts.completion(),
                Completion::Memory {
                    reason: MemoryRefusal::AddressRange,
                    ..
                }
            ),
            "{:?}",
            facts.completion()
        );
        assert!(facts.returned().is_none());
        assert!(facts.constants().is_empty());
        Ok(())
    }

    /// Builds unused local entries independently of actual parameter bindings.
    fn binding_body(locals: usize) -> SsaFunction<MockTarget> {
        let mut body = SsaFunction::new(0, 0);
        for _ in 0..locals {
            body.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I64);
        }
        body
    }

    /// The default maps only initial declared arguments and preserves exact values.
    #[test]
    fn parameter_binding_maps_declared_inputs_and_skips_missing_arguments() {
        let mut body = binding_body(1);
        let second = body.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::I64,
        );
        body.create_variable(
            VariableOrigin::Argument(0),
            1,
            DefSite::entry(),
            MockType::I64,
        );
        let first = body.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I64,
        );
        body.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I64,
        );
        let args = [Value::Const(ConstValue::I64(3)), stack(-32)];
        for (arguments, expected) in [
            (
                &args[..],
                vec![
                    (second, stack(-32)),
                    (first, Value::Const(ConstValue::I64(3))),
                ],
            ),
            (&args[..1], vec![(first, Value::Const(ConstValue::I64(3)))]),
        ] {
            let mut bindings = ParameterBindings::of_call(CallBinding {
                arguments,
                implicit: None,
                stack: None,
            });
            ().parameters(
                CallSite {
                    caller: &body,
                    caller_method: None,
                    block: 0,
                    instruction: 0,
                },
                &1,
                &body,
                &mut bindings,
            );
            assert_eq!(bindings.finish(), expected);
        }
    }

    /// A missing semantic key never borrows an ordinary value; a present one
    /// binds its implicit value.
    #[test]
    fn implicit_parameter_binding_binds_only_present_keys() {
        let ordinary = [Value::<MockTarget>::Const(ConstValue::I64(1))];
        let hidden = [Value::<MockTarget>::Const(ConstValue::I64(41))];
        let key = CallInputKey::new(0);
        let index = InputIndex::Inline([(key, 0)].into_iter().collect());
        let mut bindings = ParameterBindings::of_call(CallBinding {
            arguments: &ordinary,
            implicit: Some((&hidden, &index)),
            stack: None,
        });
        bindings.bind_implicit(SsaVarId::from_index(0), CallInputKey::new(17));
        bindings.bind_implicit(SsaVarId::from_index(1), key);
        assert_eq!(
            bindings.finish(),
            vec![(SsaVarId::from_index(1), Value::Const(ConstValue::I64(41)))]
        );
    }

    fn stack(offset: i64) -> Value<MockTarget> {
        Value::Addr(Address::new(Root::Stack, offset, PointerSize::Bit64))
    }

    #[derive(Debug)]
    struct ProtectionGuard;

    impl MemoryWriteGuard for ProtectionGuard {
        fn check_write(
            &self,
            _: &mut InstructionSources,
            _: Address,
            _: usize,
        ) -> Result<(), MemoryRefusal> {
            Ok(())
        }

        fn change_instruction_permissions(
            &self,
            sources: &mut InstructionSources,
            address: Address,
            len: usize,
            stack_owner: Option<u32>,
            permissions: InstructionPermissions,
        ) -> Result<Option<InstructionPermissions>, MemoryRefusal> {
            if address.image_address() != Some(0x2000) || len != 8 || stack_owner.is_some() {
                return Err(MemoryRefusal::InstructionSource);
            }
            let held = sources
                .permissions(address)
                .unwrap_or(InstructionPermissions {
                    writable: false,
                    executable: true,
                });
            sources.set_permissions(address, u64::try_from(len).unwrap_or(u64::MAX), permissions);
            Ok(Some(held))
        }
    }

    /// `mprotect(address, size, protection)` in POSIX's encoding.
    const MPROTECT: LibraryModel = LibraryModel::ProtectCode {
        address: Operand::Argument(0),
        size: Operand::Argument(1),
        protection: 2,
        encoding: ProtectionEncoding::Posix,
        result: ProtectionResult::ZeroOrMinusOne,
        previous: None,
        requires: Conditions::NONE,
        pages: None,
    };

    /// `VirtualProtect(address, size, protection, previous)` in Windows'
    /// encoding.
    const VIRTUAL_PROTECT: LibraryModel = LibraryModel::ProtectCode {
        address: Operand::Argument(0),
        size: Operand::Argument(1),
        protection: 2,
        encoding: ProtectionEncoding::Windows,
        result: ProtectionResult::OneOrZero,
        previous: Some(3),
        requires: Conditions::NONE,
        pages: None,
    };

    /// **Each encoding decodes every access that keeps the bytes readable,
    /// and nothing else; what it encodes decodes back.** POSIX bits without
    /// `PROT_READ`, or with a bit above `PROT_EXEC`, and Windows' no-access,
    /// execute-only and modified values decode to none.
    #[test]
    fn protection_encodings_decode_readable_accesses() {
        let permissions = |writable, executable| InstructionPermissions {
            writable,
            executable,
        };
        for (encoding, value, expected) in [
            (
                ProtectionEncoding::Posix,
                1,
                Some(permissions(false, false)),
            ),
            (ProtectionEncoding::Posix, 3, Some(permissions(true, false))),
            (ProtectionEncoding::Posix, 5, Some(permissions(false, true))),
            (ProtectionEncoding::Posix, 7, Some(permissions(true, true))),
            (ProtectionEncoding::Posix, 0, None),
            (ProtectionEncoding::Posix, 6, None),
            (ProtectionEncoding::Posix, 0x0100_0007, None),
            (
                ProtectionEncoding::Windows,
                0x02,
                Some(permissions(false, false)),
            ),
            (
                ProtectionEncoding::Windows,
                0x04,
                Some(permissions(true, false)),
            ),
            (
                ProtectionEncoding::Windows,
                0x08,
                Some(permissions(true, false)),
            ),
            (
                ProtectionEncoding::Windows,
                0x20,
                Some(permissions(false, true)),
            ),
            (
                ProtectionEncoding::Windows,
                0x40,
                Some(permissions(true, true)),
            ),
            (
                ProtectionEncoding::Windows,
                0x80,
                Some(permissions(true, true)),
            ),
            (ProtectionEncoding::Windows, 0x01, None),
            (ProtectionEncoding::Windows, 0x10, None),
            (ProtectionEncoding::Windows, 0x140, None),
        ] {
            assert_eq!(encoding.decode(value), expected, "{encoding:?} {value:#x}");
        }
        for encoding in [ProtectionEncoding::Posix, ProtectionEncoding::Windows] {
            for held in [
                permissions(false, false),
                permissions(true, false),
                permissions(false, true),
                permissions(true, true),
            ] {
                assert_eq!(encoding.decode(encoding.encode(held)), Some(held));
            }
        }
    }

    /// Runs one alternative of `model` on `arguments` under
    /// [`ProtectionGuard`] on a machine of byte order `order`, returning
    /// what it returned and its control.
    fn protect(
        model: LibraryModel,
        alternative: usize,
        arguments: Vec<Value<MockTarget>>,
        state: &mut State<MockTarget>,
        order: Option<Endianness>,
    ) -> (Option<Value<MockTarget>>, ModelControl) {
        let context = AnalysisContext::new(Machine {
            byte_order: order,
            ..MACHINE
        });
        let mut trace = Trace::new();
        let mut call = ModelCall::new(
            arguments,
            state,
            &mut trace,
            Access::new(&context).with_write_guard(Some(&ProtectionGuard)),
        );
        model.call_alternative(alternative, &mut call);
        (call.take_returned(), call.control())
    }

    /// **A protection call installs the access its argument states over the
    /// exact mapping, on success only.** `mprotect`'s `PROT_READ |
    /// PROT_EXEC` returns zero and installs it; its failure returns minus one
    /// and installs nothing; an extent the host does not attest is refused;
    /// a protection that is no constant, or no readable access, leaves the
    /// call unmodeled with nothing installed.
    #[test]
    fn protection_call_updates_only_the_successful_exact_mapping() {
        let base = Address::new(Root::Symbol, 0x2000, PointerSize::Bit64);
        let installed = InstructionPermissions {
            writable: false,
            executable: true,
        };
        let int = |value| Value::Const(ConstValue::I64(value));
        for (alternative, length, protection, expected, effective, control) in [
            (0, 8, int(5), Some(0), Some(installed), ModelControl::Return),
            (1, 8, int(5), Some(-1), None, ModelControl::Return),
            (0, 7, int(5), None, None, ModelControl::Return),
            (0, 8, Value::Unknown, None, None, ModelControl::Unsupported),
            (0, 8, int(4), None, None, ModelControl::Unsupported),
        ] {
            let mut state = state();
            state.memory_mut().store_bytes(base, b"content!");
            let (returned, ended) = protect(
                MPROTECT,
                alternative,
                vec![Value::Addr(base), int(length), protection],
                &mut state,
                None,
            );
            assert_eq!(
                returned,
                expected.map(|value| Value::Const(ConstValue::I32(value)))
            );
            assert_eq!(ended, control);
            assert_eq!(
                state.memory().instruction_sources().permissions(base),
                effective
            );
            assert_eq!(state.memory().known_run(base, 8, None), b"content!");
            assert_eq!(
                state.memory().refusal(),
                (length != 8).then_some(MemoryRefusal::InstructionSource)
            );
        }
    }

    /// **A kernel's protection call returns at the pointer width, a failure
    /// as some negated `errno`.** Linux's `mprotect` system call returns zero
    /// at the pointer width on success; on failure it returns a negated
    /// `errno`, which no one number is: what is known is its bits from 12 up,
    /// all set, so it is negative and nonzero. The control: the library's
    /// `mprotect` returns a 32-bit zero or minus one.
    #[test]
    fn a_kernel_protection_call_returns_a_negated_errno_at_the_pointer_width() {
        let kernel = ProtectionResult::ZeroOrNegatedErrno;
        assert_eq!(kernel.kind(), ModelValue::Offset);
        assert_eq!(
            kernel.returned::<MockTarget>(true, PointerSize::Bit64),
            Value::Const(ConstValue::I64(0))
        );
        let failed = kernel.returned::<MockTarget>(false, PointerSize::Bit64);
        let bits = failed.integer_bits().expect("known bits");
        assert_eq!(bits.width(), 64);
        assert_eq!(bits.ones(), u128::from(u64::MAX) & !0xfff);
        assert_eq!(bits.sign(), Some(true));
        assert!(failed.constant().is_none());
        let library = ProtectionResult::ZeroOrMinusOne;
        assert_eq!(library.kind(), ModelValue::Int32);
        assert_eq!(
            library.returned::<MockTarget>(false, PointerSize::Bit64),
            Value::Const(ConstValue::I32(-1))
        );
    }

    /// A four-byte symbol at the start of the protected mapping.
    #[derive(Debug)]
    struct ShortSymbol;

    impl ObjectExtents for ShortSymbol {
        fn object(&self, address: Address) -> Option<StorageObject> {
            let object = StorageObject {
                start: 0x2000,
                bytes: 4,
                kind: ObjectKind::Global,
                source: ExtentSource::Symbol,
            };
            object.reach(address).is_some().then_some(object)
        }
    }

    /// **A protection change is no access to the bytes it covers.** A page's
    /// protection reaches past the objects it holds: `mprotect` of the
    /// eight-byte mapping whose first four bytes are a sized symbol installs
    /// the access and finds nothing past the symbol's extent.
    #[test]
    fn a_protection_change_reaches_past_its_objects() {
        let base = Address::new(Root::Symbol, 0x2000, PointerSize::Bit64);
        let mut state = state();
        let findings = Findings::default();
        state.memory_mut().set_object_extents(Arc::new(ShortSymbol));
        state.memory_mut().set_findings(findings.clone());
        state.memory_mut().set_writer(
            0,
            WriteSite {
                body: 0,
                block: 0,
                instruction: 0,
            },
        );
        let int = |value| Value::Const(ConstValue::I64(value));
        let (returned, ended) = protect(
            MPROTECT,
            0,
            vec![Value::Addr(base), int(8), int(5)],
            &mut state,
            None,
        );
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(0))));
        assert_eq!(ended, ModelControl::Return);
        assert_eq!(findings.all(), []);
    }

    /// **`VirtualProtect` reports the access it replaces and returns a
    /// `BOOL`.** Flipping code writable writes the `PAGE_EXECUTE_READ` it
    /// held to the out cell, in the target's byte order, and returns one;
    /// flipping it back reports the `PAGE_READWRITE` installed in between.
    /// With no byte order stated the cell is written unknown. Its failure
    /// returns zero and writes nothing, and an out cell that names no
    /// address leaves the call unmodeled.
    #[test]
    fn a_protection_call_reports_the_access_it_replaces() {
        let base = Address::new(Root::Symbol, 0x2000, PointerSize::Bit64);
        let int = |value| Value::Const(ConstValue::I64(value));
        let arguments = |protection| vec![Value::Addr(base), int(8), int(protection), stack(-8)];
        let spelled = |order, value: u32| match order {
            Some(Endianness::Big) => value.to_be_bytes().to_vec(),
            Some(Endianness::Little) => value.to_le_bytes().to_vec(),
            None => Vec::new(),
        };
        for order in [Some(Endianness::Little), Some(Endianness::Big), None] {
            let mut flipped = state();
            for (protection, reported) in [(0x04, 0x20), (0x20, 0x04)] {
                let (returned, _) = protect(
                    VIRTUAL_PROTECT,
                    0,
                    arguments(protection),
                    &mut flipped,
                    order,
                );
                assert_eq!(returned, Some(Value::Const(ConstValue::I32(1))));
                assert_eq!(
                    flipped.memory().known_run(at(-8), 4, None),
                    spelled(order, reported)
                );
            }
        }

        let mut failed = state();
        let (returned, _) = protect(
            VIRTUAL_PROTECT,
            1,
            arguments(0x04),
            &mut failed,
            Some(Endianness::Little),
        );
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(0))));
        assert!(failed.memory().known_run(at(-8), 4, None).is_empty());

        let mut unnamed = state();
        let (returned, control) = protect(
            VIRTUAL_PROTECT,
            0,
            vec![Value::Addr(base), int(8), int(0x04), Value::Unknown],
            &mut unnamed,
            Some(Endianness::Little),
        );
        assert_eq!((returned, control), (None, ModelControl::Unsupported));
        assert_eq!(
            unnamed.memory().instruction_sources().permissions(base),
            None
        );
    }

    /// A guard admitting any protection change, which records the extent it
    /// was asked to change.
    #[derive(Debug, Default)]
    struct RecordingGuard(Mutex<Option<(u64, usize)>>);

    impl RecordingGuard {
        /// The extent the guard was asked to change, if any.
        fn asked(&self) -> Option<(u64, usize)> {
            self.0.lock().ok().and_then(|asked| *asked)
        }
    }

    impl MemoryWriteGuard for RecordingGuard {
        fn check_write(
            &self,
            _: &mut InstructionSources,
            _: Address,
            _: usize,
        ) -> Result<(), MemoryRefusal> {
            Ok(())
        }

        fn change_instruction_permissions(
            &self,
            sources: &mut InstructionSources,
            address: Address,
            len: usize,
            _: Option<u32>,
            permissions: InstructionPermissions,
        ) -> Result<Option<InstructionPermissions>, MemoryRefusal> {
            if let Ok(mut asked) = self.0.lock() {
                *asked = Some((address.offset().cast_unsigned(), len));
            }
            sources.set_permissions(address, u64::try_from(len).unwrap_or(u64::MAX), permissions);
            Ok(Some(InstructionPermissions {
                writable: false,
                executable: true,
            }))
        }
    }

    /// An image whose integers are its symbols' addresses (a stated load
    /// base), serving no bytes: what makes an address's bits known, which a
    /// page rounding reads.
    struct LoadedImage;

    impl ImageData for LoadedImage {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, _: u64) -> Option<ImageRegion> {
            None
        }

        fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
            None
        }

        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    /// `NtProtectVirtualMemory(process, &base, &size, protection, previous)`
    /// for the current process, over whole 4096-byte pages.
    fn nt_protect() -> LibraryModel {
        LibraryModel::ProtectCode {
            address: Operand::Through { through: 1 },
            size: Operand::Through { through: 2 },
            protection: 3,
            encoding: ProtectionEncoding::Windows,
            result: ProtectionResult::NtStatus {
                failure: 0xC000_0045_u32 as i32,
            },
            previous: Some(4),
            requires: Conditions::of([Condition {
                operand: Operand::Argument(0),
                test: Test::Equals(-1),
            }])
            .unwrap_or_else(|| process::abort()),
            pages: Some(Pages {
                size: 4096,
                exact: true,
            }),
        }
    }

    /// **A protection call through cells changes whole pages and writes the
    /// extent it changed back.** `NtProtectVirtualMemory(-1, &base, &size,
    /// PAGE_READWRITE, &previous)` with a base 0x10 into a page of an image
    /// whose load base is stated and a size of 0x20 changes the whole page:
    /// the guard is asked for 0x2000 and 4096
    /// bytes, `base` and `size` are written back as that page, `previous`
    /// holds the access it replaced, and the call returns `STATUS_SUCCESS`.
    /// Failing, it returns its status and changes nothing. The control:
    /// another process leaves the call unmodeled.
    #[test]
    fn a_protection_call_through_cells_changes_whole_pages() {
        let prepared = || {
            let mut state = state();
            state.memory_mut().store_address(
                at(-8),
                Address::new(Root::Symbol, 0x2010, PointerSize::Bit64),
                8,
            );
            state
                .memory_mut()
                .store_bytes(at(-16), &0x20_u64.to_le_bytes());
            state
        };
        let arguments = |process: i64| {
            vec![
                Value::Const(ConstValue::I64(process)),
                stack(-8),
                stack(-16),
                Value::Const(ConstValue::I64(0x04)),
                stack(-24),
            ]
        };
        let run = |alternative, process, state: &mut State<MockTarget>, guard: &RecordingGuard| {
            let image = LoadedImage;
            let context = AnalysisContext::new(Machine {
                byte_order: Some(Endianness::Little),
                ..MACHINE
            })
            .with_image(&image);
            let mut trace = Trace::new();
            let mut call = ModelCall::new(
                arguments(process),
                state,
                &mut trace,
                Access::new(&context).with_write_guard(Some(guard)),
            );
            nt_protect().call_alternative(alternative, &mut call);
            (call.take_returned(), call.control())
        };

        let guard = RecordingGuard::default();
        let mut changed = prepared();
        let (returned, _) = run(0, -1, &mut changed, &guard);
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(0))));
        assert_eq!(guard.asked(), Some((0x2000, 4096)));
        let Loaded::Address(base) = changed.memory().load(at(-8), 8, None) else {
            process::abort()
        };
        assert_eq!(base.image_address(), Some(0x2000));
        assert_eq!(
            changed.memory().known_run(at(-16), 8, None),
            4096_u64.to_le_bytes()
        );
        assert_eq!(
            changed.memory().known_run(at(-24), 4, None),
            0x20_u32.to_le_bytes()
        );

        let guard = RecordingGuard::default();
        let mut failed = prepared();
        let (returned, _) = run(1, -1, &mut failed, &guard);
        assert_eq!(
            returned,
            Some(Value::Const(ConstValue::I32(0xC000_0045_u32 as i32)))
        );
        assert_eq!(guard.asked(), None);
        assert_eq!(
            failed.memory().known_run(at(-16), 8, None),
            0x20_u64.to_le_bytes()
        );

        let guard = RecordingGuard::default();
        let (_, control) = run(0, 4, &mut prepared(), &guard);
        assert_eq!(control, ModelControl::Unsupported);
        assert_eq!(guard.asked(), None);
    }

    /// An image of a read-only region at `0x1000` holding format strings and
    /// a writable one at `0x3000`, for the routines that write data.
    struct DataImage;

    impl DataImage {
        /// The read-only region's bytes: `"%s\n"` at `0x1000`, `"%d%n"` at
        /// `0x1010`.
        const READ_ONLY: [u8; 0x20] =
            *b"%s\n\0\0\0\0\0\0\0\0\0\0\0\0\0%d%n\0\0\0\0\0\0\0\0\0\0\0\0";
    }

    impl ImageData for DataImage {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            [
                (0x1000, 0x1020, Mutability::ReadOnly),
                (0x3000, 0x3100, Mutability::Writable),
            ]
            .into_iter()
            .find(|&(start, end, _)| (start..end).contains(&address))
            .map(|(start, end, mutability)| ImageRegion {
                start,
                end,
                mutability,
            })
        }

        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            let start = usize::try_from(address.checked_sub(0x1000)?).ok()?;
            Self::READ_ONLY.get(start..start.checked_add(len)?)
        }
    }

    /// A source audit: every unresolved write is refused, but one at a
    /// known stack pointer where `stack` says the stack holds no code, and
    /// every one where `unplaced` says nothing is code; data a routine that
    /// writes no code writes only where `data` says; and each placed write
    /// is recorded.
    #[derive(Debug, Default)]
    struct AuditGuard {
        data: bool,
        stack: bool,
        unplaced: bool,
        written: Mutex<Vec<(u64, usize)>>,
    }

    impl AuditGuard {
        fn written(&self) -> Vec<(u64, usize)> {
            self.written
                .lock()
                .map(|written| written.clone())
                .unwrap_or_default()
        }
    }

    impl MemoryWriteGuard for AuditGuard {
        fn check_unresolved_write(
            &self,
            pointer: Option<Address>,
            _: Option<usize>,
        ) -> Result<(), MemoryRefusal> {
            if self.unplaced
                || (self.stack && pointer.is_some_and(|pointer| pointer.root().is_stack_storage()))
            {
                Ok(())
            } else {
                Err(MemoryRefusal::UnresolvedWrite)
            }
        }

        fn check_unresolved_data_write(&self) -> Result<(), MemoryRefusal> {
            if self.data {
                Ok(())
            } else {
                self.check_unresolved_write(None, None)
            }
        }

        fn check_write(
            &self,
            _: &mut InstructionSources,
            address: Address,
            len: usize,
        ) -> Result<(), MemoryRefusal> {
            if let (Ok(mut written), Some(at)) = (self.written.lock(), address.image_address()) {
                written.push((at, len));
            }
            Ok(())
        }
    }

    fn symbol(address: i64) -> Value<MockTarget> {
        Value::Addr(Address::new(Root::Symbol, address, PointerSize::Bit64))
    }

    /// Runs a routine that writes data, `writes`, on `arguments` under
    /// `guard` ([`run_model`]).
    fn write_data(
        writes: DataWrites,
        arguments: Vec<Value<MockTarget>>,
        guard: &AuditGuard,
    ) -> State<MockTarget> {
        run_model(&LibraryModel::WritesData(writes), arguments, guard)
    }

    /// Runs `model` on `arguments` over the data image under `guard`, with
    /// the stack's `-16` holding `"abc"` and the fetch vouched for
    /// beforehand.
    fn run_model(
        model: &LibraryModel,
        arguments: Vec<Value<MockTarget>>,
        guard: &AuditGuard,
    ) -> State<MockTarget> {
        let image = DataImage;
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        })
        .with_image(&image);
        let mut state = state();
        state.memory_mut().store_bytes(at(-16), b"abc");
        state
            .memory_mut()
            .instruction_sources_mut()
            .synchronize_fetch();
        let mut trace = Trace::new();
        let mut call = ModelCall::new(
            arguments,
            &mut state,
            &mut trace,
            Access::new(&context).with_write_guard(Some(guard)),
        );
        model.call(&mut call);
        state
    }

    /// **A routine that writes no code leaves the code vouched for.** It
    /// writes the rest of the object its pointer reaches, through the audit
    /// as any write, and what escaped; the private stack it was handed no
    /// pointer into keeps its bytes, and so does what it only reads. A
    /// routine that may write anything leaves no fetch vouched for.
    #[test]
    fn a_routine_writing_data_writes_what_its_pointers_reach_and_no_code() {
        let guard = AuditGuard {
            data: true,
            ..AuditGuard::default()
        };
        let written = write_data(
            DataWrites {
                read_only: ArgumentSet::EMPTY.with(1),
                ..DataWrites::default()
            },
            vec![symbol(0x3010), symbol(0x1000)],
            &guard,
        );
        assert_eq!(written.memory().refusal(), None);
        assert_eq!(guard.written(), vec![(0x3010, 0xf0)]);
        assert_eq!(written.memory().known_run(at(-16), 3, None), b"abc");
        assert!(written.memory().instruction_sources().fetch_synchronized());

        let mut clobbered = state();
        clobbered
            .memory_mut()
            .instruction_sources_mut()
            .synchronize_fetch();
        clobbered.memory_mut().clobber();
        assert!(
            !clobbered
                .memory()
                .instruction_sources()
                .fetch_synchronized()
        );
    }

    /// **A routine that writes data forgets the writable image but keeps
    /// what binds code.** Writable image bytes, written before or never,
    /// read unknown after it (a copy-relocated `optind` after `getopt`), and
    /// read-only ones keep the image's bytes. A pointer cell of the writable
    /// image keeps its initial binding (an import's slot): the routine writes
    /// no code, and no slot binding code. The control: a call that may write
    /// anything drops the binding.
    #[test]
    fn a_routine_writing_data_forgets_writable_image_and_keeps_bindings() {
        let image = DataImage;
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        })
        .with_image(&image);
        let guard = AuditGuard {
            data: true,
            ..AuditGuard::default()
        };
        let address = |at: i64| Address::new(Root::Symbol, at, PointerSize::Bit64);
        let mut state = state();
        state.memory_mut().store_bytes(address(0x3040), b"x");
        let mut trace = Trace::new();
        let mut call = ModelCall::new(
            Vec::new(),
            &mut state,
            &mut trace,
            Access::new(&context).with_write_guard(Some(&guard)),
        );
        LibraryModel::WritesData(DataWrites::default()).call(&mut call);
        let memory = state.memory();
        assert_eq!(memory.refusal(), None);
        for at in [0x3040, 0x3020] {
            assert!(
                matches!(memory.load(address(at), 1, Some(&image)), Loaded::Unknown),
                "{at:#x}"
            );
        }
        assert!(matches!(
            memory.load(address(0x1000), 2, Some(&image)),
            Loaded::Bytes(bytes) if bytes == b"%s"
        ));
        assert!(memory.initial_image_binding(address(0x3080), 8));

        state.memory_mut().clobber();
        assert!(!state.memory().initial_image_binding(address(0x3080), 8));
    }

    /// **A pointer into the frame makes the frame escape.** Nothing bounds
    /// what of the frame such a pointer reaches, so the routine may have
    /// written all of it; one it only reads leaves the frame private. The
    /// write is one of unknown length the policy judges: a guard for which
    /// the stack may hold code refuses it.
    #[test]
    fn a_pointer_into_the_frame_a_routine_writes_through_forgets_the_frame() {
        let refusing = AuditGuard {
            data: true,
            ..AuditGuard::default()
        };
        let state = write_data(DataWrites::default(), vec![stack(-16)], &refusing);
        assert_eq!(
            state.memory().refusal(),
            Some(MemoryRefusal::UnresolvedWrite)
        );
        for (read_only, kept) in [
            (ArgumentSet::EMPTY, false),
            (ArgumentSet::EMPTY.with(0), true),
        ] {
            let guard = AuditGuard {
                data: true,
                stack: true,
                ..AuditGuard::default()
            };
            let state = write_data(
                DataWrites {
                    read_only,
                    ..DataWrites::default()
                },
                vec![stack(-16)],
                &guard,
            );
            assert_eq!(state.memory().refusal(), None);
            assert_eq!(
                state.memory().known_run(at(-16), 3, None) == b"abc",
                kept,
                "{read_only:?}"
            );
        }
    }

    /// **An unknown pointer is an unresolved write unless its type holds no
    /// code.** `read(fd, buf, n)` into code is how a program loads code, so
    /// a byte buffer nobody can place may be code; a `FILE *` cannot be.
    /// A known integer is no pointer: null reaches nothing. So is a pointer
    /// to an object nobody can place (a `FILE *` loaded from `stdout`), and a
    /// typed one into the frame escapes the frame without asking the policy.
    #[test]
    fn an_unknown_pointer_is_unresolved_unless_its_type_holds_no_code() {
        let opaque = Value::Addr(Address::new(
            Root::Opaque {
                activation: 0,
                value: SsaVarId::from_index(3),
                run: 0,
            },
            0,
            PointerSize::Bit64,
        ));
        for (argument, typed, refused) in [
            (Value::Unknown, ArgumentSet::EMPTY, true),
            (Value::Unknown, ArgumentSet::EMPTY.with(0), false),
            (Value::Const(ConstValue::I64(0)), ArgumentSet::EMPTY, false),
            (opaque.clone(), ArgumentSet::EMPTY, true),
            (opaque.clone(), ArgumentSet::EMPTY.with(0), false),
            (stack(-16), ArgumentSet::EMPTY, true),
            (stack(-16), ArgumentSet::EMPTY.with(0), false),
        ] {
            let guard = AuditGuard {
                data: true,
                ..AuditGuard::default()
            };
            let state = write_data(
                DataWrites {
                    typed,
                    ..DataWrites::default()
                },
                vec![argument.clone()],
                &guard,
            );
            assert_eq!(
                state.memory().refusal(),
                refused.then_some(MemoryRefusal::UnresolvedWrite),
                "{argument:?} {typed:?}"
            );
        }
    }

    /// **An admitted write nobody can place may write code; one into the
    /// stack may not.** Where the policy admits it, a write through a
    /// pointer nobody knows, or into an object nobody can place, leaves no
    /// fetch vouched for; one into the frame, which a policy admits only
    /// where the stack holds no code, forgets the frame's data and keeps the
    /// fetch. The private frame the others were handed no pointer into keeps
    /// its bytes.
    #[test]
    fn an_admitted_unplaced_write_forgets_code_only_where_code_may_be() {
        let opaque = Value::Addr(Address::new(
            Root::Opaque {
                activation: 0,
                value: SsaVarId::from_index(3),
                run: 0,
            },
            0,
            PointerSize::Bit64,
        ));
        for (argument, synchronized, frame_kept) in [
            (Value::Unknown, false, true),
            (opaque, false, true),
            (stack(-16), true, false),
        ] {
            let guard = AuditGuard {
                data: true,
                unplaced: true,
                ..AuditGuard::default()
            };
            let state = write_data(DataWrites::default(), vec![argument.clone()], &guard);
            assert_eq!(state.memory().refusal(), None, "{argument:?}");
            assert_eq!(
                state.memory().instruction_sources().fetch_synchronized(),
                synchronized,
                "{argument:?}"
            );
            assert_eq!(
                state.memory().known_run(at(-16), 3, None) == b"abc",
                frame_kept,
                "{argument:?}"
            );
        }
    }

    /// **A string copy nobody can read forgets its destination, not all
    /// memory.** `strcpy(buffer, s)` with `s` unknown writes from `buffer`
    /// to the end of the object it lies in, through the audit as any write;
    /// the frame it was handed no pointer into keeps its bytes, and the
    /// fetch stays vouched for. Appending does the same past what the
    /// buffer holds.
    #[test]
    fn an_unknown_string_forgets_the_rest_of_its_destination() {
        for (model, written) in [
            // The copy checks the first byte it must write first.
            (LibraryModel::CopyString, vec![(0x3010, 1), (0x3010, 0xf0)]),
            (LibraryModel::AppendString, vec![(0x3010, 0xf0)]),
        ] {
            let guard = AuditGuard::default();
            let state = run_model(&model, vec![symbol(0x3010), Value::Unknown], &guard);
            assert_eq!(state.memory().refusal(), None, "{model:?}");
            assert_eq!(guard.written(), written, "{model:?}");
            assert_eq!(
                state.memory().known_run(at(-16), 3, None),
                b"abc",
                "{model:?}"
            );
            assert!(
                state.memory().instruction_sources().fetch_synchronized(),
                "{model:?}"
            );
        }
    }

    /// **A routine that keeps a function it is handed runs as its model.**
    /// `signal(sig, handler)` stores `handler` and calls nothing, so the
    /// model accounts for that argument and writes nothing through it. The
    /// control: one that keeps nothing accounts for no function and writes
    /// through the same pointer.
    #[test]
    fn a_routine_keeping_a_function_accounts_for_it_and_writes_nothing_through_it() {
        for (kept, written) in [
            (ArgumentSet::EMPTY.with(0), Vec::new()),
            (ArgumentSet::EMPTY, vec![(0x3010, 0xf0)]),
        ] {
            let model = LibraryModel::WritesData(DataWrites {
                kept,
                ..DataWrites::default()
            });
            assert_eq!(model.accounts_for_callback(0), !kept.is_empty());
            assert!(!model.accounts_for_callback(1));
            let guard = AuditGuard {
                data: true,
                ..AuditGuard::default()
            };
            let state = run_model(&model, vec![symbol(0x3010)], &guard);
            assert_eq!(state.memory().refusal(), None, "{kept:?}");
            assert_eq!(guard.written(), written, "{kept:?}");
        }
    }

    /// **A guard that says nothing of data refuses it.** The default answer
    /// for a routine's escaped data is the guard's answer for any unresolved
    /// write.
    #[test]
    fn a_guard_that_says_nothing_of_data_refuses_what_escaped() {
        let guard = AuditGuard::default();
        let state = write_data(DataWrites::default(), Vec::new(), &guard);
        assert_eq!(
            state.memory().refusal(),
            Some(MemoryRefusal::UnresolvedWrite)
        );
    }

    /// **A format decides the arguments after it.** `fprintf(f, "%s\n", s)`
    /// writes nothing through `s`; `"%d%n"` writes through its pointer, and
    /// so may a format nobody can read.
    #[test]
    fn variadic_pointers_are_written_only_where_the_format_names_n() {
        for (format, kept) in [
            (symbol(0x1000), true),
            (symbol(0x1010), false),
            (Value::Unknown, false),
        ] {
            let guard = AuditGuard {
                data: true,
                stack: true,
                ..AuditGuard::default()
            };
            let state = write_data(
                DataWrites {
                    read_only: ArgumentSet::EMPTY.with(0),
                    format: Some(0),
                    ..DataWrites::default()
                },
                vec![format.clone(), stack(-16)],
                &guard,
            );
            assert_eq!(state.memory().refusal(), None, "{format:?}");
            assert_eq!(
                state.memory().known_run(at(-16), 3, None) == b"abc",
                kept,
                "{format:?}"
            );
        }
    }

    /// **A `%n` through a `va_list` may write anything.** `vfprintf`'s
    /// arguments are held where no argument names them, so a format that
    /// names `%n`, or one nobody can read, is a write nobody can place,
    /// which the audit refuses; one that names none writes only data.
    #[test]
    fn a_format_writing_through_a_list_is_unresolved() {
        for (format, refused) in [
            (symbol(0x1000), false),
            (symbol(0x1010), true),
            (Value::Unknown, true),
        ] {
            let guard = AuditGuard {
                data: true,
                ..AuditGuard::default()
            };
            let state = write_data(
                DataWrites {
                    read_only: ArgumentSet::EMPTY.with(0),
                    format: Some(0),
                    list: Some(1),
                    ..DataWrites::default()
                },
                vec![format.clone(), stack(-16)],
                &guard,
            );
            assert_eq!(
                state.memory().refusal(),
                refused.then_some(MemoryRefusal::UnresolvedWrite),
                "{format:?}"
            );
        }
    }

    /// Runs `model` on `arguments` over `state`, returning what it returned.
    fn call(
        model: LibraryModel,
        arguments: Vec<Value<MockTarget>>,
        state: &mut State<MockTarget>,
        trace: &mut Trace,
    ) -> Option<Value<MockTarget>> {
        let context = AnalysisContext::new(MACHINE);
        let mut call = ModelCall::new(arguments, state, trace, Access::new(&context));
        model.call(&mut call);
        call.take_returned()
    }

    fn state() -> State<MockTarget> {
        State::new(
            Frame::enter_at(
                &SsaFunction::new(0, 0),
                0,
                0,
                Address::new(Root::Stack, 0, PointerSize::Bit64),
                vec![],
                PointerSize::Bit64,
            )
            .with_stack_extent(Some(-64..0)),
        )
    }

    fn at(offset: i64) -> Address {
        Address::new(Root::Stack, offset, PointerSize::Bit64)
    }

    /// `memcpy` copies known bytes, records its writes, and keeps a loop
    /// replaceable; `strlen` measures a known string.
    #[test]
    fn copy_and_length_act_on_known_bytes() {
        let mut state = state();
        let mut trace = Trace::new();
        state.memory_mut().store_bytes(at(-16), b"abc\0");
        trace.enter_loop(0, 0);
        let returned = call(
            LibraryModel::Copy,
            vec![stack(-8), stack(-16), Value::Const(ConstValue::I64(4))],
            &mut state,
            &mut trace,
        );
        assert_eq!(returned, Some(stack(-8)));
        assert_eq!(state.memory().known_run(at(-8), 4, None), b"abc\0".to_vec());
        assert_eq!(
            trace.loops().first().map(|visit| visit.written.len()),
            Some(4)
        );
        assert!(<LibraryModel as CallModel<MockTarget>>::only_memory(
            &LibraryModel::Copy
        ));
        let length = call(
            LibraryModel::Length,
            vec![stack(-8)],
            &mut state,
            &mut trace,
        );
        assert_eq!(length, Some(Value::Const(ConstValue::I64(3))));
    }

    /// **A model's returned argument is what it returns.** A copy, a fill, a
    /// string copy and a string append each hand back the argument
    /// `returned_argument` names, run on known bytes; a length is a value of
    /// its own and names none.
    #[test]
    fn a_models_returned_argument_is_what_it_returns() {
        let four = Value::Const(ConstValue::I64(4));
        for (model, arguments) in [
            (
                LibraryModel::Copy,
                vec![stack(-32), stack(-16), four.clone()],
            ),
            (
                LibraryModel::Fill,
                vec![stack(-32), Value::Const(ConstValue::I32(0x78)), four],
            ),
            (LibraryModel::CopyString, vec![stack(-32), stack(-16)]),
            (LibraryModel::AppendString, vec![stack(-32), stack(-16)]),
            (LibraryModel::Length, vec![stack(-16)]),
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-32), b"xy\0");
            state.memory_mut().store_bytes(at(-16), b"abc\0");
            let expected = model
                .returned_argument()
                .and_then(|index| arguments.get(index).cloned());
            let returned = call(model, arguments.clone(), &mut state, &mut trace);
            match expected {
                Some(argument) => assert_eq!(returned, Some(argument), "{model:?}"),
                None => assert_ne!(returned, arguments.first().cloned(), "{model:?}"),
            }
        }
    }

    /// Unknown or absent string addresses cannot justify removing their reads.
    #[test]
    fn length_model_requires_a_resolved_input_address() {
        for (arguments, expected) in [
            (vec![], None),
            (vec![Value::Unknown], None),
            (vec![Value::StackDerived], Some(Value::StackDerived)),
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-32), b"safe\0");
            trace.enter_loop(0, 0);
            assert_eq!(
                call(LibraryModel::Length, arguments, &mut state, &mut trace),
                expected
            );
            assert!(trace.loops().iter().all(|visit| !visit.clean));
            assert_eq!(state.memory().refusal(), None);
            assert_eq!(state.memory().known_run(at(-32), 5, None), b"safe\0");
        }
        let mut state = state();
        let mut trace = Trace::new();
        state.memory_mut().store_bytes(at(-32), b"safe\0");
        trace.enter_loop(0, 0);
        assert_eq!(
            call(
                LibraryModel::Length,
                vec![stack(-32)],
                &mut state,
                &mut trace
            ),
            Some(Value::Const(ConstValue::I64(4)))
        );
        assert!(trace.loops().iter().all(|visit| visit.clean));
    }

    /// A model costs a step per byte each of its memory operations spans: a
    /// copy or fill its length, a string scan the bytes it inspected and the
    /// terminator, a string copy its scan and its write.
    #[test]
    fn library_models_charge_a_step_per_byte_they_span() {
        let context = AnalysisContext::new(MACHINE);
        let five = Value::Const(ConstValue::I64(5));
        for (name, model, arguments, work) in [
            (
                "copy",
                LibraryModel::Copy,
                vec![stack(-32), stack(-16), five.clone()],
                5,
            ),
            (
                "fill",
                LibraryModel::Fill,
                vec![stack(-32), Value::Const(ConstValue::I32(0x78)), five],
                5,
            ),
            ("length", LibraryModel::Length, vec![stack(-16)], 7),
            (
                "copy string",
                LibraryModel::CopyString,
                vec![stack(-32), stack(-16)],
                14,
            ),
            (
                "append string",
                LibraryModel::AppendString,
                vec![stack(-32), stack(-16)],
                17,
            ),
            ("no writes", LibraryModel::NoWrites, vec![stack(-16)], 0),
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-32), b"xy\0");
            state.memory_mut().store_bytes(at(-16), b"abcdef\0");
            let mut call = ModelCall::new(arguments, &mut state, &mut trace, Access::new(&context));
            model.call(&mut call);
            assert_eq!(call.work(), work, "{name}");
        }
    }

    /// Both overlap directions and identity read the source before writing.
    #[test]
    fn copy_preserves_overlapping_source_bytes() {
        for (dest, source, expected) in [
            (-15, -16, b"aabcde".as_slice()),
            (-16, -15, b"bcdeff".as_slice()),
            (-16, -16, b"abcdef".as_slice()),
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-16), b"abcdef");
            let returned = call(
                LibraryModel::Copy,
                vec![stack(dest), stack(source), Value::Const(ConstValue::I64(5))],
                &mut state,
                &mut trace,
            );
            assert_eq!(returned, Some(stack(dest)));
            assert_eq!(state.memory().known_run(at(-16), 6, None), expected);
        }
    }

    /// Unknown bytes overwrite the destination without hiding later known bytes.
    #[test]
    fn overlapping_copy_preserves_known_suffix_after_unknown_byte() {
        let mut state = state();
        let mut trace = Trace::new();
        state.memory_mut().store_bytes(at(-16), b"abcdef");
        state.memory_mut().store_unknown(at(-14), 1);
        call(
            LibraryModel::Copy,
            vec![stack(-15), stack(-16), Value::Const(ConstValue::I64(5))],
            &mut state,
            &mut trace,
        );
        assert_eq!(state.memory().known_run(at(-16), 6, None), b"aab");
        assert_eq!(state.memory().known_run(at(-13), 1, None), b"");
        assert_eq!(state.memory().known_run(at(-12), 2, None), b"de");
    }

    /// A copy of an unknown length forgets what it may have reached.
    #[test]
    fn an_unknown_length_clobbers() {
        let mut state = state();
        state.memory_mut().store_bytes(at(-8), b"secret");
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        call(
            LibraryModel::Copy,
            vec![stack(-8), stack(-16), Value::Unknown],
            &mut state,
            &mut trace,
        );
        assert!(trace.loops().iter().all(|visit| !visit.clean));
        assert!(state.memory().known_run(at(-8), 6, None).is_empty());
    }

    /// The family the allocation tests' objects belong to.
    const LIBC: AllocatorFamily = AllocatorFamily(1);

    /// `malloc(n)`, or with `zeroed` `calloc` of a size alone.
    const fn malloc(zeroed: bool) -> LibraryModel {
        LibraryModel::Allocate(Allocation {
            family: LIBC,
            size: Operand::Argument(0),
            count: None,
            alignment: None,
            zeroed,
            flags: None,
            delivery: Delivery::Returned,
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        })
    }

    /// `free(pointer)` of `family`.
    const fn free(family: AllocatorFamily) -> LibraryModel {
        LibraryModel::Free(Deallocation {
            family,
            pointer: Operand::Argument(0),
            size: None,
            result: FreeResult::Nothing,
            requires: Conditions::NONE,
            whole: None,
            released: None,
            heap: None,
        })
    }

    /// **A model states the memory a call touches, and nothing else.** A block
    /// copy replaces its destination's counted bytes and reads its source's;
    /// a fill replaces its destination's; a string written may leave bytes
    /// past it, so it may write, unbounded; ending an object changes its
    /// bytes to nothing stated; an allocator may set the routine's own state
    /// (`errno`, escaped) and, delivering through an out argument, its cell; a
    /// routine writing nothing reads what escaped; one that never returns
    /// states nothing a caller sees. The control is the model of `free`: its
    /// object's bytes and no other, so a load of another cell after it reads
    /// what the cell held before (dash's `argstr` reloads `in.fd` past
    /// `free(in.buf)`).
    #[test]
    fn a_model_states_the_memory_a_call_touches() {
        let only = |model: LibraryModel| match model.memory::<MockTarget>(PointerSize::Bit64) {
            CallMemory::Only(accesses) => accesses
                .into_iter()
                .map(|access| (access.location, access.access))
                .collect::<Vec<_>>(),
            CallMemory::Unknown => panic!("{model:?} states no memory"),
        };
        let at = |arg: u32, extent: PointeeExtent| CallLocation::Pointee { arg, extent };
        let unbounded = |arg: u32| at(arg, PointeeExtent::Unbounded);
        assert_eq!(
            only(LibraryModel::Copy),
            vec![
                (at(0, PointeeExtent::InArgument(2)), AccessMode::Write),
                (at(1, PointeeExtent::InArgument(2)), AccessMode::Read),
            ]
        );
        assert_eq!(
            only(LibraryModel::Fill),
            vec![(at(0, PointeeExtent::InArgument(2)), AccessMode::Write)]
        );
        assert_eq!(
            only(LibraryModel::Length),
            vec![(unbounded(0), AccessMode::Read)]
        );
        assert_eq!(
            only(LibraryModel::CopyString),
            vec![
                (unbounded(0), AccessMode::MayWrite),
                (unbounded(1), AccessMode::Read)
            ]
        );
        assert_eq!(
            only(free(LIBC)),
            vec![(unbounded(0), AccessMode::MayWrite)],
            "free touches its object alone"
        );
        assert_eq!(
            only(malloc(false)),
            vec![(CallLocation::Escaped, AccessMode::MayWrite)],
            "an allocator may set errno"
        );
        let memalign = LibraryModel::Allocate(Allocation {
            family: LIBC,
            size: Operand::Argument(2),
            count: None,
            alignment: Some(1),
            zeroed: false,
            flags: None,
            delivery: Delivery::Out {
                out: 0,
                failure: 12,
            },
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        });
        assert_eq!(
            only(memalign),
            vec![
                (CallLocation::Escaped, AccessMode::MayWrite),
                (at(0, PointeeExtent::Bytes(8)), AccessMode::MayWrite),
            ]
        );
        assert_eq!(
            only(LibraryModel::NoWrites),
            vec![(CallLocation::Escaped, AccessMode::Read)]
        );
        assert_eq!(
            LibraryModel::NoReturn.memory::<MockTarget>(PointerSize::Bit64),
            CallMemory::Unknown
        );
    }

    /// Runs one alternative of `model`, returning what it returned and its
    /// control.
    fn alternative(
        model: LibraryModel,
        alternative: usize,
        arguments: Vec<Value<MockTarget>>,
        state: &mut State<MockTarget>,
    ) -> (Option<Value<MockTarget>>, ModelControl) {
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        });
        let mut trace = Trace::new();
        let mut call = ModelCall::new(arguments, state, &mut trace, Access::new(&context));
        model.call_alternative(alternative, &mut call);
        (call.take_returned(), call.control())
    }

    /// The heap object `returned` names.
    fn object(returned: Option<Value<MockTarget>>) -> Address {
        returned
            .and_then(|value| value.address())
            .filter(|address| matches!(address.root(), Root::Heap(_)))
            .unwrap_or_else(|| process::abort())
    }

    /// A known unsigned integer argument.
    fn number(value: u64) -> Value<MockTarget> {
        Value::Const(ConstValue::U64(value))
    }

    /// A zeroing allocation is a fresh block of zeros.
    #[test]
    fn a_zeroed_allocation_is_zero() {
        let mut state = state();
        let mut trace = Trace::new();
        let block = object(call(malloc(true), vec![number(3)], &mut state, &mut trace));
        assert_eq!(state.memory().known_run(block, 8, None), vec![0, 0, 0]);
        assert_eq!(state.memory().object(block.root()), Some((3, Some(LIBC))));
    }

    /// Zeroing size is metadata, while overflow must select null rather than a small block.
    #[test]
    fn allocation_models_keep_large_extents_lazy_and_reject_overflow() {
        let mut state = state();
        let mut trace = Trace::new();
        let block = object(call(
            malloc(true),
            vec![number(1 << 40)],
            &mut state,
            &mut trace,
        ));
        assert_eq!(state.memory().known_run(block, 4, None), [0; 4]);
        let tail = block
            .moved((1 << 40) - 1, PointerSize::Bit64)
            .unwrap_or_else(|| process::abort());
        assert_eq!(state.memory().known_run(tail, 2, None), [0]);
        let calloc = LibraryModel::Allocate(Allocation {
            family: LIBC,
            size: Operand::Argument(1),
            count: Some(0),
            alignment: None,
            zeroed: true,
            flags: None,
            delivery: Delivery::Returned,
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        });
        let null = call(
            calloc,
            vec![number(u64::MAX), number(2)],
            &mut state,
            &mut trace,
        );
        assert_eq!(null, Some(Value::Const(ConstValue::I64(0))));
    }

    /// **A flags argument zeroes, or leaves the call unmodeled.**
    /// `HeapAlloc(heap, flags, n)` zeroes for `HEAP_ZERO_MEMORY` and leaves its
    /// bytes unknown without it; `HEAP_GENERATE_EXCEPTIONS`, which raises
    /// rather than returning null, and flags no constant states, leave the
    /// call unmodeled on both alternatives.
    #[test]
    fn allocation_flags_zero_or_leave_the_call_unmodeled() {
        let heap_alloc = LibraryModel::Allocate(Allocation {
            family: AllocatorFamily(2),
            size: Operand::Argument(2),
            count: None,
            alignment: None,
            zeroed: false,
            flags: Some(AllocationFlags {
                argument: 1,
                zeroing: 0x8,
                unmodeled: 0x4,
            }),
            delivery: Delivery::Returned,
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        });
        let heap = Value::Addr(Address::new(Root::Symbol, 0x40, PointerSize::Bit64));
        for (flags, zeroed) in [(number(0x8), true), (number(0x9), true), (number(0), false)] {
            let mut state = state();
            let (returned, _) = alternative(
                heap_alloc,
                0,
                vec![heap.clone(), flags, number(4)],
                &mut state,
            );
            let block = object(returned);
            assert_eq!(
                state.memory().known_run(block, 4, None),
                if zeroed { vec![0; 4] } else { Vec::new() }
            );
        }
        for flags in [number(0xc), Value::Unknown] {
            for which in 0..2 {
                let mut state = state();
                let (returned, control) = alternative(
                    heap_alloc,
                    which,
                    vec![heap.clone(), flags.clone(), number(4)],
                    &mut state,
                );
                assert_eq!((returned, control), (None, ModelControl::Unsupported));
            }
        }
    }

    /// `MEM_COMMIT | MEM_RESERVE`.
    const COMMIT: u64 = 0x3000;

    /// `PAGE_READWRITE`.
    const READ_WRITE: u64 = 0x04;

    /// A page allocator of family 7, `VirtualAlloc(address, size, type,
    /// protection)`: the address null, the type committing, pages of 4096
    /// bytes, exactly where `exact`.
    fn page_allocator(exact: bool) -> LibraryModel {
        LibraryModel::Allocate(Allocation {
            family: AllocatorFamily(7),
            size: Operand::Argument(1),
            count: None,
            alignment: None,
            zeroed: true,
            flags: None,
            delivery: Delivery::Returned,
            requires: Conditions::of([
                Condition {
                    operand: Operand::Argument(0),
                    test: Test::Equals(0),
                },
                Condition {
                    operand: Operand::Argument(2),
                    test: Test::AllSet(0x1000),
                },
            ])
            .unwrap_or_else(|| process::abort()),
            pages: Some(Pages { size: 4096, exact }),
            protection: Some(AllocatedProtection {
                operand: Operand::Argument(3),
                encoding: ProtectionEncoding::Windows,
            }),
            heap: None,
        })
    }

    /// `NtAllocateVirtualMemory(process, &base, zero_bits, &size, type,
    /// protection)` of family 8: the current process, a null base, a
    /// committing type; the block delivered to `base`, its size read and
    /// written back through `size`.
    fn native_allocator() -> LibraryModel {
        LibraryModel::Allocate(Allocation {
            family: AllocatorFamily(8),
            size: Operand::Through { through: 3 },
            count: None,
            alignment: None,
            zeroed: true,
            flags: None,
            delivery: Delivery::Out {
                out: 1,
                failure: 0xC000_0017_u32 as i32,
            },
            requires: Conditions::of([
                Condition {
                    operand: Operand::Argument(0),
                    test: Test::Equals(-1),
                },
                Condition {
                    operand: Operand::Through { through: 1 },
                    test: Test::Equals(0),
                },
                Condition {
                    operand: Operand::Argument(4),
                    test: Test::AllSet(0x1000),
                },
            ])
            .unwrap_or_else(|| process::abort()),
            pages: Some(Pages {
                size: 4096,
                exact: true,
            }),
            protection: Some(AllocatedProtection {
                operand: Operand::Argument(5),
                encoding: ProtectionEncoding::Windows,
            }),
            heap: None,
        })
    }

    /// `NtFreeVirtualMemory(process, &base, &size, type)` of family 8: the
    /// current process, a size of zero, `MEM_RELEASE`; the released size
    /// written back through `size`.
    fn native_release() -> LibraryModel {
        LibraryModel::Free(Deallocation {
            family: AllocatorFamily(8),
            pointer: Operand::Through { through: 1 },
            size: None,
            result: FreeResult::Zero,
            requires: Conditions::of([
                Condition {
                    operand: Operand::Argument(0),
                    test: Test::Equals(-1),
                },
                Condition {
                    operand: Operand::Through { through: 2 },
                    test: Test::Equals(0),
                },
                Condition {
                    operand: Operand::Argument(3),
                    test: Test::AllSet(0x8000),
                },
            ])
            .unwrap_or_else(|| process::abort()),
            whole: None,
            released: Some(Operand::Through { through: 2 }),
            heap: None,
        })
    }

    /// The process `-1` names: the current one.
    fn current_process() -> Value<MockTarget> {
        Value::Const(ConstValue::I64(-1))
    }

    /// **A page allocator grants whole pages, under the requirements its
    /// routine states.** `VirtualAlloc(NULL, 100, MEM_COMMIT | MEM_RESERVE,
    /// PAGE_READWRITE)` makes a zeroed block of a page: its last byte is
    /// inside it and the next one past it. Where the platform's page size is
    /// not fixed, the page is only the least the block holds, and no access is
    /// past it. A mapping that fails returns its routine's failure value,
    /// `mmap`'s minus one. The controls: an address to place the block at, a
    /// type that only reserves, an executable protection and an unknown one
    /// each leave the call unmodeled.
    #[test]
    fn a_page_allocator_grants_whole_pages_under_its_requirements() {
        let arguments = |address: Value<MockTarget>, kind: u64, protection: Value<MockTarget>| {
            vec![address, number(100), number(kind), protection]
        };
        let mut granted = state();
        let (returned, control) = alternative(
            page_allocator(true),
            0,
            arguments(number(0), COMMIT, number(READ_WRITE)),
            &mut granted,
        );
        assert_ne!(control, ModelControl::Unsupported);
        let block = object(returned);
        assert_eq!(
            granted.memory().object(block.root()),
            Some((4096, Some(AllocatorFamily(7))))
        );
        assert_eq!(granted.memory().known_run(block, 8, None), vec![0; 8]);
        let byte = |offset| {
            block
                .moved(offset, PointerSize::Bit64)
                .unwrap_or_else(|| process::abort())
        };
        assert_eq!(granted.memory().access_finding(byte(4095), 1), None);
        assert_eq!(
            granted.memory().access_finding(byte(4096), 1),
            Some(FindingKind::PastExtent(ExtentSource::Allocator))
        );

        let mut unfixed = state();
        let (returned, _) = alternative(
            page_allocator(false),
            0,
            arguments(number(0), COMMIT, number(READ_WRITE)),
            &mut unfixed,
        );
        let block = object(returned);
        let past = block
            .moved(4096, PointerSize::Bit64)
            .unwrap_or_else(|| process::abort());
        assert_eq!(unfixed.memory().access_finding(past, 1), None);

        // `mmap` fails with `MAP_FAILED`, minus one, rather than null.
        let mapping = match page_allocator(true) {
            LibraryModel::Allocate(allocation) => LibraryModel::Allocate(Allocation {
                delivery: Delivery::ReturnedOr { failure: -1 },
                ..allocation
            }),
            _ => process::abort(),
        };
        let (returned, _) = alternative(
            mapping,
            1,
            arguments(number(0), COMMIT, number(READ_WRITE)),
            &mut state(),
        );
        assert_eq!(returned, Some(Value::Const(ConstValue::I64(-1))));

        for refused in [
            arguments(stack(-8), COMMIT, number(READ_WRITE)),
            arguments(number(0), 0x2000, number(READ_WRITE)),
            arguments(number(0), COMMIT, number(0x40)),
            arguments(number(0), COMMIT, Value::Unknown),
        ] {
            let (_, control) = alternative(page_allocator(true), 0, refused, &mut state());
            assert_eq!(control, ModelControl::Unsupported);
        }
    }

    /// **A size read through a cell is written back as the size granted.**
    /// `NtAllocateVirtualMemory(-1, &base, 0, &size, MEM_COMMIT, PAGE_READWRITE)`
    /// with `size` 100 delivers a page to `base`, writes 4096 to `size` and
    /// returns `STATUS_SUCCESS`; failing, it returns `STATUS_NO_MEMORY` and
    /// writes neither. The controls: another process, and a base already
    /// naming where to place the block, leave the call unmodeled.
    #[test]
    fn a_size_read_through_a_cell_is_written_back() {
        let prepared = |base: u64| {
            let mut state = state();
            state.memory_mut().store_bytes(at(-8), &base.to_le_bytes());
            state
                .memory_mut()
                .store_bytes(at(-16), &100_u64.to_le_bytes());
            state
        };
        let arguments = |process: Value<MockTarget>| {
            vec![
                process,
                stack(-8),
                number(0),
                stack(-16),
                number(COMMIT),
                number(READ_WRITE),
            ]
        };
        let mut granted = prepared(0);
        let (returned, _) = alternative(
            native_allocator(),
            0,
            arguments(current_process()),
            &mut granted,
        );
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(0))));
        let Loaded::Address(block) = granted.memory().load(at(-8), 8, None) else {
            process::abort()
        };
        assert_eq!(
            granted.memory().object(block.root()),
            Some((4096, Some(AllocatorFamily(8))))
        );
        assert_eq!(
            granted.memory().known_run(at(-16), 8, None),
            4096_u64.to_le_bytes()
        );

        let mut failed = prepared(0);
        let (returned, _) = alternative(
            native_allocator(),
            1,
            arguments(current_process()),
            &mut failed,
        );
        assert_eq!(
            returned,
            Some(Value::Const(ConstValue::I32(0xC000_0017_u32 as i32)))
        );
        assert_eq!(
            failed.memory().known_run(at(-16), 8, None),
            100_u64.to_le_bytes()
        );

        for (mut state, process) in [
            (prepared(0), number(4)),
            (prepared(0x1000), current_process()),
        ] {
            let (_, control) = alternative(native_allocator(), 0, arguments(process), &mut state);
            assert_eq!(control, ModelControl::Unsupported);
        }
    }

    /// **A release through a cell ends the block and reports its size.**
    /// `NtFreeVirtualMemory(-1, &base, &size, MEM_RELEASE)` with `size` zero
    /// ends the page `NtAllocateVirtualMemory` made, writes its size back and
    /// returns `STATUS_SUCCESS`. The controls: `MEM_DECOMMIT`, which keeps
    /// the reservation, and a size other than zero leave it unmodeled, the
    /// block live.
    #[test]
    fn a_release_through_a_cell_ends_the_block_and_reports_its_size() {
        let allocated = |size: u64| {
            let mut state = state();
            state.memory_mut().store_bytes(at(-8), &0_u64.to_le_bytes());
            state
                .memory_mut()
                .store_bytes(at(-16), &100_u64.to_le_bytes());
            let _ = alternative(
                native_allocator(),
                0,
                vec![
                    current_process(),
                    stack(-8),
                    number(0),
                    stack(-16),
                    number(COMMIT),
                    number(READ_WRITE),
                ],
                &mut state,
            );
            state.memory_mut().store_bytes(at(-24), &size.to_le_bytes());
            state
        };
        let block = |state: &State<MockTarget>| match state.memory().load(at(-8), 8, None) {
            Loaded::Address(block) => block,
            _ => process::abort(),
        };
        let release = |kind: u64| vec![current_process(), stack(-8), stack(-24), number(kind)];
        let mut released = allocated(0);
        let made = block(&released);
        let (returned, _) = alternative(native_release(), 0, release(0x8000), &mut released);
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(0))));
        assert_eq!(released.memory().object(made.root()), None);
        assert_eq!(
            released.memory().known_run(at(-24), 8, None),
            4096_u64.to_le_bytes()
        );

        for (mut state, kind) in [(allocated(0), 0x4000), (allocated(4096), 0x8000)] {
            let made = block(&state);
            let (_, control) = alternative(native_release(), 0, release(kind), &mut state);
            assert_eq!(control, ModelControl::Unsupported);
            assert!(state.memory().object(made.root()).is_some());
        }
    }

    /// **A release of an extent must release the whole block.** `munmap(p,
    /// n)` of a block of 5000 bytes, granted as two pages, releases it for
    /// any `n` that rounds up to both pages, and leaves the call unmodeled
    /// for one page, which keeps the second mapped.
    #[test]
    fn a_release_of_an_extent_must_release_the_whole_block() {
        let pages = Some(Pages {
            size: 4096,
            exact: true,
        });
        let munmap = LibraryModel::Free(Deallocation {
            family: AllocatorFamily(7),
            pointer: Operand::Argument(0),
            size: None,
            result: FreeResult::Zero,
            requires: Conditions::NONE,
            whole: Some((Operand::Argument(1), pages)),
            released: None,
            heap: None,
        });
        for (len, whole) in [(5000, true), (8192, true), (4096, false)] {
            let mut state = state();
            let (returned, _) = alternative(
                page_allocator(true),
                0,
                vec![number(0), number(5000), number(COMMIT), number(READ_WRITE)],
                &mut state,
            );
            let block = object(returned);
            let (returned, control) =
                alternative(munmap, 0, vec![Value::Addr(block), number(len)], &mut state);
            assert_eq!(
                state.memory().object(block.root()).is_none(),
                whole,
                "{len}"
            );
            if whole {
                assert_eq!(returned, Some(Value::Const(ConstValue::I32(0))));
            } else {
                assert_eq!(control, ModelControl::Unsupported);
            }
        }
    }

    /// **A routine returning its argument writes nothing.** `LocalLock(h)`
    /// of a fixed block returns `h` and changes no byte; it reads a pointer
    /// and returns one, which it is.
    #[test]
    fn a_routine_returning_its_argument_writes_nothing() {
        let lock = LibraryModel::Identity { argument: 0 };
        let mut state = state();
        state.memory_mut().store_bytes(at(-8), b"held");
        let mut trace = Trace::new();
        let returned = call(lock, vec![stack(-8)], &mut state, &mut trace);
        assert_eq!(returned, Some(stack(-8)));
        assert_eq!(state.memory().known_run(at(-8), 4, None), b"held");
        assert_eq!(lock.arguments(), Some(vec![ModelValue::Pointer]));
        assert_eq!(lock.returns(), Some(ModelValue::Pointer));
        assert_eq!(lock.returned_argument(), Some(0));
    }

    /// The family of heap handles in [`a_block_belongs_to_the_heap_its_handle_names`].
    const HANDLES: AllocatorFamily = AllocatorFamily(20);

    /// The family of heap blocks there.
    const HEAP: AllocatorFamily = AllocatorFamily(6);

    /// `HeapAlloc(heap, flags, size)` in the heap its handle names.
    fn heap_alloc() -> LibraryModel {
        LibraryModel::Allocate(Allocation {
            family: HEAP,
            size: Operand::Argument(2),
            count: None,
            alignment: None,
            zeroed: false,
            flags: None,
            delivery: Delivery::Returned,
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: Some(Operand::Argument(0)),
        })
    }

    /// `HeapFree(heap, flags, block)` in the heap its handle names.
    fn heap_free() -> LibraryModel {
        LibraryModel::Free(Deallocation {
            family: HEAP,
            pointer: Operand::Argument(2),
            size: None,
            result: FreeResult::One,
            requires: Conditions::NONE,
            whole: None,
            released: None,
            heap: Some(Operand::Argument(0)),
        })
    }

    /// **A block belongs to the heap its handle names.** Two heaps
    /// `HeapCreate` made: a block of the first released on the second stays
    /// live, a finding of another allocator's release, and released on its
    /// own it ends; `HeapDestroy` of the first ends every block made in it and
    /// leaves the second's. The control: a block made with a handle the run
    /// cannot name, the process heap's, is released on any heap, there being
    /// nothing to judge.
    #[test]
    fn a_block_belongs_to_the_heap_its_handle_names() {
        let mut state = state();
        let findings = Findings::default();
        state.memory_mut().set_findings(findings.clone());
        state.memory_mut().set_writer(
            0,
            WriteSite {
                body: 0,
                block: 0,
                instruction: 0,
            },
        );
        let create = LibraryModel::CreateHeap { family: HANDLES };
        let (first, _) = alternative(create, 0, Vec::new(), &mut state);
        let (second, _) = alternative(create, 0, Vec::new(), &mut state);
        let (first, second) = (object(first), object(second));
        let allocate = |state: &mut State<MockTarget>, heap: Value<MockTarget>| {
            object(alternative(heap_alloc(), 0, vec![heap, number(0), number(8)], state).0)
        };
        let release = |state: &mut State<MockTarget>, heap: Value<MockTarget>, block: Address| {
            alternative(
                heap_free(),
                0,
                vec![heap, number(0), Value::Addr(block)],
                state,
            )
            .0
        };
        let block = allocate(&mut state, Value::Addr(first));
        release(&mut state, Value::Addr(second), block);
        assert!(
            state.memory().object(block.root()).is_some(),
            "another heap's release"
        );
        assert_eq!(findings.all().len(), 1);
        assert_eq!(
            release(&mut state, Value::Addr(first), block),
            Some(Value::Const(ConstValue::I32(1)))
        );
        assert!(state.memory().object(block.root()).is_none());

        let made_first = allocate(&mut state, Value::Addr(first));
        let made_second = allocate(&mut state, Value::Addr(second));
        let destroy = LibraryModel::DestroyHeap {
            handle: Operand::Argument(0),
            family: HANDLES,
        };
        let (returned, _) = alternative(destroy, 0, vec![Value::Addr(first)], &mut state);
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(1))));
        assert!(state.memory().object(made_first.root()).is_none());
        assert!(
            state.memory().object(first.root()).is_none(),
            "the handle ends too"
        );
        assert!(state.memory().object(made_second.root()).is_some());

        let unnamed = allocate(&mut state, Value::Unknown);
        release(&mut state, Value::Addr(second), unnamed);
        assert!(
            state.memory().object(unnamed.root()).is_none(),
            "the control: a block of a heap the run cannot name"
        );
        assert_eq!(findings.all().len(), 1);
    }

    /// **An allocator hands its object over as its routine does.**
    /// `posix_memalign(&p, 16, n)` writes the object to `p` and returns
    /// zero, and on failure returns `ENOMEM` and writes nothing; an alignment
    /// that is no power of two leaves it unmodeled. `operator new` raises
    /// rather than returning null.
    #[test]
    fn allocators_deliver_and_fail_as_their_routines_do() {
        let posix_memalign = LibraryModel::Allocate(Allocation {
            family: LIBC,
            size: Operand::Argument(2),
            count: None,
            alignment: Some(1),
            zeroed: false,
            flags: None,
            delivery: Delivery::Out {
                out: 0,
                failure: 12,
            },
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        });
        let mut delivered = state();
        let (returned, _) = alternative(
            posix_memalign,
            0,
            vec![stack(-8), number(16), number(32)],
            &mut delivered,
        );
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(0))));
        let Loaded::Address(stored) = delivered.memory().load(at(-8), 8, None) else {
            process::abort()
        };
        assert_eq!(
            delivered.memory().object(stored.root()),
            Some((32, Some(LIBC)))
        );

        let mut failed = state();
        let (returned, _) = alternative(
            posix_memalign,
            1,
            vec![stack(-8), number(16), number(32)],
            &mut failed,
        );
        assert_eq!(returned, Some(Value::Const(ConstValue::I32(12))));
        assert!(failed.memory().known_run(at(-8), 8, None).is_empty());

        let mut misaligned = state();
        let (_, control) = alternative(
            posix_memalign,
            0,
            vec![stack(-8), number(24), number(32)],
            &mut misaligned,
        );
        assert_eq!(control, ModelControl::Unsupported);

        let new = LibraryModel::Allocate(Allocation {
            delivery: Delivery::ReturnedOrRaises,
            ..match malloc(false) {
                LibraryModel::Allocate(allocation) => allocation,
                _ => process::abort(),
            }
        });
        let (_, control) = alternative(new, 1, vec![number(8)], &mut state());
        assert_eq!(control, ModelControl::Trap);
    }

    /// **A reallocation moves the object's bytes and ends it; its failure
    /// leaves it.** `realloc(p, 6)` of a 4-byte object holds its 4 bytes and
    /// 2 unknown ones, and `p` is no object afterwards; its failure returns
    /// null with `p` intact. `realloc(NULL, n)` allocates; a size of zero
    /// leaves the call unmodeled. An old pointer the execution cannot name is
    /// deallocated as `free` deallocates it: one that may be any object ends
    /// none for certain, and the new object's bytes are written but unknown,
    /// since any may come from it; another family's object is a finding and
    /// ends nothing.
    #[test]
    fn a_reallocation_moves_the_bytes_and_ends_the_old_object() {
        let realloc = |family| {
            LibraryModel::Reallocate(Reallocation {
                family,
                pointer: 0,
                size: 1,
                count: None,
                flags: None,
                null_allocates: true,
                heap: None,
            })
        };
        let mut state = state();
        let mut trace = Trace::new();
        let old = object(call(malloc(false), vec![number(4)], &mut state, &mut trace));
        state.memory_mut().store_bytes(old, b"abcd");

        let mut moved = state.clone();
        let (returned, _) = alternative(
            realloc(LIBC),
            0,
            vec![Value::Addr(old), number(6)],
            &mut moved,
        );
        let new = object(returned);
        assert_ne!(new.root(), old.root());
        assert_eq!(moved.memory().known_run(new, 6, None), b"abcd");
        assert_eq!(moved.memory().object(new.root()), Some((6, Some(LIBC))));
        assert_eq!(moved.memory().object(old.root()), None);
        // The two bytes it grew by were never written.
        let Some(grown) = new.moved(4, PointerSize::Bit64) else {
            process::abort()
        };
        assert!(moved.memory().unwritten_bits(grown, 2).is_some());

        let mut failed = state.clone();
        let (returned, _) = alternative(
            realloc(LIBC),
            1,
            vec![Value::Addr(old), number(6)],
            &mut failed,
        );
        assert_eq!(returned, Some(Value::Const(ConstValue::I64(0))));
        assert_eq!(failed.memory().known_run(old, 4, None), b"abcd");

        let mut fresh = state.clone();
        let (returned, _) = alternative(realloc(LIBC), 0, vec![number(0), number(3)], &mut fresh);
        assert_eq!(
            fresh.memory().object(object(returned).root()),
            Some((3, Some(LIBC)))
        );

        let mut refused = state.clone();
        let (_, control) = alternative(
            realloc(LIBC),
            0,
            vec![Value::Addr(old), number(0)],
            &mut refused,
        );
        assert_eq!(control, ModelControl::Unsupported);
        assert_eq!(refused.memory().known_run(old, 4, None), b"abcd");

        let mut foreign = state.clone();
        let (returned, control) = alternative(
            realloc(AllocatorFamily(9)),
            0,
            vec![Value::Addr(old), number(6)],
            &mut foreign,
        );
        assert_eq!(control, ModelControl::Return);
        assert_eq!(foreign.memory().known_run(old, 4, None), b"abcd");
        assert_eq!(
            foreign.memory().object(object(returned).root()),
            Some((6, Some(AllocatorFamily(9))))
        );

        let opaque = Value::Addr(Address::new(
            Root::Opaque {
                activation: 0,
                value: SsaVarId::from_index(3),
                run: 0,
            },
            0,
            PointerSize::Bit64,
        ));
        let mut unnamed = state.clone();
        let (returned, control) =
            alternative(realloc(LIBC), 0, vec![opaque, number(6)], &mut unnamed);
        assert_eq!(control, ModelControl::Return);
        let new = object(returned);
        assert_eq!(unnamed.memory().object(new.root()), Some((6, Some(LIBC))));
        assert_eq!(unnamed.memory().unwritten_bits(new, 6), None);
        assert!(unnamed.memory().known_run(new, 6, None).is_empty());
    }

    /// **A duplicate holds the string and one terminator.** `strdup` copies
    /// the whole string; `strndup(s, 2)` copies two bytes of a longer one and
    /// all of a shorter one; its failure returns null.
    #[test]
    fn a_duplicate_holds_the_string_and_its_terminator() {
        let duplicate = |limit| {
            LibraryModel::Duplicate(Duplication {
                family: LIBC,
                source: 0,
                limit,
            })
        };
        for (model, source, limit, expected) in [
            (
                duplicate(None),
                b"abc\0".as_slice(),
                None,
                b"abc\0".as_slice(),
            ),
            (duplicate(Some(1)), b"abc\0", Some(2), b"ab\0"),
            (duplicate(Some(1)), b"a\0zz", Some(3), b"a\0"),
        ] {
            let mut state = state();
            state.memory_mut().store_bytes(at(-16), source);
            let mut arguments = vec![stack(-16)];
            arguments.extend(limit.map(number));
            let (returned, _) = alternative(model, 0, arguments.clone(), &mut state);
            let copy = object(returned);
            assert_eq!(state.memory().known_run(copy, 8, None), expected);
            let (returned, _) = alternative(model, 1, arguments, &mut state);
            assert_eq!(returned, Some(Value::Const(ConstValue::I64(0))));
        }
    }

    /// Freeing null preserves other objects; freeing a live object ends its
    /// lifetime; a freed object, and another family's, are no object the
    /// deallocator ends: a finding, and the call returns.
    #[test]
    fn free_distinguishes_null_live_and_invalid_pointers() {
        let mut state = state();
        let mut trace = Trace::new();
        let block = object(call(malloc(true), vec![number(4)], &mut state, &mut trace));
        call(
            free(LIBC),
            vec![Value::Const(ConstValue::I64(0))],
            &mut state,
            &mut trace,
        );
        assert_eq!(state.memory().known_run(block, 4, None), [0; 4]);
        let (_, control) = alternative(
            free(AllocatorFamily(2)),
            0,
            vec![Value::Addr(block)],
            &mut state,
        );
        assert_eq!(control, ModelControl::Return);
        assert_eq!(state.memory().known_run(block, 4, None), [0; 4]);
        call(free(LIBC), vec![Value::Addr(block)], &mut state, &mut trace);
        assert!(state.memory().known_run(block, 4, None).is_empty());
        let (_, control) = alternative(free(LIBC), 0, vec![Value::Addr(block)], &mut state);
        assert_eq!(control, ModelControl::Return);
    }

    /// **A deallocator returns as its routine does.** `HeapFree` returns a
    /// `BOOL` one and `LocalFree` a null pointer.
    #[test]
    fn deallocators_return_as_their_routines_do() {
        for (result, expected) in [
            (FreeResult::One, Some(Value::Const(ConstValue::I32(1)))),
            (FreeResult::Null, Some(Value::Const(ConstValue::I64(0)))),
            (FreeResult::Nothing, None),
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            let block = object(call(malloc(false), vec![number(4)], &mut state, &mut trace));
            let model = LibraryModel::Free(Deallocation {
                family: LIBC,
                pointer: Operand::Argument(0),
                size: None,
                result,
                requires: Conditions::NONE,
                whole: None,
                released: None,
                heap: None,
            });
            let (returned, _) = alternative(model, 0, vec![Value::Addr(block)], &mut state);
            assert_eq!(returned, expected);
        }
    }

    /// **A model states its arguments' kinds where it names every position.**
    /// `memset` is a pointer, an `int` and a size; `calloc` two sizes;
    /// `posix_memalign` a pointer and two sizes; `HeapAlloc` skips its heap,
    /// so only a signature states it.
    #[test]
    fn a_model_states_the_arguments_it_names() {
        let calloc = LibraryModel::Allocate(Allocation {
            family: LIBC,
            size: Operand::Argument(1),
            count: Some(0),
            alignment: None,
            zeroed: true,
            flags: None,
            delivery: Delivery::Returned,
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        });
        let posix_memalign = LibraryModel::Allocate(Allocation {
            family: LIBC,
            size: Operand::Argument(2),
            count: None,
            alignment: Some(1),
            zeroed: false,
            flags: None,
            delivery: Delivery::Out {
                out: 0,
                failure: 12,
            },
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        });
        let heap_alloc = LibraryModel::Allocate(Allocation {
            family: LIBC,
            size: Operand::Argument(2),
            count: None,
            alignment: None,
            zeroed: false,
            flags: Some(AllocationFlags {
                argument: 1,
                zeroing: 0x8,
                unmodeled: 0,
            }),
            delivery: Delivery::Returned,
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        });
        let (pointer, size, int) = (ModelValue::Pointer, ModelValue::Size, ModelValue::Int32);
        assert_eq!(
            LibraryModel::Fill.arguments(),
            Some(vec![pointer, int, size])
        );
        assert_eq!(calloc.arguments(), Some(vec![size, size]));
        assert_eq!(posix_memalign.arguments(), Some(vec![pointer, size, size]));
        assert_eq!(posix_memalign.returns(), Some(int));
        assert_eq!(heap_alloc.arguments(), None);
        assert_eq!(LibraryModel::NoWrites.arguments(), None);
        assert_eq!(free(LIBC).returns(), Some(ModelValue::Nothing));
    }

    /// String copies retain exactly one terminator and leave adjacent bytes untouched.
    #[test]
    fn string_models_write_exact_terminated_bytes() {
        for (model, source, before, expected, written) in [
            (
                LibraryModel::CopyString,
                b"abc\0extra".as_slice(),
                b"XXXXXX".as_slice(),
                b"abc\0XX".as_slice(),
                4,
            ),
            (
                LibraryModel::CopyString,
                b"\xff\x80\0tail".as_slice(),
                b"abcdef".as_slice(),
                b"\xff\x80\0def".as_slice(),
                3,
            ),
            (
                LibraryModel::CopyString,
                b"\0hidden".as_slice(),
                b"ABCD".as_slice(),
                b"\0BCD".as_slice(),
                1,
            ),
            (
                LibraryModel::AppendString,
                b"xy\0tail".as_slice(),
                b"ab\0???".as_slice(),
                b"abxy\0?".as_slice(),
                3,
            ),
            (
                LibraryModel::AppendString,
                b"\0hidden".as_slice(),
                b"ab\0???".as_slice(),
                b"ab\0???".as_slice(),
                1,
            ),
            (
                LibraryModel::AppendString,
                b"xy\0tail".as_slice(),
                b"\0?????".as_slice(),
                b"xy\0???".as_slice(),
                3,
            ),
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-128), source);
            state.memory_mut().store_bytes(at(-64), before);
            trace.enter_loop(0, 0);
            let returned = call(model, vec![stack(-64), stack(-128)], &mut state, &mut trace);
            assert_eq!(returned, Some(stack(-64)), "{model:?}");
            assert_eq!(
                state.memory().known_run(at(-64), before.len(), None),
                expected,
                "{model:?}"
            );
            assert_eq!(
                state.memory().known_run(at(-128), source.len(), None),
                source,
                "{model:?}"
            );
            assert_eq!(trace.loops()[0].written.len(), written, "{model:?}");
            assert!(trace.loops()[0].clean, "{model:?}");
            assert!(<LibraryModel as CallModel<MockTarget>>::only_memory(&model));
        }
    }

    /// A normal return is the destination even when the bytes cannot be established.
    #[test]
    fn memory_models_preserve_destination_with_unknown_inputs() {
        let size = Value::Const(ConstValue::I64(6));
        let byte = Value::Const(ConstValue::I64(0x41));
        let cases = [
            (
                "copy source",
                LibraryModel::Copy,
                vec![stack(-64), Value::Unknown, size.clone()],
                false,
            ),
            (
                "copy length",
                LibraryModel::Copy,
                vec![stack(-64), stack(-128), Value::Unknown],
                false,
            ),
            (
                "fill length",
                LibraryModel::Fill,
                vec![stack(-64), byte, Value::Unknown],
                false,
            ),
            (
                "string source pointer",
                LibraryModel::CopyString,
                vec![stack(-64), Value::Unknown],
                false,
            ),
            (
                "append source pointer",
                LibraryModel::AppendString,
                vec![stack(-64), Value::Unknown],
                false,
            ),
            (
                "append prefix",
                LibraryModel::AppendString,
                vec![stack(-64), stack(-128)],
                true,
            ),
            (
                "string source bytes",
                LibraryModel::CopyString,
                vec![stack(-64), stack(-128)],
                false,
            ),
            (
                "append source bytes",
                LibraryModel::AppendString,
                vec![stack(-64), stack(-128)],
                false,
            ),
        ];
        let mut observed = Vec::new();
        for (name, model, arguments, unknown_prefix) in cases {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-64), b"guard\0");
            if unknown_prefix {
                state.memory_mut().store_unknown(at(-64), 1);
                state.memory_mut().store_bytes(at(-128), b"text\0");
            }
            trace.enter_loop(0, 0);
            let returned = call(model, arguments, &mut state, &mut trace);
            assert!(
                state.memory().known_run(at(-64), 6, None).is_empty(),
                "{name}"
            );
            assert!(!trace.loops()[0].clean, "{name}");
            observed.push((name, returned));
        }
        assert_eq!(
            observed,
            [
                ("copy source", Some(stack(-64))),
                ("copy length", Some(stack(-64))),
                ("fill length", Some(stack(-64))),
                ("string source pointer", Some(stack(-64))),
                ("append source pointer", Some(stack(-64))),
                ("append prefix", Some(stack(-64))),
                ("string source bytes", Some(stack(-64))),
                ("append source bytes", Some(stack(-64))),
            ]
        );
    }

    /// Unknown destinations never acquire the source's exact address as a result.
    #[test]
    fn memory_models_with_unknown_destination_keep_only_possible_aliases() {
        for model in [
            LibraryModel::Copy,
            LibraryModel::Fill,
            LibraryModel::CopyString,
            LibraryModel::AppendString,
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-128), b"abc\0");
            let returned = call(
                model,
                vec![
                    Value::Unknown,
                    stack(-128),
                    Value::Const(ConstValue::I64(4)),
                ],
                &mut state,
                &mut trace,
            );
            assert_eq!(returned, Some(Value::StackDerived), "{model:?}");
            assert!(
                state.memory().known_run(at(-128), 4, None).is_empty(),
                "{model:?}"
            );
        }
    }

    /// Reads only the bounded, NUL-terminated payload and leaves source memory intact.
    #[test]
    fn string_reads_preserve_source_and_requested_extent() {
        let context = AnalysisContext::new(MACHINE);
        let mut state = state();
        let mut trace = Trace::new();
        state.memory_mut().store_bytes(at(-64), b"abcd\0tail");
        let mut call = ModelCall::new(vec![], &mut state, &mut trace, Access::new(&context));
        assert_eq!(call.read_string(at(-64), 4), None);
        assert_eq!(call.read_string(at(-64), 5), Some(b"abcd".to_vec()));
        assert_eq!(state.memory().known_run(at(-64), 9, None), b"abcd\0tail");
    }

    /// Length needs no retained bytes, even when entry capacity is zero.
    #[test]
    fn string_length_uses_no_retained_storage() {
        let context = AnalysisContext::new(MACHINE);

        let mut state = state();
        let mut trace = Trace::new();
        state.memory_mut().store_bytes(at(-64), b"abcd\0tail");
        let mut call = ModelCall::new(
            vec![stack(-64)],
            &mut state,
            &mut trace,
            Access::new(&context),
        );
        LibraryModel::Length.call(&mut call);
        assert_eq!(call.take_returned(), Some(Value::Const(ConstValue::I64(4))));
    }

    /// Length counts the terminator's inspection and stops before adjacent bytes.
    #[test]
    fn string_length_boundaries_do_not_retain_or_scan_adjacent_bytes() {
        let context = AnalysisContext::new(MACHINE);
        for (bytes, limit, expected, inspected) in [
            (b"abc\0tail".as_slice(), 4, Some(3), 4),
            (b"abc\0tail".as_slice(), 3, None, 3),
            (b"\0tail".as_slice(), 1, Some(0), 1),
            (b"\0tail".as_slice(), 0, None, 0),
            (b"ab".as_slice(), 64, None, 3),
            (b"ab\0\xff".as_slice(), 64, Some(2), 3),
        ] {
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-64), bytes);
            trace.enter_loop(0, 0);
            let mut call = ModelCall::new(vec![], &mut state, &mut trace, Access::new(&context));
            assert_eq!(call.string_length(at(-64), limit), expected);
            assert_eq!(call.work(), inspected);
            assert_eq!(trace.loops()[0].clean, expected.is_some());
        }
    }

    /// Length scanning finds no length where it cannot know one: in a block
    /// the path never made, or past the end of one, the bytes are unknown,
    /// and a span that wraps the address space is refused.
    #[test]
    fn string_length_withholds_invalid_object_and_address_reads() {
        let context = AnalysisContext::new(MACHINE);
        for (made, wraps) in [(false, false), (true, false), (false, true)] {
            let mut state = state();
            let mut trace = Trace::new();
            let address = if wraps {
                let address = Address::new(Root::Symbol, -1, PointerSize::Bit128);
                state.memory_mut().store_bytes(address, b"a");
                state
                    .memory_mut()
                    .store_bytes(Address::of_root(Root::Symbol), b"\0");
                address
            } else {
                let address = Address::new(Root::Heap(42), 0, PointerSize::Bit64);
                if made {
                    state
                        .memory_mut()
                        .allocate(address.root(), Some(3), None, None);
                    state.memory_mut().store_bytes(address, b"abc");
                }
                address
            };
            let mut call = ModelCall::new(vec![], &mut state, &mut trace, Access::new(&context));
            assert_eq!(call.string_length(address, 4), None, "{made} {wraps}");
            assert_eq!(
                state.memory().refusal(),
                wraps.then_some(MemoryRefusal::AddressRange),
                "{made} {wraps}"
            );
        }
    }

    /// A target-width advance wraps before reading the following NUL.
    #[test]
    fn string_length_preserves_target_address_wrap() {
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
        ] {
            let mut machine = MACHINE;
            machine.ptr_size = width;
            let context = AnalysisContext::new(machine);
            let mut state = state();
            let mut trace = Trace::new();
            let address = Address::new(Root::Symbol, -1, width);
            state.memory_mut().store_bytes(address, b"a");
            state
                .memory_mut()
                .store_bytes(Address::new(Root::Symbol, 0, width), b"\0");
            let mut call = ModelCall::new(vec![], &mut state, &mut trace, Access::new(&context));
            assert_eq!(call.string_length(address, 2), Some(1), "{width:?}");
            assert_eq!(state.memory().refusal(), None);
        }
    }
    /// String payloads obey buffer configuration while length queries retain no payload.
    #[test]
    fn string_payloads_respect_configured_buffer_extent() {
        for maximum in [0, 3, 4, 5] {
            let context = AnalysisContext::new(MACHINE)
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, maximum));
            let mut state = state();
            let mut trace = Trace::new();
            state.memory_mut().store_bytes(at(-64), b"abcd\0tail");
            let mut call = ModelCall::new(vec![], &mut state, &mut trace, Access::new(&context));
            assert_eq!(
                call.read_string(at(-64), 32),
                (maximum >= 4).then(|| b"abcd".to_vec())
            );
            assert_eq!(call.string_length(at(-64), 32), Some(4));
            assert_eq!(state.memory().known_run(at(-64), 9, None), b"abcd\0tail");
        }
    }
}
