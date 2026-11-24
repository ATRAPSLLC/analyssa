//! The address each native body returns to: given to it on entry, checked at
//! each of its returns.

use crate::{
    analysis::execution::{
        ReturnAddressStorage, ReturnBinding, ReturnContract,
        access::Access,
        arrivals::Path,
        explore::Explorer,
        facts::Completion,
        findings::{Finding, FindingKind},
        frame::HeldRegister,
        host::CallSite,
        memory::MemoryRefusal,
        value::{Address, Root, Value},
    },
    ir::{CallInputKey, SsaVarId, function::SsaFunction},
    target::Target,
};

/// A call's entry, as the explorer establishes the callee's return address.
pub struct CallControlEntry<'a, T: Target> {
    /// The body entered.
    pub body: &'a SsaFunction<T>,
    /// Its method.
    pub method: &'a T::MethodRef,
    /// The calling instruction.
    pub site: CallSite<'a, T>,
    /// The callee's entry stack pointer.
    pub entry: Address,
    /// The callee's activation.
    pub activation: u32,
    /// Whether the transfer is a tail call, which returns where its caller would.
    pub tail: bool,
}

/// The address a callee returns to, and the entry value of the register
/// that receives it where a register does.
///
/// That register's value on entry is what the call left there, which the
/// caller's inputs bind like any other register's: a call that links through
/// another register leaves the callee's link register as it was. Where the
/// call states nothing of it (no native caller, or one that names no link),
/// it receives the address.
pub struct EnteredReturn<T: Target> {
    /// The address.
    pub expected: Value<T>,
    /// The callee's entry value of its link register.
    pub entry: Option<SsaVarId>,
    /// Where the callee returns through a register, the key its call
    /// states that register under (`Some(None)` for none).
    pub link: Option<Option<CallInputKey>>,
}

impl<T: Target> Explorer<'_, T> {
    /// Creates a formal return address, an identity no SSA definition can
    /// mint: what a root or a caller of no known continuation returns to.
    ///
    /// Consuming its run before execution prevents collision with ordinary
    /// definitions. A return is compared with it; it establishes no numeric
    /// or executable address.
    fn formal_return(&mut self, path: &mut Path<T>, activation: u32) -> Option<Value<T>> {
        let Some(run) = path.state.fresh_run() else {
            self.end(Completion::IdentityExhausted);
            return None;
        };
        Some(Value::Addr(Address::new(
            Root::Opaque {
                activation,
                value: SsaVarId::from_index(0),
                run,
            },
            0,
            self.context.machine().ptr_size,
        )))
    }

    /// The return contract of `body`, ending the run on a refusal.
    fn return_contract(
        &mut self,
        body: &SsaFunction<T>,
        method: Option<&T::MethodRef>,
        block: usize,
    ) -> Option<Option<ReturnContract>> {
        match self.host.body_return(body, method) {
            Ok(contract) => Some(contract),
            Err(reason) => {
                self.end(Completion::Memory { block, reason });
                None
            }
        }
    }

    /// Writes `expected` into the stack cell at `entry`, where a call
    /// instruction leaves its callee the address it returns to.
    fn write_return_cell(
        &mut self,
        path: &mut Path<T>,
        contract: ReturnContract,
        entry: Address,
        expected: &Value<T>,
        block: usize,
    ) -> bool {
        let pointer = self.context.machine().ptr_size;
        if ReturnBinding::for_layout(contract.layout, pointer)
            != ReturnBinding::Word(pointer.bits())
        {
            self.end(Completion::Memory {
                block,
                reason: MemoryRefusal::ReturnControl,
            });
            return false;
        }
        if let Err(exhaustion) = Access::new(self.context)
            .with_write_guard(self.host.write_guard())
            .store_layout(
                path.state.memory_mut(),
                &Value::Addr(entry),
                expected,
                contract.layout,
                None,
            )
        {
            self.end(Completion::Exhausted(exhaustion));
            return false;
        }
        path.trace.record_write(entry, pointer.bytes(), pointer);
        if let Some(reason) = path.state.memory().refusal() {
            self.end(Completion::Memory { block, reason });
            return false;
        }
        true
    }

    /// Gives the root the formal address it returns to, before it runs.
    pub fn initialize_root_control(&mut self, path: &mut Path<T>, body: &SsaFunction<T>) -> bool {
        let method = self.methods.first().cloned().flatten();
        let Some(contract) = self.return_contract(body, method.as_ref(), 0) else {
            return false;
        };
        let Some(contract) = contract else {
            return true;
        };
        let Some(frame) = path.state.frame() else {
            return false;
        };
        let (entry, activation) = (frame.entry_stack_address(), frame.activation());
        let Some(expected) = self.formal_return(path, activation) else {
            return false;
        };
        match contract.storage {
            ReturnAddressStorage::Stack => {
                if !self.write_return_cell(path, contract, entry, &expected, 0) {
                    return false;
                }
            }
            ReturnAddressStorage::Register { entry, .. } => {
                if let Some(entry) = entry {
                    path.state.define(entry, expected.clone());
                }
            }
        }
        if let Some(frame) = path.state.frame_mut() {
            if let ReturnAddressStorage::Register { key: Some(key), .. } = contract.storage {
                let mut registers = frame.registers().clone();
                registers.insert(
                    key,
                    HeldRegister {
                        value: expected.clone(),
                        layout: None,
                    },
                );
                frame.set_registers(registers);
            }
            frame.set_expected_return(Some(expected));
        }
        true
    }

    /// Establishes the address a callee returns to: the call's continuation,
    /// or for a tail call the one its caller returns to. `None` ends the run;
    /// `Some(None)` is a callee with no return to check.
    pub fn enter_control(
        &mut self,
        path: &mut Path<T>,
        call: CallControlEntry<'_, T>,
    ) -> Option<Option<EnteredReturn<T>>> {
        let CallControlEntry {
            body,
            method,
            site,
            entry,
            activation,
            tail,
        } = call;
        let Some(contract) = self.return_contract(body, Some(method), site.block)? else {
            return Some(None);
        };
        let inherited = tail
            .then(|| path.state.frame()?.expected_return().cloned())
            .flatten();
        let expected = match inherited {
            Some(expected) => expected,
            None => match self.host.call_return_address(site) {
                Ok(Some(value)) if value != Value::Unknown => value,
                Ok(_) => self.formal_return(path, activation)?,
                Err(reason) => {
                    self.end(Completion::Memory {
                        block: site.block,
                        reason,
                    });
                    return None;
                }
            },
        };
        let (entry_value, link) = match contract.storage {
            ReturnAddressStorage::Stack => {
                if !self.write_return_cell(path, contract, entry, &expected, site.block) {
                    return None;
                }
                (None, None)
            }
            ReturnAddressStorage::Register { entry, key } => (entry, Some(key)),
        };
        Some(Some(EnteredReturn {
            expected,
            entry: entry_value,
            link,
        }))
    }

    /// Checks a return goes back through the address the body was entered
    /// with: the stack cell still holds it, or the value the return names
    /// is it. A body that overwrote its return address with another value,
    /// or returns through one, cannot resume its caller. Where the run
    /// cannot tell what the cell or value holds, a write it could not place
    /// may or may not have overwritten it: the return goes back through the
    /// address the call left, a finding records it
    /// ([`FindingKind::UnknownReturn`]).
    pub fn validate_return_control(
        &mut self,
        path: &Path<T>,
        body: &SsaFunction<T>,
        target: Option<SsaVarId>,
        block: usize,
    ) -> bool {
        let Some(frame) = path.state.frame() else {
            return false;
        };
        let method = self.methods.get(frame.body()).cloned().flatten();
        let Some(contract) = self.return_contract(body, method.as_ref(), block) else {
            return false;
        };
        let Some(frame) = path.state.frame() else {
            return false;
        };
        // What the return goes back through, and the cell it is read from.
        let (held, access) = match (contract, frame.expected_return()) {
            (None, _) => return true,
            (Some(_), None) => (None, None),
            // The cell is where the call left it: the stack pointer the
            // activation was entered with, which a continuation in other code
            // does not move.
            (Some(contract), Some(_)) => match contract.storage {
                ReturnAddressStorage::Stack => {
                    let cell = Address::new(
                        frame.stack_root(),
                        frame.release_entry_stack(),
                        self.context.machine().ptr_size,
                    );
                    let held = Access::new(self.context)
                        .load_layout::<T>(path.state.memory(), cell, contract.layout)
                        .ok();
                    let bytes = u64::try_from(self.context.machine().ptr_size.bytes()).ok();
                    (held, bytes.map(|bytes| (cell, bytes)))
                }
                ReturnAddressStorage::Register { .. } => {
                    (target.map(|target| frame.value(target)), None)
                }
            },
        };
        // A word the run knows to be some other value is a return the caller
        // cannot resume from: a constant, an address in storage the run
        // names, a cell whose word reads as an integer constant though no
        // address, or the address another activation returns to, which goes
        // back past the activations between. A word the run does not know, or
        // knows only as a value it cannot compute, may still be the address.
        let enclosing = path.state.frames();
        let another_return = held.as_ref().is_some_and(|held| {
            enclosing
                .iter()
                .take(enclosing.len().saturating_sub(1))
                .any(|other| other.expected_return() == Some(held))
        });
        let known_word = another_return
            || match &held {
                Some(Value::Const(_)) => true,
                Some(Value::Addr(address)) => !matches!(address.root(), Root::Opaque { .. }),
                Some(Value::Unknown) => access.is_some_and(|(cell, _)| {
                    let ptr_size = self.context.machine().ptr_size;
                    T::integer_type(ptr_size.bits(), false)
                        .and_then(|integer| T::type_layout(&integer, ptr_size))
                        .and_then(|layout| {
                            Access::new(self.context)
                                .load_layout::<T>(path.state.memory(), cell, layout)
                                .ok()
                        })
                        .is_some_and(|word| word.constant().is_some())
                }),
                _ => false,
            };
        let expected = frame.expected_return();
        let valid = match (&held, expected) {
            (Some(held), Some(expected)) if held == expected => true,
            (Some(_), Some(_)) if !known_word => {
                let instruction = body
                    .block(block)
                    .map_or(0, |ending| ending.instructions().len().saturating_sub(1));
                self.findings.record(Finding {
                    body: frame.body(),
                    block,
                    instruction,
                    kind: FindingKind::UnknownReturn,
                    access,
                    object: None,
                    witness: path.trace.decisions().witness(),
                });
                true
            }
            _ => false,
        };
        if !valid {
            self.end(Completion::Memory {
                block,
                reason: MemoryRefusal::ReturnControl,
            });
        }
        valid
    }
}
