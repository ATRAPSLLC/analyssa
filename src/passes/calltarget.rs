//! Calls through a constant function pointer are direct calls.
//!
//! An indirect call whose pointer folded to a known address - a function
//! pointer loaded from a read-only table, or `__libc_csu_init` calling through
//! a pointer its bounds pinned to one entry - calls exactly one function. Left
//! indirect, nothing that follows a call edge follows it: the callee's
//! signature binds no argument, the call graph has no edge, and the printer
//! writes `(*0x10ce0)(...)` for a call whose callee has a name.
//!
//! The rewrite asks the target which callable an address is
//! ([`Target::method_at`]), so it names the callee the way the host's own lift
//! does; a target that names none leaves the call as it is. A pointer that is a
//! symbol is resolved through [`Target::symbol_address`] first.
//!
//! # Soundness
//!
//! The pointer is a constant, so the call reaches that address on every path,
//! and a call through a pointer to an address is a call to the code there. The
//! arguments, destination and stated effects are kept as they are.

use crate::{
    Result,
    ir::{Callee, ConstValue, SsaFunction, SsaOp},
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Pass that turns calls through a constant function pointer into direct
/// calls.
///
/// Names the callee through [`Target::method_at`].
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - replaces call operations and
/// never changes the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConstantCallTargetPass;

impl ConstantCallTargetPass {
    /// Creates a new constant-call-target pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for ConstantCallTargetPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "constant-call-target"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Turn calls through a constant function pointer into direct calls"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Rewrites every indirect call through a constant pointer into a direct call
    /// to the callable at that address.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` when at least one call was rewritten.
    fn run(&self, ssa: &mut SsaFunction<T>, _env: &PassEnv<'_, T>) -> Result<bool> {
        let constants = ssa.find_constants();
        let mut rewrites: Vec<(usize, usize, SsaOp<T>)> = Vec::new();
        for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
            let Some(call) = instr.op().as_call() else {
                continue;
            };
            let Some(fptr) = call.callee.fptr() else {
                continue;
            };
            let address = match constants.get(&fptr) {
                Some(ConstValue::Symbol(symbol)) => T::symbol_address(symbol),
                Some(value) => value.as_i64(),
                None => None,
            };
            let Some(method) = address
                .and_then(|address| u64::try_from(address).ok())
                .and_then(T::method_at)
            else {
                continue;
            };
            // The call is made as it was, its arguments, result, effects and
            // tail call kept; only what it reaches is named.
            let mut direct = call.clone();
            direct.callee = Callee::Direct(method);
            rewrites.push((block_idx, instr_idx, direct.into()));
        }
        if rewrites.is_empty() {
            return Ok(false);
        }
        let report = ssa.edit(Default::default(), |editor| {
            for (block_idx, instr_idx, op) in rewrites {
                editor.replace_instruction_op(block_idx, instr_idx, op)?;
            }
            Ok(())
        });
        Ok(report.is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::NullListener,
        ir::{CallData, SsaDefSpec, SsaFunctionBuilder, SsaVarId},
        testing::{MACHINE, MockTarget, MockType, mock_i32, mock_ptr, run_mock_pass_boundary},
    };

    /// A function calling through `pointer` and returning the result.
    fn calling(constant: bool) -> SsaFunction<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        builder
            .in_block(0, |block| {
                let pointer: SsaVarId = if constant {
                    block.const_i64(SsaDefSpec::tmp(MockType::Ptr), 0x1000)?
                } else {
                    block.load_arg(mock_ptr(), 0)?
                };
                let result = block.emit_def(mock_i32(), |dest| {
                    SsaOp::from(CallData::new(
                        Some(dest),
                        Callee::Indirect {
                            fptr: pointer,
                            signature: 0,
                        },
                        Vec::new(),
                    ))
                })?;
                block.ret(Some(result))?;
                Ok(())
            })
            .expect("build block");
        builder.finish().expect("finish")
    }

    /// The calls `ssa` makes: `Some(method)` for a direct one, `None` for one
    /// through a pointer.
    fn calls(ssa: &SsaFunction<MockTarget>) -> Vec<Option<u32>> {
        ssa.iter_instructions()
            .filter_map(|(_, _, instr)| match instr.op() {
                SsaOp::Call(call) => Some(call.callee.direct().copied()),
                _ => None,
            })
            .collect()
    }

    /// A call through a pointer that holds a known address calls the callable
    /// there. The control is the same call through a pointer the caller
    /// supplies, which stays indirect.
    #[test]
    fn a_call_through_a_constant_pointer_is_a_direct_call() {
        let mut constant = calling(true);
        assert!(run_mock_pass_boundary(&mut constant, "constant", |ssa| {
            ConstantCallTargetPass.run(
                ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        }));
        assert_eq!(calls(&constant), vec![Some(0x1000)]);

        let mut supplied = calling(false);
        assert!(!run_mock_pass_boundary(&mut supplied, "supplied", |ssa| {
            ConstantCallTargetPass.run(
                ssa,
                &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        }));
        assert_eq!(calls(&supplied), vec![None]);
    }
}
