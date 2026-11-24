//! Values a branch decides equal are one value past the edge it decides so on.
//!
//! A `BranchCmp` of two values for equality takes one of its edges only where
//! they are equal: past that edge, in every block it alone leads into, the two
//! are one value, and a use of the one defined later may read the one defined
//! earlier. A frame too large to probe a page at a time in line is allocated by
//! a loop stepping the stack pointer down until it equals a bound formed
//! before the loop (`lea r11, [rsp - 0x10000]` ... `cmp rsp, r11; jne`): past
//! the loop the stack pointer is the bound, an address of the frame, where
//! left as the loop's own value every address past it was the stack pointer's.
//!
//! # Scope
//!
//! Integers and pointers of one width only, whose equality is of their bits; a
//! float's is not (`-0.0 == 0.0`). Two values, not a value and a constant,
//! which the constant passes answer for. The edge's target must have the
//! branch as its one predecessor, so every block it dominates is reached only
//! through the edge. The value defined earlier is the one kept: its
//! definition dominates the later one's, so it is available wherever the
//! later one is used.

use crate::{
    Result,
    analysis::cfg::SsaCfg,
    bitset::BitSet,
    graph::{NodeId, Predecessors, algorithms::compute_dominators},
    ir::{
        function::{SsaEditOptions, SsaFunction},
        ops::{SsaOp, kinds::CmpKind},
        variable::SsaVarId,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Pass that makes two values a branch decides equal one value past the edge
/// it decides so on.
///
/// Past an edge a branch
/// takes only where two integers or pointers are equal, the uses of the one
/// defined later read the one defined earlier.
///
/// # Modification scope
///
/// [`ModificationScope::UsesOnly`] - rewrites operands and never changes an
/// operation or the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct EqualityPropagationPass;

impl EqualityPropagationPass {
    /// Creates a new equality-propagation pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for EqualityPropagationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "equality-propagation"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Make values a branch decides equal one value past its edge"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Rewrites, past every edge a branch takes only where two values are equal,
    /// the uses of the one defined later as the one defined earlier.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `_env` - The pass environment, which this pass does not read.
    ///
    /// # Returns
    ///
    /// `true` when at least one use was rewritten.
    fn run(&self, ssa: &mut SsaFunction<T>, _env: &PassEnv<'_, T>) -> Result<bool> {
        let cfg = SsaCfg::from_ssa(ssa);
        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let constants = ssa.find_constants();
        // Whether `value` is an integer or a pointer, and its width: a pointer's
        // is the target's, whatever its type states.
        let shape = |value: SsaVarId| {
            ssa.variable(value).and_then(|variable| {
                let ty = variable.var_type();
                T::float_format(ty)
                    .is_none()
                    .then(|| (T::is_pointer(ty), T::bit_width(ty)))
                    .filter(|(pointer, width)| *pointer || width.is_some())
            })
        };
        // Where `value` is defined: its block, and its place in it, an entry
        // value or a phi before every instruction.
        let defined = |value: SsaVarId| {
            ssa.variable(value).map(|variable| {
                let site = variable.def_site();
                (
                    site.block(),
                    site.index().map_or(0, |index| index.saturating_add(1)),
                )
            })
        };
        // The earlier of two definitions, where one dominates the other.
        let earlier = |first: SsaVarId, second: SsaVarId| -> Option<(SsaVarId, SsaVarId)> {
            let (first_block, first_at) = defined(first)?;
            let (second_block, second_at) = defined(second)?;
            if first_block == second_block {
                return match first_at.cmp(&second_at) {
                    std::cmp::Ordering::Less => Some((first, second)),
                    std::cmp::Ordering::Greater => Some((second, first)),
                    std::cmp::Ordering::Equal => None,
                };
            }
            if dominators.dominates(NodeId::new(first_block), NodeId::new(second_block)) {
                Some((first, second))
            } else if dominators.dominates(NodeId::new(second_block), NodeId::new(first_block)) {
                Some((second, first))
            } else {
                None
            }
        };
        // Each later value, the earlier it is equal to, and the blocks past the
        // edge.
        let mut rewrites: Vec<(SsaVarId, SsaVarId, BitSet)> = Vec::new();
        for (block_idx, block) in ssa.blocks().iter().enumerate() {
            let Some(SsaOp::BranchCmp {
                left,
                right,
                cmp,
                true_target,
                false_target,
                ..
            }) = block.control_terminator()
            else {
                continue;
            };
            let equal = match cmp {
                CmpKind::Eq => *true_target,
                CmpKind::Ne => *false_target,
                _ => continue,
            };
            if true_target == false_target
                || left == right
                || constants.contains_key(left)
                || constants.contains_key(right)
                || shape(*left).is_none()
                || shape(*left) != shape(*right)
            {
                continue;
            }
            let mut predecessors = cfg.predecessors(NodeId::new(equal));
            if predecessors.next().map(NodeId::index) != Some(block_idx)
                || predecessors.next().is_some()
            {
                continue;
            }
            let Some((kept, replaced)) = earlier(*left, *right) else {
                continue;
            };
            let mut past = BitSet::new(ssa.block_count());
            for other in 0..ssa.block_count() {
                if dominators.dominates(NodeId::new(equal), NodeId::new(other)) {
                    past.insert(other);
                }
            }
            rewrites.push((replaced, kept, past));
        }
        if rewrites.is_empty() {
            return Ok(false);
        }
        let mut rewritten = false;
        let result = ssa.edit(SsaEditOptions::new(), |editor| {
            for (replaced, kept, past) in &rewrites {
                let result = editor.replace_uses_within(*replaced, *kept, past, Some(&dominators));
                rewritten |= result.replaced > 0;
            }
            Ok(())
        });
        Ok(rewritten && result.is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::NullListener,
        ir::{SsaDefSpec, SsaFunctionBuilder},
        testing::{MACHINE, MockTarget, MockType, run_mock_pass_boundary},
    };

    /// A loop stepping `cursor` down by 16 from `start` until it is `bound`,
    /// `cmp` deciding the way out, then returning `cursor + 8`; with
    /// `joined`, a second way into the exit block. `ty` is the values' type.
    /// Blocks: 0 the entry, 1 the loop, 2 the exit, 3 the second way in.
    fn stepping(cmp: CmpKind, joined: bool, ty: MockType) -> (SsaFunction<MockTarget>, SsaVarId) {
        let def = || SsaDefSpec::<MockTarget>::tmp(ty);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
        let (start, bound) = builder
            .in_block(0, |block| {
                let start = block.load_arg(def(), 0)?;
                let bound = block.load_arg(def(), 1)?;
                let other = block.load_arg(SsaDefSpec::tmp(MockType::I32), 2)?;
                let zero = block.const_i32(SsaDefSpec::tmp(MockType::I32), 0)?;
                if joined {
                    block.branch_cmp(other, zero, CmpKind::Eq, false, 3, 1)?;
                } else {
                    block.jump(1)?;
                }
                Ok((start, bound))
            })
            .expect("entry");
        let cursor = builder
            .in_block(1, |block| block.empty_phi(def()))
            .expect("phi");
        let stepped = builder
            .in_block(1, |block| {
                let sixteen = block.const_i64(def(), 16)?;
                let stepped = block.sub(def(), cursor, sixteen)?;
                let (into_loop, out) = match cmp {
                    CmpKind::Ne => (1, 2),
                    _ => (2, 1),
                };
                block.branch_cmp(stepped, bound, cmp, false, into_loop, out)?;
                Ok(stepped)
            })
            .expect("loop");
        builder
            .add_phi_operand(1, cursor, 0, start)
            .expect("entry operand");
        builder
            .add_phi_operand(1, cursor, 1, stepped)
            .expect("loop operand");
        // The exit reads the cursor the loop left, joined with `start` where a
        // second way reaches it.
        let leaving = if joined {
            let at = builder
                .in_block(2, |block| block.empty_phi(def()))
                .expect("exit phi");
            builder
                .add_phi_operand(2, at, 1, stepped)
                .expect("loop's way out");
            builder
                .add_phi_operand(2, at, 3, start)
                .expect("second way in");
            at
        } else {
            stepped
        };
        builder
            .in_block(2, |block| {
                let eight = block.const_i64(def(), 8)?;
                let past = block.add(def(), leaving, eight)?;
                block.ret(Some(past))?;
                Ok(())
            })
            .expect("exit");
        if joined {
            builder
                .in_block(3, |block| block.jump(2))
                .expect("second way in");
        }
        (builder.finish().expect("finish"), bound)
    }

    /// What the exit block's `cursor + 8` reads.
    fn read_past(ssa: &SsaFunction<MockTarget>) -> Option<SsaVarId> {
        ssa.block(2)?
            .instructions()
            .iter()
            .find_map(|row| match row.op() {
                SsaOp::Add { left, .. } => Some(*left),
                _ => None,
            })
    }

    /// **Past the way out of a loop that runs until two values are equal,
    /// the later is the earlier.** The cursor a loop steps down leaves where
    /// it equals the bound, and the address past the loop is the bound's.
    /// The controls keep the cursor: a loop leaving where the cursor is
    /// below the bound says nothing of its value; an exit block another way
    /// reaches may be entered with any cursor; and floats equal in value are
    /// not one value (`-0.0 == 0.0`).
    #[test]
    fn past_an_equality_the_later_value_is_the_earlier() {
        let rewrites = |cmp, joined, ty| {
            let (mut ssa, bound) = stepping(cmp, joined, ty);
            let changed = run_mock_pass_boundary(&mut ssa, "equalities", |ssa| {
                EqualityPropagationPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            (changed, read_past(&ssa) == Some(bound))
        };
        assert_eq!(
            rewrites(CmpKind::Ne, false, MockType::Ptr),
            (true, true),
            "the bound"
        );
        assert_eq!(
            rewrites(CmpKind::Ne, false, MockType::I64),
            (true, true),
            "an integer"
        );
        assert_eq!(
            rewrites(CmpKind::Lt, false, MockType::Ptr),
            (false, false),
            "below it"
        );
        assert_eq!(
            rewrites(CmpKind::Ne, true, MockType::Ptr),
            (false, false),
            "joined"
        );
        assert_eq!(
            rewrites(CmpKind::Ne, false, MockType::F64),
            (false, false),
            "floats"
        );
    }
}
