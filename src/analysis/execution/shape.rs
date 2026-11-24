//! The structure of a function the explorer needs: the order to advance
//! paths in, its loops, and where a raise in each block may dispatch.

use crate::{
    analysis::{
        cfg::SsaCfg,
        execution::thrown::Thrown,
        loops::{LoopForest, detect_loops},
    },
    graph::{
        NodeId, RootedGraph,
        algorithms::{compute_dominators, reverse_postorder},
    },
    ir::{HandlerKind, function::SsaFunction},
    target::Target,
};

/// The order and the loops of one function, and where a raise in each block
/// may dispatch.
#[derive(Debug)]
pub struct Shape {
    /// Each block's position in reverse postorder; unreachable blocks last.
    rank: Vec<usize>,
    /// The natural loops.
    loops: LoopForest,
    /// Where a raise in each block may dispatch; empty where no clause
    /// protects the block.
    dispatch: Vec<Dispatch>,
}

/// Where a raise in one protected block may dispatch, in its own function.
///
/// A handler that may decline the raise (a typed catch, a filter's verdict)
/// lets it go on outward, so a raise reaches every entry here and the
/// function's caller too.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dispatch {
    /// Each block the raise may enter, with whether it is where the function
    /// resumes once a handler that runs elsewhere is done with the raise
    /// ([`Target::handler_resumes_protected`]).
    entries: Vec<(usize, bool)>,
    /// Whether a clause protecting the block runs its handler elsewhere and
    /// names no place the function resumes at: the raise may resume this
    /// function where the execution cannot say.
    unresolved: bool,
    /// The covering clauses whose runtime hands their entry the clause it
    /// chose ([`SsaExceptionHandler::selector`]), in the order the runtime
    /// tries them: an Itanium landing pad's catches and cleanup.
    ///
    /// [`SsaExceptionHandler::selector`]: crate::ir::SsaExceptionHandler::selector
    clauses: Vec<Clause>,
}

/// A clause whose entry the runtime hands the clause it chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Clause {
    /// The block it enters.
    entry: usize,
    /// What it takes: the module address of the type a catch names, `None`
    /// for `catch (...)`; a cleanup takes every exception and catches none.
    takes: Takes,
    /// The value its entry receives.
    selector: i64,
}

/// What an exception a clause takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Takes {
    /// A catch of the type at a module address, or of every type (`None`).
    Catch(Option<u64>),
    /// A cleanup: run on every exception that passes, which resumes unwinding.
    Cleanup,
}

/// An entry the personality enters a raise at, with the selector it hands
/// it ([`Dispatch::chosen`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chosen {
    /// The block entered.
    pub entry: usize,
    /// The selector the entry receives.
    pub selector: i64,
}

impl Dispatch {
    /// Where a raise in `block` may dispatch: the handler entry of every
    /// clause whose protected blocks hold it (an unstated protected range
    /// holds every block), never a filter's, whose expression returns its
    /// verdict to the runtime rather than to this function.
    fn of<T: Target>(ssa: &SsaFunction<T>, block: usize) -> Self {
        let mut dispatch = Self::default();
        let covering = ssa.exception_handlers().iter().filter(|clause| {
            clause
                .protected_range
                .is_none_or(|range| range.contains(block))
        });
        let mut elsewhere = false;
        for clause in covering {
            match clause.handler_range {
                Some(handler) => {
                    let entry = (handler.start(), T::handler_resumes_protected(&clause.flags));
                    if !dispatch.entries.contains(&entry) {
                        dispatch.entries.push(entry);
                    }
                    if let Some(selector) = clause.selector {
                        // A chosen clause's caught type is its module
                        // address, zero naming every type.
                        let takes = match clause.kind() {
                            HandlerKind::Catch => Takes::Catch(
                                (clause.class_token_or_filter != 0)
                                    .then_some(u64::from(clause.class_token_or_filter)),
                            ),
                            _ => Takes::Cleanup,
                        };
                        dispatch.clauses.push(Clause {
                            entry: handler.start(),
                            takes,
                            selector,
                        });
                    }
                }
                None => elsewhere = true,
            }
        }
        dispatch.unresolved = elsewhere && !dispatch.entries.iter().any(|(_, resumes)| *resumes);
        dispatch
    }

    /// Whether no clause protects the block.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && !self.unresolved
    }

    /// Each block the raise may enter, with whether it resumes after a
    /// handler that ran elsewhere.
    #[must_use]
    pub fn entries(&self) -> &[(usize, bool)] {
        &self.entries
    }

    /// Whether the raise may resume the function where the execution cannot
    /// say.
    #[must_use]
    pub const fn is_unresolved(&self) -> bool {
        self.unresolved
    }

    /// Where an Itanium personality enters `thrown`'s raise in this block,
    /// and whether the raise goes on outward; `None` where no covering clause
    /// hands its entry a choice, so the raise enters [`Self::entries`].
    ///
    /// The clauses are tried in their order. One that takes the exception for
    /// certain (a catch-all, a catch of the very type thrown) is entered with
    /// its selector and ends the search: the raise is caught, so it goes no
    /// further. One that may take it (a catch of another type, which takes a
    /// derived one) is entered with its selector, and the search goes on, as
    /// the raise may pass it. A cleanup is entered with its selector, zero,
    /// where no catch took the exception for certain, and the raise goes on
    /// from the cleanup when it resumes unwinding, not from here.
    #[must_use]
    pub fn chosen<T: Target>(&self, thrown: &Thrown<T>) -> Option<(Vec<Chosen>, bool)> {
        if self.clauses.is_empty() {
            return None;
        }
        let mut chosen: Vec<Chosen> = Vec::new();
        let mut cleanup = None;
        let mut caught = false;
        for clause in &self.clauses {
            let pick = Chosen {
                entry: clause.entry,
                selector: clause.selector,
            };
            match clause.takes {
                Takes::Catch(takes) => {
                    if !chosen.contains(&pick) {
                        chosen.push(pick);
                    }
                    if thrown.caught_by(takes) == Some(true) {
                        caught = true;
                        break;
                    }
                }
                Takes::Cleanup => {
                    cleanup.get_or_insert(pick);
                }
            }
        }
        if !caught && let Some(cleanup) = cleanup {
            chosen.push(cleanup);
        }
        // A raise a catch takes for certain, or a cleanup resumes, goes no
        // further from here.
        let onward = !caught && cleanup.is_none();
        Some((chosen, onward))
    }
}

impl Shape {
    /// Computes a body's traversal rank and natural loops.
    pub fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let cfg = SsaCfg::from_ssa(ssa);
        let order = reverse_postorder(&cfg, cfg.entry());
        let mut rank = vec![usize::MAX; ssa.block_count()];
        for (position, node) in order.iter().enumerate() {
            if let Some(slot) = rank.get_mut(usize::from(*node)) {
                *slot = position;
            }
        }
        let dominators = compute_dominators(&cfg, cfg.entry());
        let loops = detect_loops(&cfg, &dominators);
        let dispatch = if ssa.has_exception_handlers() {
            (0..ssa.block_count())
                .map(|block| Dispatch::of(ssa, block))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            rank,
            loops,
            dispatch,
        }
    }

    /// Where a raise in `block` may dispatch; `None` where no clause
    /// protects it.
    pub fn dispatch(&self, block: usize) -> Option<&Dispatch> {
        self.dispatch
            .get(block)
            .filter(|dispatch| !dispatch.is_empty())
    }

    /// A block's position in reverse postorder.
    pub fn rank(&self, block: usize) -> usize {
        self.rank.get(block).copied().unwrap_or(usize::MAX)
    }

    /// Whether `block` lies in the loop headed by `header`.
    pub fn in_loop(&self, header: usize, block: usize) -> bool {
        self.loops
            .loop_for_header(NodeId::new(header))
            .is_some_and(|info| info.contains(NodeId::new(block)))
    }

    /// Whether `block` heads a loop.
    pub fn is_header(&self, block: usize) -> bool {
        self.loops.loop_for_header(NodeId::new(block)).is_some()
    }

    /// Every block of the loop headed by `header`.
    #[cfg(test)]
    pub fn loop_blocks(&self, header: usize) -> Vec<usize> {
        (0..self.rank.len())
            .filter(|block| self.in_loop(header, *block))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        ir::function::{SsaDefSpec, SsaFunctionBuilder},
        testing::{MockTarget, MockType},
    };

    fn abort<E, R>(_: E) -> R {
        process::abort()
    }

    /// Entry, a self loop, and an exit: the loop is found and the blocks are
    /// ranked in the order control reaches them.
    #[test]
    fn a_self_loop_is_ranked_and_found() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.in_block(0, |b| b.jump(1)).unwrap_or_else(abort);
        builder
            .in_block(1, |b| {
                let flag = b.const_i32(SsaDefSpec::tmp(MockType::I32), 1)?;
                b.branch(flag, 1, 2)
            })
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let shape = Shape::of(&ssa);
        assert!(shape.rank(0) < shape.rank(1) && shape.rank(1) < shape.rank(2));
        assert!(shape.is_header(1));
        assert!(!shape.is_header(2));
        assert!(shape.in_loop(1, 1) && !shape.in_loop(1, 2));
        assert_eq!(shape.loop_blocks(1), vec![1]);
    }
}
