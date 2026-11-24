//! Work handed to the pool that keeps its caller's context.
//!
//! A host may keep a context per thread that the work it asks for should run
//! under wherever it runs: an account its allocations charge, say. A pool
//! thread that takes an item of a parallel iterator is not the thread that
//! asked for it, and a thread-local context does not follow the item by
//! itself. A host that keeps one says how to carry it ([`carry_with`]), and
//! every parallel iterator of this crate and its users maps through
//! [`MapCarried`], which captures the asking thread's context once and enters
//! it around each item on the thread that runs the item. A thread that takes
//! an item while it waits inside other work has that work's context marked;
//! the item's own replaces it for the item, and the thread's comes back after.
//!
//! Without a host's carrier, carrying is nothing: the items run as a plain
//! `map` would.

use std::sync::{Arc, OnceLock};

use rayon::iter::IndexedParallelIterator;

/// A host's context, captured on the thread that asked for work, entered on
/// the thread that runs it.
pub trait Context: Send + Sync {
    /// Marks this context on the calling thread and returns what it
    /// replaced, for [`Self::leave`] to restore.
    fn enter(&self) -> usize;

    /// Restores on the calling thread what [`Self::enter`] replaced.
    fn leave(&self, replaced: usize);
}

/// Captures the calling thread's context, if it has one.
pub type Capture = fn() -> Option<Arc<dyn Context>>;

/// The host's capture, once it has said how to carry its context.
static CAPTURE: OnceLock<Capture> = OnceLock::new();

/// Says how to capture the host's context for work handed to the pool.
///
/// The first capture stated stands for the process. Returns whether it is
/// this one.
pub fn carry_with(capture: Capture) -> bool {
    CAPTURE.set(capture).is_ok()
}

/// The asking thread's context, carried to the threads its work runs on.
#[derive(Clone, Default)]
pub struct Carry {
    /// The context, when the host carries one and the thread had it.
    context: Option<Arc<dyn Context>>,
}

impl Carry {
    /// Captures the calling thread's context.
    #[must_use]
    pub fn current() -> Self {
        Self {
            context: CAPTURE.get().and_then(|capture| capture()),
        }
    }

    /// Runs `f` with the carried context entered, and restores the thread's
    /// own after, however `f` ends.
    pub fn within<R>(&self, f: impl FnOnce() -> R) -> R {
        let Some(context) = &self.context else {
            return f();
        };
        let _left = Left {
            context: context.as_ref(),
            replaced: context.enter(),
        };
        f()
    }
}

/// Restores what a [`Carry::within`] replaced when it is dropped, an
/// unwinding item included.
struct Left<'a> {
    /// The context entered.
    context: &'a dyn Context,
    /// What it replaced.
    replaced: usize,
}

impl Drop for Left<'_> {
    fn drop(&mut self) {
        self.context.leave(self.replaced);
    }
}

/// A parallel map whose items run under the asking thread's context
/// ([`Carry`]).
///
/// The first step of every parallel iterator of this crate and its users: a
/// plain `map` runs an item under whatever context the thread that takes it
/// has marked.
pub trait MapCarried: IndexedParallelIterator {
    /// Maps each item through `f` with the asking thread's context entered.
    fn map_carried<R, F>(self, f: F) -> impl IndexedParallelIterator<Item = R>
    where
        R: Send,
        F: Fn(Self::Item) -> R + Sync + Send,
    {
        let carry = Carry::current();
        self.map(move |item| carry.within(|| f(item)))
    }

    /// Maps each item through `f` with state `init` makes for each piece of
    /// work rayon splits off, with the asking thread's context entered.
    fn map_init_carried<S, R, I, F>(self, init: I, f: F) -> impl IndexedParallelIterator<Item = R>
    where
        R: Send,
        I: Fn() -> S + Sync + Send,
        F: Fn(&mut S, Self::Item) -> R + Sync + Send,
    {
        let carry = Carry::current();
        self.map_init(init, move |state, item| carry.within(|| f(state, item)))
    }
}

impl<I: IndexedParallelIterator> MapCarried for I {}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        panic::{self, AssertUnwindSafe},
    };

    use rayon::{ThreadPoolBuilder, iter::IntoParallelIterator, prelude::ParallelIterator};

    use super::*;

    thread_local! {
        /// The context marked on this thread: a number, zero for none.
        static MARKED: Cell<usize> = const { Cell::new(0) };
    }

    /// A context that is a number.
    struct Numbered(usize);

    impl Context for Numbered {
        fn enter(&self) -> usize {
            MARKED.with(|marked| marked.replace(self.0))
        }

        fn leave(&self, replaced: usize) {
            MARKED.with(|marked| marked.set(replaced));
        }
    }

    /// A carry of the context `number`, as a host's capture would make one.
    fn carrying(number: usize) -> Carry {
        Carry {
            context: Some(Arc::new(Numbered(number))),
        }
    }

    /// **An item runs under the context it was asked for under, and the
    /// thread's own comes back**, an unwinding item's too.
    #[test]
    #[allow(clippy::panic, reason = "an item that unwinds is the case under test")]
    fn an_item_runs_under_its_askers_context_and_leaves_the_threads() {
        MARKED.with(|marked| marked.set(7));
        let carry = carrying(3);
        assert_eq!(carry.within(|| MARKED.with(Cell::get)), 3);
        assert_eq!(MARKED.with(Cell::get), 7, "the thread's own is back");
        let unwound = panic::catch_unwind(AssertUnwindSafe(|| {
            carry.within(|| panic!("the item fails"));
        }));
        assert!(unwound.is_err());
        assert_eq!(
            MARKED.with(Cell::get),
            7,
            "an unwinding item restores it too"
        );
        assert_eq!(
            Carry::default().within(|| MARKED.with(Cell::get)),
            7,
            "carrying nothing runs as asked"
        );
    }

    /// **Items a pool's threads take run under the asking thread's context**:
    /// asked for from outside the pool, every item runs on a pool thread,
    /// none of which has it marked.
    #[test]
    fn the_pools_threads_run_items_under_the_askers_context() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap_or_else(|error| unreachable_pool(&error));
        let carry = carrying(5);
        let seen: Vec<usize> = pool.install(|| {
            (0..64)
                .into_par_iter()
                .map(|_| carry.within(|| MARKED.with(Cell::get)))
                .collect()
        });
        assert!(seen.iter().all(|&marked| marked == 5), "{seen:?}");
        let plain: Vec<usize> = pool.install(|| {
            (0..64)
                .into_par_iter()
                .map(|_| MARKED.with(Cell::get))
                .collect()
        });
        assert!(
            plain.iter().all(|&marked| marked == 0),
            "the control: a plain map carries nothing"
        );
    }

    /// Reports a pool that could not be built, as the test's failure.
    #[allow(clippy::panic, reason = "a test with no pool has nothing to measure")]
    fn unreachable_pool(error: &rayon::ThreadPoolBuildError) -> rayon::ThreadPool {
        panic!("a pool of four threads: {error}")
    }
}
