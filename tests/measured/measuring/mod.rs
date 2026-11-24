//! Measures the largest single allocation a piece of work makes on its thread.
//!
//! A table sized by an id that malformed IR names is at least half a gigabyte
//! for the ids that matter; [`LIMIT`] is a few thousand times smaller than that
//! and a few hundred times larger than anything the fixtures need. Declaring the
//! allocator here installs it in every test binary that includes this module.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

/// The largest single allocation a measured closure may make.
pub const LIMIT: usize = 1 << 20;

thread_local! {
    /// The largest allocation this thread has requested since the last reset.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

/// The system allocator, recording the largest request each thread makes.
struct Measuring;

impl Measuring {
    /// Records one request of `size` bytes.
    fn note(size: usize) {
        LARGEST.with(|largest| largest.set(largest.get().max(size)));
    }
}

// SAFETY: every method forwards to `System` with its arguments unchanged; the
// recording only touches a thread-local counter.
unsafe impl GlobalAlloc for Measuring {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::note(layout.size());
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract for `layout`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::note(layout.size());
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are the pair `GlobalAlloc::dealloc` requires.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::note(new_size);
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Measuring = Measuring;

/// Runs `work` and returns its result with the largest allocation it made.
pub fn measured<R>(work: impl FnOnce() -> R) -> (R, usize) {
    LARGEST.with(|largest| largest.set(0));
    let result = work();
    (result, LARGEST.with(Cell::get))
}

/// Asserts that `largest` is within [`LIMIT`].
pub fn assert_bounded(what: &str, largest: usize) {
    assert_within(what, largest, LIMIT);
}

/// Asserts that `largest` is within `limit`.
pub fn assert_within(what: &str, largest: usize, limit: usize) {
    assert!(
        largest <= limit,
        "{what}: allocated {largest} bytes at once, limit {limit}"
    );
}
