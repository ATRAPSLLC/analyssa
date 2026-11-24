//! The tests that measure what a piece of work allocates: one binary of their
//! own, because the allocator that measures it is the process's
//! (`measuring`), installed for every test the binary holds.

mod frame;
mod measuring;

mod largeids;
mod placeholder;
mod vector_bytes;
