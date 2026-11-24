//! analyssa's integration tests, one binary: each concern a module, the
//! helpers they share in `common`.
//!
//! One binary links once; one per file linked every file's copy of the
//! library and its dependencies, which is the build a run waits on.

mod address_taken;
mod arms;
mod bindingtime;
mod boolean_bytes;
mod builder_infrastructure;
mod call_effects;
mod common;
mod control;
mod dataflow;
mod demand;
mod diamond;
mod evaluator;
mod execution;
mod execution_calls;
mod execution_constrained;
mod execution_exceptions;
mod execution_findings;
mod execution_inputs;
mod execution_resources;
mod execution_tail;
mod execution_vector_loads;
mod execution_vectors;
mod executor;
mod facts;
mod float_raise;
mod identity;
mod image;
mod infra;
mod interproc;
mod malformed;
mod memory;
mod memory_opt;
mod mock_target_pipeline;
mod moot;
mod partial_bits;
mod passes;
mod ranges_soundness;
mod rebuild;
mod rejoin;
mod scheduling;
mod serialization;
mod symbolic;
mod unthread;
mod verifier;
