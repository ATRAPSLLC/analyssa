//! Static execution: the executor, its host contract and the residual rewrite,
//! over constructed SSA. One binary, one module per concern.

mod branch_continuations;
mod call_arguments;
mod call_resolution_inputs;
mod call_results;
mod call_targets;
mod call_use_bounds;
mod call_use_nested;
mod code_limits;
mod code_stability;
mod division_scalar_exceptions;
mod effect_admission;
mod execution_limits;
mod execution_model_bounds;
mod execution_return_bindings;
mod faults;
mod machine_models;
mod machine_state;
mod nested_scopes;
mod observation_payload_bounds;
mod owned_execution_bodies;
mod relocated_cells;
mod residual_arithmetic_visits;
mod residual_bounds;
mod residual_exit_phis;
mod return_bindings;
mod return_control;
mod scopes;
mod source_writes;
mod stack_congruence;
mod termination;
mod vector_writes;
