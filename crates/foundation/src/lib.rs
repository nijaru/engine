//! Execution primitives that are intentionally independent of inference policy.
//!
//! Shared byte-pool accounting, readiness notifications, parameter versions and
//! scalar metadata. Physical storage and execution policy remain with their owners;
//! this crate does not model hypothetical placement or materialization APIs.

mod parameter;
mod pool;
mod readiness;

pub use parameter::{ParameterVersion, ScalarType};
pub use pool::{AllocationId, BytePool, PoolLease, ReserveError};
pub use readiness::{Readiness, ReadinessWait};
