//! Execution primitives that are intentionally independent of inference policy.
//!
//! This crate is a design-validation surface. It contains model/parameter identity,
//! resource topology, prepared placement metadata, and shared-pool accounting without
//! request, token, KV, autograd, or optimizer semantics.

mod parameter;
mod plan;
mod pool;
mod topology;

pub use parameter::{
    LayoutId, MaterializationPart, ParameterError, ParameterId, ParameterMaterialization,
    ParameterSet, ParameterSpec, ParameterVersion, ScalarType, StorageEncoding, StorageId,
};
pub use plan::{ExecutionPlan, ModelIdentity, PlanError, RuntimeClass, StageId, StagePlacement};
pub use pool::{AllocationId, BytePool, PoolLease, ReserveError};
pub use topology::{
    BackendId, ComputeDevice, DeviceId, DeviceLink, NodeId, ResourceTopology, TopologyError,
};
