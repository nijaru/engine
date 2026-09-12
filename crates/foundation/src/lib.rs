//! Execution primitives that are intentionally independent of inference policy.
//!
//! This crate is a design-validation surface. It contains model/parameter identity,
//! resource topology, and prepared placement metadata without request, token, KV,
//! autograd, or optimizer semantics.

mod parameter;
mod plan;
mod topology;

pub use parameter::{
    LayoutId, MaterializationPart, ParameterError, ParameterId, ParameterMaterialization,
    ParameterSet, ParameterSpec, ParameterVersion, ScalarType, StorageEncoding, StorageId,
};
pub use plan::{ExecutionPlan, ModelIdentity, PlanError, RuntimeClass, StageId, StagePlacement};
pub use topology::{
    BackendId, ComputeDevice, DeviceId, DeviceLink, NodeId, ResourceTopology, TopologyError,
};
