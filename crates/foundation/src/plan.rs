use std::collections::HashSet;
use std::fmt;

use crate::{DeviceId, ParameterVersion, ResourceTopology};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelIdentity(String);

impl ModelIdentity {
    /// # Errors
    /// Returns an error for an empty model identity.
    pub fn new(value: impl Into<String>) -> Result<Self, PlanError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(PlanError::EmptyIdentity("model"));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StageId(u32);

impl StageId {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RuntimeClass(String);

impl RuntimeClass {
    /// # Errors
    /// Returns an error for an empty runtime class.
    pub fn new(value: impl Into<String>) -> Result<Self, PlanError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(PlanError::EmptyIdentity("runtime class"));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePlacement {
    stage: StageId,
    runtime: RuntimeClass,
    devices: Vec<DeviceId>,
}

impl StagePlacement {
    /// # Errors
    /// Returns an error when a stage has no assigned device.
    pub fn new(
        stage: StageId,
        runtime: RuntimeClass,
        devices: Vec<DeviceId>,
    ) -> Result<Self, PlanError> {
        if devices.is_empty() {
            return Err(PlanError::EmptyStagePlacement(stage));
        }
        Ok(Self {
            stage,
            runtime,
            devices,
        })
    }

    #[must_use]
    pub const fn stage(&self) -> StageId {
        self.stage
    }

    #[must_use]
    pub fn runtime(&self) -> &RuntimeClass {
        &self.runtime
    }

    #[must_use]
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlan {
    model: ModelIdentity,
    parameter_version: ParameterVersion,
    stages: Vec<StagePlacement>,
}

impl ExecutionPlan {
    /// # Errors
    /// Returns an error for duplicate stage IDs or devices outside the supplied topology.
    pub fn new(
        model: ModelIdentity,
        parameter_version: ParameterVersion,
        stages: Vec<StagePlacement>,
        topology: &ResourceTopology,
    ) -> Result<Self, PlanError> {
        let mut stage_ids = HashSet::with_capacity(stages.len());
        for stage in &stages {
            if !stage_ids.insert(stage.stage()) {
                return Err(PlanError::DuplicateStage(stage.stage()));
            }
            if stage.devices().iter().any(|device| !topology.contains(device)) {
                return Err(PlanError::UnknownDevice(stage.stage()));
            }
        }
        Ok(Self {
            model,
            parameter_version,
            stages,
        })
    }

    #[must_use]
    pub fn model(&self) -> &ModelIdentity {
        &self.model
    }

    #[must_use]
    pub const fn parameter_version(&self) -> ParameterVersion {
        self.parameter_version
    }

    #[must_use]
    pub fn stages(&self) -> &[StagePlacement] {
        &self.stages
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanError {
    EmptyIdentity(&'static str),
    EmptyStagePlacement(StageId),
    DuplicateStage(StageId),
    UnknownDevice(StageId),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentity(kind) => write!(f, "{kind} identity must not be empty"),
            Self::EmptyStagePlacement(stage) => {
                write!(f, "stage {} has no assigned device", stage.get())
            }
            Self::DuplicateStage(stage) => {
                write!(f, "execution plan contains duplicate stage {}", stage.get())
            }
            Self::UnknownDevice(stage) => {
                write!(f, "stage {} references a device outside the resource topology", stage.get())
            }
        }
    }
}

impl std::error::Error for PlanError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendId, ComputeDevice, NodeId};

    fn device(node: u32, ordinal: u32) -> DeviceId {
        DeviceId::new(
            NodeId::new(node),
            BackendId::new("test").expect("backend"),
            ordinal,
        )
    }

    #[test]
    fn same_model_can_be_placed_locally_or_across_nodes() {
        let first = device(0, 0);
        let second = device(1, 0);
        let local_topology =
            ResourceTopology::new(vec![ComputeDevice::new(first.clone(), None)], vec![])
                .expect("local topology");
        let distributed_topology = ResourceTopology::new(
            vec![
                ComputeDevice::new(first.clone(), None),
                ComputeDevice::new(second.clone(), None),
            ],
            vec![],
        )
        .expect("distributed topology");
        let model = ModelIdentity::new("fixture/model@revision").expect("model");
        let runtime = RuntimeClass::new("fixture-runtime").expect("runtime");
        let local = ExecutionPlan::new(
            model.clone(),
            ParameterVersion::new(7),
            vec![StagePlacement::new(StageId::new(0), runtime.clone(), vec![first.clone()])
                .expect("placement")],
            &local_topology,
        )
        .expect("local plan");
        let distributed = ExecutionPlan::new(
            model,
            ParameterVersion::new(7),
            vec![
                StagePlacement::new(StageId::new(0), runtime.clone(), vec![first])
                    .expect("placement"),
                StagePlacement::new(StageId::new(1), runtime, vec![second])
                    .expect("placement"),
            ],
            &distributed_topology,
        )
        .expect("distributed plan");
        assert_eq!(local.model(), distributed.model());
        assert_eq!(local.parameter_version(), distributed.parameter_version());
        assert_eq!(local.stages().len(), 1);
        assert_eq!(distributed.stages().len(), 2);
    }
}
