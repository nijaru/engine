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
            if stage
                .devices()
                .iter()
                .any(|device| !topology.contains(device))
            {
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
                write!(
                    f,
                    "stage {} references a device outside the resource topology",
                    stage.get()
                )
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

    fn topology(devices: &[DeviceId]) -> ResourceTopology {
        ResourceTopology::new(
            devices
                .iter()
                .map(|device| ComputeDevice::new(device.clone(), None))
                .collect(),
            vec![],
        )
        .expect("topology")
    }

    fn model() -> ModelIdentity {
        ModelIdentity::new("fixture/model@revision").expect("model")
    }

    fn runtime(name: &str) -> RuntimeClass {
        RuntimeClass::new(name).expect("runtime class")
    }

    /// The logical decomposition (which stages exist, with which runtimes) and the
    /// placement (which devices run them) are separate decisions. Holding the
    /// first fixed and moving only the second must produce the same logical plan.
    ///
    /// The earlier version of this test changed both at once - one stage on one
    /// device versus two stages on two devices - so any difference it observed
    /// could have come from either.
    #[test]
    fn placement_moves_without_changing_the_logical_stages() {
        let first = device(0, 0);
        let second = device(0, 1);
        let topology = topology(&[first.clone(), second.clone()]);

        // One logical decomposition, reused for every placement below.
        let stages = |decoder_devices: Vec<DeviceId>| {
            vec![
                StagePlacement::new(
                    StageId::new(0),
                    runtime("fixture-embedding"),
                    vec![first.clone()],
                )
                .expect("placement"),
                StagePlacement::new(StageId::new(1), runtime("fixture-decoder"), decoder_devices)
                    .expect("placement"),
            ]
        };
        let logical = |plan: &ExecutionPlan| {
            plan.stages()
                .iter()
                .map(|stage| (stage.stage(), stage.runtime().clone()))
                .collect::<Vec<_>>()
        };

        let co_located = ExecutionPlan::new(
            model(),
            ParameterVersion::new(7),
            stages(vec![first.clone()]),
            &topology,
        )
        .expect("co-located plan");
        let spread = ExecutionPlan::new(
            model(),
            ParameterVersion::new(7),
            stages(vec![second.clone()]),
            &topology,
        )
        .expect("spread plan");

        assert_eq!(co_located.model(), spread.model());
        assert_eq!(co_located.parameter_version(), spread.parameter_version());
        assert_eq!(
            logical(&co_located),
            logical(&spread),
            "only placement was supposed to change"
        );
        assert_eq!(co_located.stages()[0].devices(), &[first]);
        assert_eq!(spread.stages()[1].devices(), &[second]);
        assert_ne!(
            co_located.stages()[1].devices(),
            spread.stages()[1].devices()
        );
    }

    /// The mirror property: with a fixed set of available devices, the same model
    /// may be decomposed into one stage or into several, and both remain valid
    /// plans over the same resources.
    #[test]
    fn stage_decomposition_is_independent_of_available_devices() {
        let first = device(0, 0);
        let second = device(1, 0);
        let topology = topology(&[first.clone(), second.clone()]);

        let single = ExecutionPlan::new(
            model(),
            ParameterVersion::new(7),
            vec![
                StagePlacement::new(
                    StageId::new(0),
                    runtime("fixture-runtime"),
                    vec![first.clone()],
                )
                .expect("placement"),
            ],
            &topology,
        )
        .expect("single-stage plan");
        let staged = ExecutionPlan::new(
            model(),
            ParameterVersion::new(7),
            vec![
                StagePlacement::new(StageId::new(0), runtime("fixture-encoder"), vec![first])
                    .expect("placement"),
                StagePlacement::new(StageId::new(1), runtime("fixture-decoder"), vec![second])
                    .expect("placement"),
            ],
            &topology,
        )
        .expect("two-stage plan");

        assert_eq!(single.model(), staged.model());
        assert_eq!(single.parameter_version(), staged.parameter_version());
        assert_eq!(single.stages().len(), 1);
        assert_eq!(staged.stages().len(), 2);
        assert_eq!(single.stages()[0].stage(), staged.stages()[0].stage());
        assert_eq!(
            single.stages()[0].devices(),
            staged.stages()[0].devices(),
            "the first stage kept its placement"
        );
    }

    /// A stage's device list is an ordered placement, not a set: replication
    /// across two devices on one node and across nodes must both round-trip.
    #[test]
    fn one_stage_records_replication_across_devices_and_nodes() {
        let local_first = device(0, 0);
        let local_second = device(0, 1);
        let remote = device(1, 0);
        let topology = topology(&[local_first.clone(), local_second.clone(), remote.clone()]);

        let plan = ExecutionPlan::new(
            model(),
            ParameterVersion::new(1),
            vec![
                StagePlacement::new(
                    StageId::new(0),
                    runtime("fixture-runtime"),
                    vec![local_first.clone(), local_second.clone(), remote.clone()],
                )
                .expect("placement"),
            ],
            &topology,
        )
        .expect("replicated plan");
        assert_eq!(
            plan.stages()[0].devices(),
            &[local_first, local_second, remote]
        );
    }

    /// The two rejections a plan performs, which nothing exercised before: a stage
    /// cannot appear twice, and it cannot name a device the topology does not have.
    #[test]
    fn plans_reject_duplicate_stages_and_absent_devices() {
        let present = device(0, 0);
        let absent = device(1, 0);
        let topology = topology(std::slice::from_ref(&present));

        assert_eq!(
            StagePlacement::new(StageId::new(0), runtime("fixture-runtime"), vec![]),
            Err(PlanError::EmptyStagePlacement(StageId::new(0)))
        );

        let duplicate = ExecutionPlan::new(
            model(),
            ParameterVersion::new(1),
            vec![
                StagePlacement::new(
                    StageId::new(0),
                    runtime("fixture-runtime"),
                    vec![present.clone()],
                )
                .expect("placement"),
                StagePlacement::new(StageId::new(0), runtime("fixture-runtime"), vec![present])
                    .expect("placement"),
            ],
            &topology,
        );
        assert_eq!(duplicate, Err(PlanError::DuplicateStage(StageId::new(0))));

        let outside = ExecutionPlan::new(
            model(),
            ParameterVersion::new(1),
            vec![
                StagePlacement::new(StageId::new(3), runtime("fixture-runtime"), vec![absent])
                    .expect("placement"),
            ],
            &topology,
        );
        assert_eq!(outside, Err(PlanError::UnknownDevice(StageId::new(3))));
    }
}
