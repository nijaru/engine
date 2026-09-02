//! Model descriptions and provider boundary.

use std::fmt;

use crate::execution::ExecutionPlan;
use crate::state::StateRequirement;
use crate::tensor::{Quantization, WeightFormat};

/// Stable model identity used by requests, plans, and providers.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(String);

impl ModelId {
    /// # Errors
    ///
    /// Returns [`ModelIdError`] when `value` is empty or whitespace-only.
    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ModelIdError);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Returned when a model identity is empty or whitespace-only.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelIdError;

impl fmt::Display for ModelIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("model identity must not be empty")
    }
}

impl std::error::Error for ModelIdError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelRegionId(u16);

impl ModelRegionId {
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A model region is intentionally coarser than an operator or compiler IR.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ModelRegionKind {
    RecurrentAttention,
    FullAttention,
    FeedForward,
    Embedding,
    OutputProjection,
    Encoder,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelRegion {
    id: ModelRegionId,
    kind: ModelRegionKind,
}

impl ModelRegion {
    #[must_use]
    pub const fn new(id: ModelRegionId, kind: ModelRegionKind) -> Self {
        Self { id, kind }
    }

    #[must_use]
    pub const fn id(self) -> ModelRegionId {
        self.id
    }

    #[must_use]
    pub const fn kind(self) -> ModelRegionKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MtpCapability {
    draft_layers: u8,
    max_draft_tokens: u8,
}

impl MtpCapability {
    #[must_use]
    pub const fn new(draft_layers: u8, max_draft_tokens: u8) -> Option<Self> {
        if draft_layers == 0 || max_draft_tokens == 0 {
            None
        } else {
            Some(Self {
                draft_layers,
                max_draft_tokens,
            })
        }
    }

    #[must_use]
    pub const fn draft_layers(self) -> u8 {
        self.draft_layers
    }

    #[must_use]
    pub const fn max_draft_tokens(self) -> u8 {
        self.max_draft_tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelCapabilities {
    mtp: Option<MtpCapability>,
    vision_encoder: bool,
}

impl ModelCapabilities {
    #[must_use]
    pub const fn new(mtp: Option<MtpCapability>, vision_encoder: bool) -> Self {
        Self {
            mtp,
            vision_encoder,
        }
    }

    #[must_use]
    pub const fn mtp(self) -> Option<MtpCapability> {
        self.mtp
    }

    #[must_use]
    pub const fn has_vision_encoder(self) -> bool {
        self.vision_encoder
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WeightDescription {
    format: WeightFormat,
    quantization: Quantization,
}

impl WeightDescription {
    #[must_use]
    pub const fn new(format: WeightFormat, quantization: Quantization) -> Self {
        Self {
            format,
            quantization,
        }
    }

    #[must_use]
    pub fn format(&self) -> &WeightFormat {
        &self.format
    }

    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }
}

/// Static metadata needed by a planner before it selects a concrete execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDescription {
    id: ModelId,
    architecture: String,
    regions: Vec<ModelRegion>,
    state_requirements: Vec<StateRequirement>,
    capabilities: ModelCapabilities,
    weights: WeightDescription,
}

impl ModelDescription {
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidDescription`] when required metadata is
    /// missing.
    pub fn new(
        id: ModelId,
        architecture: impl Into<String>,
        regions: Vec<ModelRegion>,
        state_requirements: Vec<StateRequirement>,
        capabilities: ModelCapabilities,
        weights: WeightDescription,
    ) -> Result<Self, ModelError> {
        let architecture = architecture.into();
        if architecture.trim().is_empty() {
            return Err(ModelError::InvalidDescription(
                "architecture must not be empty",
            ));
        }
        if regions.is_empty() {
            return Err(ModelError::InvalidDescription(
                "model description must contain at least one region",
            ));
        }
        Ok(Self {
            id,
            architecture,
            regions,
            state_requirements,
            capabilities,
            weights,
        })
    }

    #[must_use]
    pub fn id(&self) -> &ModelId {
        &self.id
    }

    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    #[must_use]
    pub fn regions(&self) -> &[ModelRegion] {
        &self.regions
    }

    #[must_use]
    pub fn state_requirements(&self) -> &[StateRequirement] {
        &self.state_requirements
    }

    #[must_use]
    pub const fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    #[must_use]
    pub const fn weights(&self) -> &WeightDescription {
        &self.weights
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    InvalidDescription(&'static str),
    PlanModelMismatch,
    UnknownRegion(ModelRegionId),
    UndeclaredState(StateRequirement),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescription(reason) => write!(f, "invalid model description: {reason}"),
            Self::PlanModelMismatch => f.write_str("execution plan targets a different model"),
            Self::UnknownRegion(region) => write!(
                f,
                "execution plan references unknown region {}",
                region.get()
            ),
            Self::UndeclaredState(requirement) => {
                write!(
                    f,
                    "execution plan requires undeclared state: {requirement:?}"
                )
            }
        }
    }
}

impl std::error::Error for ModelError {}

/// Provider implementations may be native or compatibility-backed. The core
/// only requires metadata and plan validation; loading and execution remain
/// provider/backend concerns.
pub trait ModelProvider: Send + Sync {
    fn description(&self) -> &ModelDescription;

    /// # Errors
    ///
    /// Returns [`ModelError::PlanModelMismatch`] when the plan targets another
    /// model.
    fn validate_plan(&self, plan: &ExecutionPlan) -> Result<(), ModelError> {
        if plan.model() != self.description().id() {
            return Err(ModelError::PlanModelMismatch);
        }
        if let Some(region) = plan.stages().iter().find_map(|stage| {
            (!self
                .description()
                .regions()
                .iter()
                .any(|known| known.id() == stage.region()))
            .then_some(stage.region())
        }) {
            return Err(ModelError::UnknownRegion(region));
        }
        if let Some(requirement) = plan.state_requirements().iter().find(|requirement| {
            !self
                .description()
                .state_requirements()
                .contains(requirement)
        }) {
            return Err(ModelError::UndeclaredState(*requirement));
        }
        Ok(())
    }
}
