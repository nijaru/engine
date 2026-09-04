//! Model-owned resource residency, separate from per-request inference state.

use std::fmt;

use crate::device::DeviceId;
use crate::model::ModelId;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResidencyLocation {
    Device(DeviceId),
    Host,
    Mapped,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelResourceId(String);

impl ModelResourceId {
    /// # Errors
    ///
    /// Returns [`ResidencyError::EmptyResource`] when the identifier is empty.
    pub fn new(value: impl Into<String>) -> Result<Self, ResidencyError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ResidencyError::EmptyResource);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResidencyOverride {
    resource: ModelResourceId,
    location: ResidencyLocation,
    prefetchable: bool,
}

impl ResidencyOverride {
    #[must_use]
    pub const fn new(
        resource: ModelResourceId,
        location: ResidencyLocation,
        prefetchable: bool,
    ) -> Self {
        Self {
            resource,
            location,
            prefetchable,
        }
    }

    #[must_use]
    pub const fn resource(&self) -> &ModelResourceId {
        &self.resource
    }

    #[must_use]
    pub const fn location(&self) -> ResidencyLocation {
        self.location
    }

    #[must_use]
    pub const fn prefetchable(&self) -> bool {
        self.prefetchable
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelResidencyPlan {
    model: ModelId,
    default_location: ResidencyLocation,
    overrides: Vec<ResidencyOverride>,
}

impl ModelResidencyPlan {
    #[must_use]
    pub fn single_device(model: ModelId, device: DeviceId) -> Self {
        Self {
            model,
            default_location: ResidencyLocation::Device(device),
            overrides: Vec::new(),
        }
    }

    /// # Errors
    ///
    /// Returns [`ResidencyError::DuplicateResource`] when a resource is
    /// overridden more than once.
    pub fn new(
        model: ModelId,
        default_location: ResidencyLocation,
        overrides: Vec<ResidencyOverride>,
    ) -> Result<Self, ResidencyError> {
        for (index, entry) in overrides.iter().enumerate() {
            if overrides[..index]
                .iter()
                .any(|previous| previous.resource() == entry.resource())
            {
                return Err(ResidencyError::DuplicateResource(
                    entry.resource().as_str().to_owned(),
                ));
            }
        }
        Ok(Self {
            model,
            default_location,
            overrides,
        })
    }

    #[must_use]
    pub const fn model(&self) -> &ModelId {
        &self.model
    }

    #[must_use]
    pub const fn default_location(&self) -> ResidencyLocation {
        self.default_location
    }

    #[must_use]
    pub fn overrides(&self) -> &[ResidencyOverride] {
        &self.overrides
    }

    #[must_use]
    pub fn location_for(&self, resource: &ModelResourceId) -> ResidencyLocation {
        self.overrides
            .iter()
            .find(|entry| entry.resource() == resource)
            .map_or(self.default_location, ResidencyOverride::location)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ResidencyError {
    EmptyResource,
    DuplicateResource(String),
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyResource => f.write_str("model resource identity must not be empty"),
            Self::DuplicateResource(resource) => {
                write!(
                    f,
                    "model residency overrides resource {resource:?} more than once"
                )
            }
        }
    }
}

impl std::error::Error for ResidencyError {}
