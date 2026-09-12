use std::fmt;

use crate::DeviceId;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ParameterId(String);

impl ParameterId {
    /// # Errors
    /// Returns an error for an empty identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, ParameterError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ParameterError::EmptyIdentity("parameter"));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParameterVersion(u64);

impl ParameterVersion {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterSpec {
    id: ParameterId,
    shape: Vec<u64>,
}

impl ParameterSpec {
    #[must_use]
    pub fn new(id: ParameterId, shape: Vec<u64>) -> Self {
        Self { id, shape }
    }

    #[must_use]
    pub fn id(&self) -> &ParameterId {
        &self.id
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ScalarType {
    F16,
    Bf16,
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Bool,
    Named(String),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum StorageEncoding {
    Dense,
    Named(String),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LayoutId(String);

impl LayoutId {
    /// # Errors
    /// Returns an error for an empty layout identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, ParameterError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ParameterError::EmptyIdentity("layout"));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StorageId(u64);

impl StorageId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializationPart {
    storage: StorageId,
    device: DeviceId,
}

impl MaterializationPart {
    #[must_use]
    pub fn new(storage: StorageId, device: DeviceId) -> Self {
        Self { storage, device }
    }

    #[must_use]
    pub const fn storage(&self) -> StorageId {
        self.storage
    }

    #[must_use]
    pub fn device(&self) -> &DeviceId {
        &self.device
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterMaterialization {
    parameter: ParameterId,
    version: ParameterVersion,
    scalar_type: ScalarType,
    encoding: StorageEncoding,
    layout: LayoutId,
    parts: Vec<MaterializationPart>,
}

impl ParameterMaterialization {
    /// # Errors
    /// Returns an error when no physical storage part is supplied.
    pub fn new(
        parameter: ParameterId,
        version: ParameterVersion,
        scalar_type: ScalarType,
        encoding: StorageEncoding,
        layout: LayoutId,
        parts: Vec<MaterializationPart>,
    ) -> Result<Self, ParameterError> {
        if parts.is_empty() {
            return Err(ParameterError::EmptyMaterialization);
        }
        Ok(Self {
            parameter,
            version,
            scalar_type,
            encoding,
            layout,
            parts,
        })
    }

    #[must_use]
    pub fn parameter(&self) -> &ParameterId {
        &self.parameter
    }

    #[must_use]
    pub const fn version(&self) -> ParameterVersion {
        self.version
    }

    #[must_use]
    pub fn scalar_type(&self) -> &ScalarType {
        &self.scalar_type
    }

    #[must_use]
    pub fn encoding(&self) -> &StorageEncoding {
        &self.encoding
    }

    #[must_use]
    pub fn layout(&self) -> &LayoutId {
        &self.layout
    }

    #[must_use]
    pub fn parts(&self) -> &[MaterializationPart] {
        &self.parts
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterSet {
    version: ParameterVersion,
    materializations: Vec<ParameterMaterialization>,
}

impl ParameterSet {
    /// # Errors
    /// Returns an error when a materialization belongs to another parameter version.
    pub fn new(
        version: ParameterVersion,
        materializations: Vec<ParameterMaterialization>,
    ) -> Result<Self, ParameterError> {
        if materializations
            .iter()
            .any(|materialization| materialization.version() != version)
        {
            return Err(ParameterError::MixedParameterVersions);
        }
        Ok(Self {
            version,
            materializations,
        })
    }

    #[must_use]
    pub const fn version(&self) -> ParameterVersion {
        self.version
    }

    #[must_use]
    pub fn materializations(&self) -> &[ParameterMaterialization] {
        &self.materializations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterError {
    EmptyIdentity(&'static str),
    EmptyMaterialization,
    MixedParameterVersions,
}

impl fmt::Display for ParameterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentity(kind) => write!(f, "{kind} identity must not be empty"),
            Self::EmptyMaterialization => {
                f.write_str("parameter materialization must contain physical storage")
            }
            Self::MixedParameterVersions => {
                f.write_str("parameter set contains more than one parameter version")
            }
        }
    }
}

impl std::error::Error for ParameterError {}
