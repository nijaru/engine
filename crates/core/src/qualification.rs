//! Execution-variant identity and qualification state.

use std::fmt;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionVariantId(String);

impl ExecutionVariantId {
    /// # Errors
    ///
    /// Returns [`ExecutionVariantIdError`] when the identifier is empty.
    pub fn new(value: impl Into<String>) -> Result<Self, ExecutionVariantIdError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ExecutionVariantIdError);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QualificationStatus {
    Qualified,
    Experimental,
    Incompatible,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionVariant {
    id: ExecutionVariantId,
    status: QualificationStatus,
}

impl ExecutionVariant {
    #[must_use]
    pub fn baseline() -> Self {
        Self {
            id: ExecutionVariantId("baseline".to_owned()),
            status: QualificationStatus::Qualified,
        }
    }

    #[must_use]
    pub const fn new(id: ExecutionVariantId, status: QualificationStatus) -> Self {
        Self { id, status }
    }

    #[must_use]
    pub const fn id(&self) -> &ExecutionVariantId {
        &self.id
    }

    #[must_use]
    pub const fn status(&self) -> QualificationStatus {
        self.status
    }

    #[must_use]
    pub const fn eligible_for_automatic_selection(&self) -> bool {
        matches!(self.status, QualificationStatus::Qualified)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionVariantIdError;

impl fmt::Display for ExecutionVariantIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("execution variant identity must not be empty")
    }
}

impl std::error::Error for ExecutionVariantIdError {}
