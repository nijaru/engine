//! Execution-variant identity and qualification state.

use std::fmt;

use crate::backend::BackendId;
use crate::device::DeviceId;
use crate::model::ModelId;

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
    evidence: Option<QualificationEvidence>,
}

impl ExecutionVariant {
    #[must_use]
    pub fn baseline() -> Self {
        Self {
            id: ExecutionVariantId("baseline".to_owned()),
            status: QualificationStatus::Experimental,
            evidence: None,
        }
    }

    #[must_use]
    pub const fn new(id: ExecutionVariantId, status: QualificationStatus) -> Self {
        Self {
            id,
            status,
            evidence: None,
        }
    }

    #[must_use]
    pub const fn id(&self) -> &ExecutionVariantId {
        &self.id
    }

    #[must_use]
    pub const fn status(&self) -> QualificationStatus {
        self.status
    }

    /// Attach qualification issued for this exact execution variant.
    /// Core checks identity, not the validity of the external measurements;
    /// callers are responsible for issuing evidence only after qualification.
    ///
    /// # Errors
    ///
    /// Returns an error when the evidence names another variant.
    pub fn with_qualification(
        mut self,
        evidence: QualificationEvidence,
    ) -> Result<Self, QualificationError> {
        if evidence.variant != self.id {
            return Err(QualificationError::VariantMismatch);
        }
        self.status = QualificationStatus::Qualified;
        self.evidence = Some(evidence);
        Ok(self)
    }

    #[must_use]
    pub const fn evidence(&self) -> Option<&QualificationEvidence> {
        self.evidence.as_ref()
    }

    /// Automatic selection requires qualification for the exact execution
    /// context being selected. A status label alone never authorizes it.
    #[must_use]
    pub fn eligible_for_automatic_selection(&self, scope: &QualificationScope) -> bool {
        self.status == QualificationStatus::Qualified
            && self
                .evidence
                .as_ref()
                .is_some_and(|evidence| &evidence.scope == scope)
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

/// Caller-supplied compatibility identity for a correctness qualification.
/// Artifact identity must include the exact revision and quantization. Runtime
/// identity must include the relevant hardware and software versions. State
/// execution identity must cover state layout, graph/capture mode, speculation,
/// and distributed layout. Core compares these identities without discovering
/// the environment or claiming the supplied identity was independently verified.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct QualificationScope {
    model: ModelId,
    artifact_identity: String,
    backend: BackendId,
    device: DeviceId,
    runtime_identity: String,
    state_execution_identity: String,
}

impl QualificationScope {
    /// # Errors
    ///
    /// Returns an error when a compatibility dimension is empty.
    pub fn new(
        model: ModelId,
        artifact_identity: impl Into<String>,
        backend: BackendId,
        device: DeviceId,
        runtime_identity: impl Into<String>,
        state_execution_identity: impl Into<String>,
    ) -> Result<Self, QualificationError> {
        let artifact_identity = artifact_identity.into();
        let runtime_identity = runtime_identity.into();
        let state_execution_identity = state_execution_identity.into();
        if [
            &artifact_identity,
            &runtime_identity,
            &state_execution_identity,
        ]
        .iter()
        .any(|identity| identity.trim().is_empty())
        {
            return Err(QualificationError::EmptyIdentity);
        }
        Ok(Self {
            model,
            artifact_identity,
            backend,
            device,
            runtime_identity,
            state_execution_identity,
        })
    }

    #[must_use]
    pub const fn model(&self) -> &ModelId {
        &self.model
    }
    #[must_use]
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }
    #[must_use]
    pub const fn backend(&self) -> &BackendId {
        &self.backend
    }
    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }
    #[must_use]
    pub fn runtime_identity(&self) -> &str {
        &self.runtime_identity
    }
    #[must_use]
    pub fn state_execution_identity(&self) -> &str {
        &self.state_execution_identity
    }
}

/// Reference to externally established correctness evidence for one variant
/// and compatibility scope. Constructing this value does not run a parity test
/// or verify the evidence; its issuer owns those checks.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct QualificationEvidence {
    variant: ExecutionVariantId,
    scope: QualificationScope,
    evidence_id: String,
}

impl QualificationEvidence {
    /// # Errors
    ///
    /// Returns an error when the external evidence reference is empty.
    pub fn new(
        variant: ExecutionVariantId,
        scope: QualificationScope,
        evidence_id: impl Into<String>,
    ) -> Result<Self, QualificationError> {
        let evidence_id = evidence_id.into();
        if evidence_id.trim().is_empty() {
            return Err(QualificationError::EmptyIdentity);
        }
        Ok(Self {
            variant,
            scope,
            evidence_id,
        })
    }
    #[must_use]
    pub const fn variant(&self) -> &ExecutionVariantId {
        &self.variant
    }
    #[must_use]
    pub const fn scope(&self) -> &QualificationScope {
        &self.scope
    }
    #[must_use]
    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QualificationError {
    EmptyIdentity,
    VariantMismatch,
}
impl fmt::Display for QualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentity => f.write_str("qualification identity must not be empty"),
            Self::VariantMismatch => {
                f.write_str("qualification evidence names another execution variant")
            }
        }
    }
}
impl std::error::Error for QualificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> QualificationScope {
        QualificationScope::new(
            ModelId::new("qwen").expect("model"),
            "sha256:artifact;q4",
            BackendId::new("cuda").expect("backend"),
            DeviceId::new(0),
            "sm89;driver-test;runtime-test",
            "kv-v1;recurrent-v1;eager;no-spec;single-device",
        )
        .expect("scope")
    }

    #[test]
    fn automatic_selection_requires_evidence_matching_every_identity_dimension() {
        let expected = scope();
        let baseline = ExecutionVariant::baseline();
        assert!(!baseline.eligible_for_automatic_selection(&expected));
        let status_only =
            ExecutionVariant::new(baseline.id().clone(), QualificationStatus::Qualified);
        assert!(!status_only.eligible_for_automatic_selection(&expected));
        let evidence =
            QualificationEvidence::new(baseline.id().clone(), expected.clone(), "parity-run-1")
                .expect("evidence");
        let qualified = baseline.with_qualification(evidence).expect("qualify");
        assert!(qualified.eligible_for_automatic_selection(&expected));
        for dimension in 0..6 {
            let mut changed = expected.clone();
            match dimension {
                0 => changed.model = ModelId::new("other-model").expect("model"),
                1 => changed.artifact_identity = "other-artifact".into(),
                2 => changed.backend = BackendId::new("other-backend").expect("backend"),
                3 => changed.device = DeviceId::new(1),
                4 => changed.runtime_identity = "other-runtime".into(),
                5 => changed.state_execution_identity = "other-state-execution".into(),
                _ => unreachable!(),
            }
            assert!(
                !qualified.eligible_for_automatic_selection(&changed),
                "dimension {dimension}"
            );
        }
    }

    #[test]
    fn evidence_cannot_be_reused_for_another_variant() {
        let evidence = QualificationEvidence::new(
            ExecutionVariantId::new("other").expect("variant"),
            scope(),
            "parity-run-1",
        )
        .expect("evidence");
        assert_eq!(
            ExecutionVariant::baseline().with_qualification(evidence),
            Err(QualificationError::VariantMismatch)
        );
        assert!(
            QualificationEvidence::new(
                ExecutionVariantId::new("baseline").expect("variant"),
                scope(),
                " "
            )
            .is_err()
        );
    }
}
