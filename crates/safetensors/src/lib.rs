//! `SafeTensors` artifact adapter for Ribn's model-loading pressure tests.
//!
//! This crate validates artifact bytes and exposes borrowed tensor payloads plus
//! format metadata. It does not allocate execution tensors, choose parameter
//! semantics, decide placement, or create [`ribn_foundation::ParameterMaterialization`]
//! values. Those responsibilities belong to model integration and preparation.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ribn_foundation::ScalarType;
use safetensors::tensor::{Dtype, SafeTensors};

/// Validated `SafeTensors` bytes owned independently of any execution backend.
#[derive(Clone)]
pub struct SafeTensorArtifact {
    bytes: Arc<[u8]>,
    source: Option<PathBuf>,
}

impl SafeTensorArtifact {
    /// Validate and retain an in-memory `SafeTensors` artifact.
    ///
    /// # Errors
    /// Returns [`ArtifactError::InvalidFormat`] when the bytes are not a valid
    /// `SafeTensors` artifact.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, ArtifactError> {
        let bytes = bytes.into();
        SafeTensors::deserialize(bytes.as_ref()).map_err(ArtifactError::format)?;
        Ok(Self {
            bytes,
            source: None,
        })
    }

    /// Read and validate a `SafeTensors` artifact from a local file.
    ///
    /// # Errors
    /// Returns an I/O error when the file cannot be read, or a format error when
    /// its contents are not a valid `SafeTensors` artifact.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, ArtifactError> {
        let path = path.into();
        let bytes = fs::read(&path).map_err(|error| ArtifactError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?;
        let mut artifact = Self::from_bytes(bytes)?;
        artifact.source = Some(path);
        Ok(artifact)
    }

    #[must_use]
    pub fn source(&self) -> Option<&Path> {
        self.source.as_deref()
    }

    /// Return tensor names without copying tensor payloads.
    ///
    /// # Errors
    /// Returns a format error if validated bytes somehow fail to deserialize.
    pub fn names(&self) -> Result<Vec<String>, ArtifactError> {
        let tensors = self.parse()?;
        Ok(tensors.names().into_iter().map(str::to_owned).collect())
    }

    /// Borrow one tensor payload from the owned artifact bytes.
    ///
    /// # Errors
    /// Returns [`ArtifactError::MissingTensor`] when `name` is absent, or a
    /// format error if validated bytes somehow fail to deserialize.
    pub fn tensor(&self, name: &str) -> Result<ArtifactTensor<'_>, ArtifactError> {
        let tensors = self.parse()?;
        if !tensors.names().contains(&name) {
            return Err(ArtifactError::MissingTensor(name.to_owned()));
        }
        let view = tensors.tensor(name).map_err(ArtifactError::format)?;
        Ok(ArtifactTensor {
            scalar_type: scalar_type(view.dtype()),
            shape: view.shape().to_vec(),
            data: view.data(),
        })
    }

    fn parse(&self) -> Result<SafeTensors<'_>, ArtifactError> {
        SafeTensors::deserialize(self.bytes.as_ref()).map_err(ArtifactError::format)
    }
}

/// Format-level tensor view. Shape and bytes describe the artifact representation,
/// not a prepared device tensor or serving parameter materialization.
pub struct ArtifactTensor<'a> {
    scalar_type: ScalarType,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl ArtifactTensor<'_> {
    #[must_use]
    pub fn scalar_type(&self) -> &ScalarType {
        &self.scalar_type
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub const fn data(&self) -> &[u8] {
        self.data
    }
}

fn scalar_type(dtype: Dtype) -> ScalarType {
    match dtype {
        Dtype::F16 => ScalarType::F16,
        Dtype::BF16 => ScalarType::Bf16,
        Dtype::F32 => ScalarType::F32,
        Dtype::F64 => ScalarType::F64,
        Dtype::I8 => ScalarType::I8,
        Dtype::U8 => ScalarType::U8,
        Dtype::I16 => ScalarType::I16,
        Dtype::U16 => ScalarType::U16,
        Dtype::I32 => ScalarType::I32,
        Dtype::U32 => ScalarType::U32,
        Dtype::I64 => ScalarType::I64,
        Dtype::U64 => ScalarType::U64,
        Dtype::BOOL => ScalarType::Bool,
        _ => ScalarType::Named(dtype.to_string()),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    Io { path: PathBuf, message: String },
    InvalidFormat(String),
    MissingTensor(String),
}

impl ArtifactError {
    fn format(error: impl fmt::Display) -> Self {
        Self::InvalidFormat(error.to_string())
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(
                    f,
                    "could not read SafeTensors artifact {}: {message}",
                    path.display()
                )
            }
            Self::InvalidFormat(message) => write!(f, "invalid SafeTensors artifact: {message}"),
            Self::MissingTensor(name) => write!(f, "SafeTensors artifact has no tensor {name:?}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let mut header = format!(
            r#"{{"weight":{{"dtype":"F32","shape":[2,2],"data_offsets":[0,{}]}}}}"#,
            data.len()
        )
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut artifact = Vec::new();
        artifact.extend_from_slice(
            &u64::try_from(header.len())
                .expect("fixture header length")
                .to_le_bytes(),
        );
        artifact.extend_from_slice(&header);
        artifact.extend_from_slice(&data);
        artifact
    }

    #[test]
    fn exposes_format_metadata_and_borrowed_payload() {
        let artifact = SafeTensorArtifact::from_bytes(fixture()).expect("artifact");
        assert_eq!(artifact.names().expect("names"), vec!["weight"]);
        let tensor = artifact.tensor("weight").expect("weight");
        assert_eq!(tensor.scalar_type(), &ScalarType::F32);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(tensor.data().len(), 16);
        assert!(matches!(
            artifact.tensor("missing"),
            Err(ArtifactError::MissingTensor(name)) if name == "missing"
        ));
    }
}
