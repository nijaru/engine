//! `SafeTensors` artifact adapter for Ribn's model-loading pressure tests.
//!
//! This crate validates artifact bytes and exposes borrowed tensor payloads plus
//! format metadata. It does not allocate execution tensors, choose parameter
//! semantics, decide placement, or create [`ribn_foundation::ParameterMaterialization`]
//! values. Those responsibilities belong to model integration and preparation.
//!
//! Validation and metadata extraction happen once, when the artifact is
//! constructed. Tensor lookups afterwards are index lookups into retained
//! metadata, so reading every tensor in a shard does not reparse the header or
//! rescan the tensor names.
//!
//! Artifact bytes are owned in host memory. Mapping an immutable local file
//! instead would avoid that copy, but mapping needs an `unsafe` call that this
//! workspace forbids, so owned bytes remain the only storage here. Callers that
//! cannot afford whole-artifact residency should bound how many artifacts stay
//! open rather than expecting this layer to stream.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ribn_foundation::ScalarType;
use safetensors::tensor::{Dtype, SafeTensors};

/// Validated `SafeTensors` bytes owned independently of any execution backend.
#[derive(Clone)]
pub struct SafeTensorArtifact {
    bytes: Arc<[u8]>,
    index: Arc<BTreeMap<String, TensorEntry>>,
    source: Option<PathBuf>,
}

#[derive(Clone)]
struct TensorEntry {
    scalar_type: ScalarType,
    shape: Vec<usize>,
    range: Range<usize>,
}

impl SafeTensorArtifact {
    /// Validate and retain an in-memory `SafeTensors` artifact.
    ///
    /// # Errors
    /// Returns [`ArtifactError::InvalidFormat`] when the bytes are not a valid
    /// `SafeTensors` artifact.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, ArtifactError> {
        let bytes = bytes.into();
        let parsed = SafeTensors::deserialize(bytes.as_ref()).map_err(ArtifactError::format)?;
        let index = index_tensors(bytes.as_ref(), &parsed)?;
        Ok(Self {
            bytes,
            index: Arc::new(index),
            source: None,
        })
    }

    /// Read and validate a `SafeTensors` artifact from a local file.
    ///
    /// The whole file is read into owned host memory and stays resident for the
    /// lifetime of the artifact.
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

    /// Tensor names in the artifact, read from retained metadata.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// Borrow one tensor payload from the owned artifact bytes.
    ///
    /// # Errors
    /// Returns [`ArtifactError::MissingTensor`] when `name` is absent.
    pub fn tensor(&self, name: &str) -> Result<ArtifactTensor<'_>, ArtifactError> {
        let entry = self
            .index
            .get(name)
            .ok_or_else(|| ArtifactError::MissingTensor(name.to_owned()))?;
        let data = self
            .bytes
            .get(entry.range.clone())
            .ok_or_else(|| ArtifactError::format("retained tensor payload is out of range"))?;
        Ok(ArtifactTensor {
            scalar_type: entry.scalar_type.clone(),
            shape: entry.shape.clone(),
            data,
        })
    }

    /// Payload bytes this artifact keeps in host memory.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Payload bytes the artifact exposes, regardless of storage.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Number of tensors described by the retained metadata.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }
}

/// Retain validated metadata and payload offsets for every tensor once.
fn index_tensors(
    bytes: &[u8],
    parsed: &SafeTensors<'_>,
) -> Result<BTreeMap<String, TensorEntry>, ArtifactError> {
    let base = bytes.as_ptr() as usize;
    let mut index = BTreeMap::new();
    for name in parsed.names() {
        let view = parsed.tensor(name).map_err(ArtifactError::format)?;
        let data = view.data();
        let start = (data.as_ptr() as usize)
            .checked_sub(base)
            .ok_or_else(|| ArtifactError::format("tensor payload precedes the artifact"))?;
        let end = start
            .checked_add(data.len())
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| ArtifactError::format("tensor payload exceeds the artifact"))?;
        index.insert(
            name.to_owned(),
            TensorEntry {
                scalar_type: scalar_type(view.dtype()),
                shape: view.shape().to_vec(),
                range: start..end,
            },
        );
    }
    Ok(index)
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
    use std::fmt::Write as _;

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

    /// Several tensors with different shapes, so index offsets are exercised
    /// beyond a single leading payload.
    fn multi_tensor_fixture() -> Vec<u8> {
        let payloads: [(&str, &[usize], Vec<f32>); 3] = [
            ("first.weight", &[2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ("second.weight", &[3], vec![5.0, 6.0, 7.0]),
            ("third.weight", &[1, 1], vec![8.0]),
        ];
        let mut data = Vec::new();
        let mut header = String::from("{");
        for (position, (name, shape, values)) in payloads.iter().enumerate() {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            if position > 0 {
                header.push(',');
            }
            let shape = shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let _ = write!(
                header,
                r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{start},{}]}}"#,
                data.len()
            );
        }
        header.push('}');
        let mut header = header.into_bytes();
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
        assert_eq!(artifact.names(), vec!["weight"]);
        let tensor = artifact.tensor("weight").expect("weight");
        assert_eq!(tensor.scalar_type(), &ScalarType::F32);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(tensor.data().len(), 16);
        assert!(matches!(
            artifact.tensor("missing"),
            Err(ArtifactError::MissingTensor(name)) if name == "missing"
        ));
    }

    /// Every tensor keeps its own payload after the metadata is indexed once, so a
    /// wrong retained offset cannot pass as a lookup miss.
    #[test]
    fn retained_offsets_address_each_tensor_payload() {
        let artifact = SafeTensorArtifact::from_bytes(multi_tensor_fixture()).expect("artifact");
        assert_eq!(
            artifact.names(),
            vec!["first.weight", "second.weight", "third.weight"]
        );
        assert_eq!(artifact.tensor_count(), 3);
        assert_eq!(artifact.heap_bytes(), artifact.len());

        let read = |name: &str| {
            artifact
                .tensor(name)
                .expect("tensor")
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect::<Vec<_>>()
        };
        assert_eq!(read("first.weight"), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(read("second.weight"), vec![5.0, 6.0, 7.0]);
        assert_eq!(read("third.weight"), vec![8.0]);
        assert_eq!(
            artifact.tensor("second.weight").expect("shape").shape(),
            [3]
        );
    }

    #[test]
    fn rejects_truncated_and_malformed_artifacts() {
        let mut truncated = fixture();
        truncated.truncate(truncated.len() - 4);
        assert!(matches!(
            SafeTensorArtifact::from_bytes(truncated),
            Err(ArtifactError::InvalidFormat(_))
        ));
        assert!(matches!(
            SafeTensorArtifact::from_bytes(vec![0_u8; 8]),
            Err(ArtifactError::InvalidFormat(_))
        ));
    }
}
