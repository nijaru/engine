//! Minimal GGUF metadata and tensor-directory reader.
//!
//! This crate owns the optional GGUF format boundary. It reads headers,
//! metadata, and tensor descriptors without loading bulk tensor data; model
//! execution remains owned by a provider and compute backend.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use engine_core::{ModelLoadError, WeightArtifact, WeightDescription, WeightLoader, WeightSource};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 1_000_000;
const MAX_METADATA_DEPTH: u32 = 16;
const MAX_TENSOR_DIMS: u32 = 16;
const MAX_TENSORS: u64 = 10_000_000;

/// GGUF metadata value types from the on-disk format.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(u64::from(*value)),
            Self::U16(value) => Some(u64::from(*value)),
            Self::U32(value) => Some(u64::from(*value)),
            Self::U64(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    name: String,
    dimensions: Vec<u64>,
    value_type: u32,
    offset: u64,
}

impl TensorInfo {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    /// Raw GGML type ID. Keeping unknown IDs lossless avoids making this
    /// metadata reader reject newer quantization types.
    #[must_use]
    pub const fn value_type(&self) -> u32 {
        self.value_type
    }

    /// Offset relative to the beginning of aligned GGUF tensor data.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// # Errors
    ///
    /// Returns [`GgufError::ElementCountOverflow`] when the dimensions do not
    /// fit in a `u64` element count.
    pub fn element_count(&self) -> Result<u64, GgufError> {
        self.dimensions.iter().try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(GgufError::ElementCountOverflow)
        })
    }

    /// Encoded byte length for the supported GGML tensor types. Unknown types
    /// remain parseable but are rejected here until a backend provides their
    /// block layout.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor type is unsupported, its dimensions do
    /// not fit, or its dimensions are not valid for its quantization block.
    pub fn byte_len(&self) -> Result<u64, GgufError> {
        let (block_elements, block_bytes) = tensor_layout(self.value_type)?;
        let elements = self.element_count()?;
        if elements % block_elements != 0 {
            return Err(GgufError::QuantizationBlockMismatch {
                name: self.name.clone(),
                elements,
                block_elements,
            });
        }
        (elements / block_elements)
            .checked_mul(block_bytes)
            .ok_or(GgufError::TensorByteLengthOverflow)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufFile {
    path: PathBuf,
    version: u32,
    tensor_count: u64,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    tensor_data_offset: u64,
    file_len: u64,
}

impl GgufFile {
    /// Open and parse the GGUF header, metadata, and tensor directory.
    ///
    /// Bulk tensor data is not read into memory.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] for an invalid or unsupported GGUF file.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, GgufError> {
        let path = path.into();
        let mut file = File::open(&path).map_err(|error| GgufError::io(&path, &error))?;
        Self::read_from(&mut file, path)
    }

    fn read_from<R: Read + Seek>(reader: &mut R, path: PathBuf) -> Result<Self, GgufError> {
        let file_len = reader
            .seek(SeekFrom::End(0))
            .map_err(|error| GgufError::io(&path, &error))?;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|error| GgufError::io(&path, &error))?;

        let magic = read_u32(reader, &path)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }
        let version = read_u32(reader, &path)?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = read_count(reader, &path, "tensor", MAX_TENSORS)?;
        let metadata_count = read_count(reader, &path, "metadata", MAX_ARRAY_ITEMS)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = read_string(reader, &path)?;
            let value_type = read_u32(reader, &path)?;
            let value = read_value(reader, &path, value_type, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(GgufError::DuplicateMetadata(key));
            }
        }

        let mut tensors = Vec::with_capacity(
            usize::try_from(tensor_count).map_err(|_| GgufError::CountOverflow("tensor"))?,
        );
        for _ in 0..tensor_count {
            let name = read_string(reader, &path)?;
            let dimension_count = read_u32(reader, &path)?;
            if dimension_count > MAX_TENSOR_DIMS {
                return Err(GgufError::InvalidDimensionCount(dimension_count));
            }
            let mut dimensions = Vec::with_capacity(
                usize::try_from(dimension_count)
                    .map_err(|_| GgufError::CountOverflow("tensor dimension"))?,
            );
            for _ in 0..dimension_count {
                dimensions.push(read_u64(reader, &path)?);
            }
            let value_type = read_u32(reader, &path)?;
            let offset = read_u64(reader, &path)?;
            tensors.push(TensorInfo {
                name,
                dimensions,
                value_type,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetadataValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(alignment));
        }
        let directory_end = reader
            .stream_position()
            .map_err(|error| GgufError::io(&path, &error))?;
        let tensor_data_offset = align(directory_end, alignment)?;
        if tensor_data_offset > file_len {
            return Err(GgufError::Truncated);
        }
        for tensor in &tensors {
            let remaining = file_len
                .checked_sub(tensor_data_offset)
                .ok_or(GgufError::Truncated)?;
            if tensor.offset() > remaining {
                return Err(GgufError::TensorOffsetOutOfBounds {
                    name: tensor.name().to_owned(),
                    offset: tensor.offset(),
                });
            }
        }

        Ok(Self {
            path,
            version,
            tensor_count,
            metadata,
            tensors,
            tensor_data_offset,
            file_len,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    #[must_use]
    pub const fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    #[must_use]
    pub fn metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    #[must_use]
    pub fn metadata_map(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name() == name)
    }

    /// Return the absolute byte range for one encoded tensor without reading
    /// its bulk data.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is missing, unsupported, or outside the
    /// file bounds.
    pub fn tensor_data_range(&self, name: &str) -> Result<std::ops::Range<u64>, GgufError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_owned()))?;
        let byte_len = tensor.byte_len()?;
        let start = self
            .tensor_data_offset
            .checked_add(tensor.offset())
            .ok_or(GgufError::TensorByteLengthOverflow)?;
        let end = start
            .checked_add(byte_len)
            .ok_or(GgufError::TensorByteLengthOverflow)?;
        if end > self.file_len {
            return Err(GgufError::TensorOffsetOutOfBounds {
                name: tensor.name().to_owned(),
                offset: tensor.offset(),
            });
        }
        Ok(start..end)
    }

    #[must_use]
    pub const fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }
}

/// A GGUF implementation of the core weight-loader boundary. It validates the
/// GGUF directory and returns artifact metadata; it does not claim to decode
/// tensors or provide an execution-ready model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufWeightLoader {
    path: PathBuf,
    description: WeightDescription,
}

impl GgufWeightLoader {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, description: WeightDescription) -> Self {
        Self {
            path: path.into(),
            description,
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Parse the directory independently when callers need tensor metadata.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the file is not a valid supported GGUF.
    pub fn inspect(&self) -> Result<GgufFile, GgufError> {
        GgufFile::open(self.path.clone())
    }
}

impl WeightLoader for GgufWeightLoader {
    fn load(&self) -> Result<WeightArtifact, ModelLoadError> {
        let file = self
            .inspect()
            .map_err(|error| ModelLoadError::InvalidArtifact {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        if file.tensor_count() == 0 {
            return Err(ModelLoadError::InvalidArtifact {
                path: self.path.clone(),
                message: "GGUF contains no tensors".to_owned(),
            });
        }
        let byte_len = fs::metadata(&self.path)
            .map_err(|error| ModelLoadError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?
            .len();
        WeightArtifact::new(
            WeightSource::file(self.path.clone()),
            byte_len,
            self.description.clone(),
        )
        .ok_or_else(|| ModelLoadError::EmptyArtifact(self.path.clone()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufError {
    Io {
        path: PathBuf,
        message: String,
    },
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    UnexpectedEof,
    InvalidUtf8,
    InvalidValueType(u32),
    InvalidBool(u8),
    InvalidLength {
        kind: &'static str,
        value: u64,
    },
    CountOverflow(&'static str),
    ArrayNestingTooDeep,
    DuplicateMetadata(String),
    InvalidDimensionCount(u32),
    InvalidAlignment(u64),
    Truncated,
    TensorOffsetOutOfBounds {
        name: String,
        offset: u64,
    },
    MissingTensor(String),
    UnsupportedTensorType(u32),
    ElementCountOverflow,
    TensorByteLengthOverflow,
    QuantizationBlockMismatch {
        name: String,
        elements: u64,
        block_elements: u64,
    },
}

impl GgufError {
    fn io(path: &Path, error: &io::Error) -> Self {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            Self::UnexpectedEof
        } else {
            Self::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        }
    }
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(f, "could not read GGUF {}: {message}", path.display())
            }
            Self::InvalidMagic(value) => write!(f, "invalid GGUF magic 0x{value:08x}"),
            Self::UnsupportedVersion(value) => write!(f, "unsupported GGUF version {value}"),
            Self::UnexpectedEof => f.write_str("truncated GGUF value"),
            Self::InvalidUtf8 => f.write_str("GGUF string is not valid UTF-8"),
            Self::InvalidValueType(value) => write!(f, "invalid GGUF metadata value type {value}"),
            Self::InvalidBool(value) => write!(f, "invalid GGUF boolean value {value}"),
            Self::InvalidLength { kind, value } => {
                write!(f, "GGUF {kind} length {value} exceeds the parser bound")
            }
            Self::CountOverflow(kind) => write!(f, "GGUF {kind} count does not fit this platform"),
            Self::ArrayNestingTooDeep => f.write_str("GGUF metadata array nesting is too deep"),
            Self::DuplicateMetadata(key) => write!(f, "duplicate GGUF metadata key {key:?}"),
            Self::InvalidDimensionCount(value) => {
                write!(f, "invalid GGUF tensor dimension count {value}")
            }
            Self::InvalidAlignment(value) => {
                write!(f, "invalid GGUF tensor-data alignment {value}")
            }
            Self::Truncated => f.write_str("GGUF tensor directory points past the file"),
            Self::TensorOffsetOutOfBounds { name, offset } => {
                write!(f, "GGUF tensor {name:?} has out-of-bounds offset {offset}")
            }
            Self::MissingTensor(name) => write!(f, "GGUF tensor {name:?} was not found"),
            Self::UnsupportedTensorType(value) => {
                write!(f, "unsupported GGML tensor type {value}")
            }
            Self::ElementCountOverflow => f.write_str("GGUF tensor element count overflowed"),
            Self::TensorByteLengthOverflow => f.write_str("GGUF tensor byte length overflowed"),
            Self::QuantizationBlockMismatch {
                name,
                elements,
                block_elements,
            } => write!(
                f,
                "GGUF tensor {name:?} has {elements} elements, not divisible by block size {block_elements}"
            ),
        }
    }
}

impl std::error::Error for GgufError {}

fn tensor_layout(value_type: u32) -> Result<(u64, u64), GgufError> {
    match value_type {
        0 | 26 => Ok((1, 4)),
        1 | 25 | 30 => Ok((1, 2)),
        2 | 20 => Ok((32, 18)),
        3 => Ok((32, 20)),
        6 => Ok((32, 22)),
        7 => Ok((32, 24)),
        8 => Ok((32, 34)),
        9 => Ok((32, 36)),
        10 => Ok((256, 84)),
        11 => Ok((256, 110)),
        12 => Ok((256, 144)),
        13 => Ok((256, 176)),
        14 => Ok((256, 210)),
        15 => Ok((256, 292)),
        16 | 35 => Ok((256, 66)),
        17 => Ok((256, 74)),
        18 => Ok((256, 98)),
        19 => Ok((256, 50)),
        21 => Ok((256, 106)),
        22 => Ok((256, 82)),
        23 => Ok((256, 136)),
        24 => Ok((1, 1)),
        27 | 28 => Ok((1, 8)),
        29 => Ok((256, 56)),
        34 => Ok((256, 54)),
        39 => Ok((32, 17)),
        40 => Ok((64, 36)),
        41 => Ok((128, 18)),
        42 => Ok((64, 18)),
        other => Err(GgufError::UnsupportedTensorType(other)),
    }
}

fn read_value<R: Read>(
    reader: &mut R,
    path: &Path,
    value_type: u32,
    depth: u32,
) -> Result<MetadataValue, GgufError> {
    if depth > MAX_METADATA_DEPTH {
        return Err(GgufError::ArrayNestingTooDeep);
    }
    match value_type {
        0 => Ok(MetadataValue::U8(read_u8(reader, path)?)),
        1 => Ok(MetadataValue::I8(read_u8(reader, path)?.cast_signed())),
        2 => Ok(MetadataValue::U16(read_u16(reader, path)?)),
        3 => Ok(MetadataValue::I16(read_u16(reader, path)?.cast_signed())),
        4 => Ok(MetadataValue::U32(read_u32(reader, path)?)),
        5 => Ok(MetadataValue::I32(read_u32(reader, path)?.cast_signed())),
        6 => Ok(MetadataValue::F32(f32::from_bits(read_u32(reader, path)?))),
        7 => {
            let value = read_u8(reader, path)?;
            match value {
                0 => Ok(MetadataValue::Bool(false)),
                1 => Ok(MetadataValue::Bool(true)),
                _ => Err(GgufError::InvalidBool(value)),
            }
        }
        8 => Ok(MetadataValue::String(read_string(reader, path)?)),
        9 => {
            let element_type = read_u32(reader, path)?;
            let count = read_count(reader, path, "array", MAX_ARRAY_ITEMS)?;
            let mut values = Vec::with_capacity(
                usize::try_from(count).map_err(|_| GgufError::CountOverflow("array"))?,
            );
            for _ in 0..count {
                values.push(read_value(reader, path, element_type, depth + 1)?);
            }
            Ok(MetadataValue::Array(values))
        }
        10 => Ok(MetadataValue::U64(read_u64(reader, path)?)),
        11 => Ok(MetadataValue::I64(read_u64(reader, path)?.cast_signed())),
        12 => Ok(MetadataValue::F64(f64::from_bits(read_u64(reader, path)?))),
        other => Err(GgufError::InvalidValueType(other)),
    }
}

fn read_count<R: Read>(
    reader: &mut R,
    path: &Path,
    kind: &'static str,
    max: u64,
) -> Result<u64, GgufError> {
    let count = read_u64(reader, path)?;
    if count > max {
        return Err(GgufError::InvalidLength { kind, value: count });
    }
    Ok(count)
}

fn read_string<R: Read>(reader: &mut R, path: &Path) -> Result<String, GgufError> {
    let length = read_u64(reader, path)?;
    if length > MAX_STRING_BYTES {
        return Err(GgufError::InvalidLength {
            kind: "string",
            value: length,
        });
    }
    let length = usize::try_from(length).map_err(|_| GgufError::CountOverflow("string"))?;
    let mut bytes = vec![0; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| GgufError::io(path, &error))?;
    String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8)
}

fn read_u8<R: Read>(reader: &mut R, path: &Path) -> Result<u8, GgufError> {
    let mut bytes = [0; 1];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| GgufError::io(path, &error))?;
    Ok(bytes[0])
}

fn read_u16<R: Read>(reader: &mut R, path: &Path) -> Result<u16, GgufError> {
    let mut bytes = [0; 2];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| GgufError::io(path, &error))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32<R: Read>(reader: &mut R, path: &Path) -> Result<u32, GgufError> {
    let mut bytes = [0; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| GgufError::io(path, &error))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R, path: &Path) -> Result<u64, GgufError> {
    let mut bytes = [0; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| GgufError::io(path, &error))?;
    Ok(u64::from_le_bytes(bytes))
}

fn align(value: u64, alignment: u64) -> Result<u64, GgufError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(GgufError::Truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use engine_core::{
        FileModelProvider, ModelCapabilities, ModelDescription, ModelId, ModelRegion,
        ModelRegionId, ModelRegionKind, Quantization, WeightFormat,
    };

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend(value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend(value.as_bytes());
    }

    fn fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, GGUF_MAGIC);
        push_u32(&mut bytes, GGUF_VERSION);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 3);

        push_string(&mut bytes, "general.architecture");
        push_u32(&mut bytes, 8);
        push_string(&mut bytes, "qwen3");
        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 32);
        push_string(&mut bytes, "test.array");
        push_u32(&mut bytes, 9);
        push_u32(&mut bytes, 4);
        push_u64(&mut bytes, 2);
        push_u32(&mut bytes, 1);
        push_u32(&mut bytes, 0);

        push_string(&mut bytes, "token_embd.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 4);
        push_u64(&mut bytes, 8);
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 0);
        bytes.resize(bytes.len() + 256, 0);
        bytes
    }

    #[test]
    fn parses_metadata_and_tensor_directory_without_loading_tensor_data() {
        let mut reader = Cursor::new(fixture());
        let parsed =
            GgufFile::read_from(&mut reader, PathBuf::from("fixture.gguf")).expect("valid fixture");
        assert_eq!(parsed.version(), 3);
        assert_eq!(parsed.tensor_count(), 1);
        assert_eq!(
            parsed
                .metadata("general.architecture")
                .and_then(MetadataValue::as_str),
            Some("qwen3")
        );
        assert_eq!(
            parsed
                .metadata("general.alignment")
                .and_then(MetadataValue::as_u64),
            Some(32)
        );
        assert_eq!(parsed.tensors()[0].name(), "token_embd.weight");
        assert_eq!(parsed.tensors()[0].dimensions(), &[4, 8]);
        assert_eq!(parsed.tensors()[0].value_type(), 1);
        assert_eq!(parsed.tensors()[0].offset(), 0);
        let range = parsed
            .tensor_data_range("token_embd.weight")
            .expect("tensor range");
        assert_eq!(range.end - range.start, 64);
    }

    #[test]
    fn computes_quantized_tensor_block_sizes() {
        let tensor = TensorInfo {
            name: "q4".to_owned(),
            dimensions: vec![256],
            value_type: 12,
            offset: 0,
        };
        assert_eq!(tensor.element_count().expect("element count"), 256);
        assert_eq!(tensor.byte_len().expect("Q4_K byte length"), 144);

        let iq_tensor = TensorInfo {
            name: "iq4_xs".to_owned(),
            dimensions: vec![256],
            value_type: 23,
            offset: 0,
        };
        assert_eq!(iq_tensor.byte_len().expect("IQ4_XS byte length"), 136);
    }

    #[test]
    fn rejects_invalid_magic_and_duplicate_metadata() {
        let mut invalid = fixture();
        invalid[0] = 0;
        let mut reader = Cursor::new(invalid);
        assert!(matches!(
            GgufFile::read_from(&mut reader, PathBuf::from("fixture.gguf")),
            Err(GgufError::InvalidMagic(_))
        ));

        let mut duplicate = fixture();
        duplicate[16..24].copy_from_slice(&4u64.to_le_bytes());
        let mut duplicate_entry = Vec::new();
        push_string(&mut duplicate_entry, "general.architecture");
        push_u32(&mut duplicate_entry, 8);
        push_string(&mut duplicate_entry, "other");
        let marker = b"token_embd.weight";
        let position = duplicate
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("tensor marker")
            .checked_sub(8)
            .expect("tensor name length");
        duplicate.splice(position..position, duplicate_entry);
        let mut reader = Cursor::new(duplicate);
        assert!(matches!(
            GgufFile::read_from(&mut reader, PathBuf::from("fixture.gguf")),
            Err(GgufError::DuplicateMetadata(key)) if key == "general.architecture"
        ));
    }

    #[test]
    fn gguf_loader_composes_with_core_file_provider() {
        let path =
            std::env::temp_dir().join(format!("engine-gguf-{}-{}.gguf", std::process::id(), 1));
        std::fs::write(&path, fixture()).expect("write fixture");
        let weights =
            engine_core::WeightDescription::new(WeightFormat::Gguf, Quantization::GgufQ4Km);
        let loader = GgufWeightLoader::new(&path, weights.clone());
        let model = ModelDescription::new(
            ModelId::new("test-model").expect("model ID"),
            "qwen3",
            vec![ModelRegion::new(
                ModelRegionId::new(0),
                ModelRegionKind::Embedding,
            )],
            Vec::new(),
            ModelCapabilities::new(None, false),
            weights,
        )
        .expect("model description");
        let provider = FileModelProvider::load(model, &loader).expect("provider");
        assert_eq!(provider.artifact().source().as_path(), path);
        std::fs::remove_file(path).expect("remove fixture");
    }
}
