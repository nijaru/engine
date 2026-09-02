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

mod iq3_s;

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
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I8(value) => Some(i32::from(*value)),
            Self::I16(value) => Some(i32::from(*value)),
            Self::I32(value) => Some(*value),
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

    /// Open a bounded streaming reader for one encoded tensor. This keeps bulk
    /// weights out of the metadata object and lets a backend choose its own
    /// chunking or mapping strategy.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor range is invalid or the file cannot be
    /// opened at that range.
    pub fn open_tensor(&self, name: &str) -> Result<TensorDataReader, GgufError> {
        let range = self.tensor_data_range(name)?;
        let mut file = File::open(&self.path).map_err(|error| GgufError::io(&self.path, &error))?;
        file.seek(SeekFrom::Start(range.start))
            .map_err(|error| GgufError::io(&self.path, &error))?;
        Ok(TensorDataReader {
            path: self.path.clone(),
            file,
            remaining: range.end - range.start,
        })
    }

    #[must_use]
    pub const fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Decode and validate the Qwen3.8 hybrid dimensions carried by this
    /// artifact. This remains a GGUF adapter; it does not create model kernels.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the architecture marker, metadata, or hybrid
    /// dimensions are missing or inconsistent.
    pub fn qwen35_config(&self) -> Result<Qwen35Config, GgufError> {
        let architecture = self
            .metadata("general.architecture")
            .and_then(MetadataValue::as_str)
            .ok_or_else(|| GgufError::MissingMetadata("general.architecture".to_owned()))?;
        if architecture != "qwen35" {
            return Err(GgufError::UnsupportedArchitecture(architecture.to_owned()));
        }
        let config = Qwen35Config::from_metadata(&self.metadata)?;
        config.validate()?;
        Ok(config)
    }

    /// Extract the embedded GPT-2/BPE vocabulary and merge table.
    ///
    /// The vocabulary is owned by the returned value so callers can build
    /// tokenizer lookup tables without retaining the parsed GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::MissingMetadata`], [`GgufError::MetadataTypeMismatch`],
    /// or [`GgufError::InvalidTokenizer`] when tokenizer metadata is absent or
    /// internally inconsistent.
    pub fn tokenizer(&self) -> Result<GgufTokenizer, GgufError> {
        GgufTokenizer::from_metadata(&self.metadata)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTokenizer {
    model: String,
    pretokenizer: String,
    tokens: Vec<String>,
    merges: Vec<String>,
    token_types: Vec<i32>,
    bos_token_id: u32,
    eos_token_id: u32,
    padding_token_id: Option<u32>,
}

impl GgufTokenizer {
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    #[must_use]
    pub fn pretokenizer(&self) -> &str {
        &self.pretokenizer
    }

    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    #[must_use]
    pub fn merges(&self) -> &[String] {
        &self.merges
    }

    #[must_use]
    pub fn token_types(&self) -> &[i32] {
        &self.token_types
    }

    #[must_use]
    pub const fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    #[must_use]
    pub const fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    #[must_use]
    pub const fn padding_token_id(&self) -> Option<u32> {
        self.padding_token_id
    }

    fn from_metadata(metadata: &BTreeMap<String, MetadataValue>) -> Result<Self, GgufError> {
        let tokens = required_string_array(metadata, "tokenizer.ggml.tokens")?;
        let merges = required_string_array(metadata, "tokenizer.ggml.merges")?;
        let token_types = required_i32_array(metadata, "tokenizer.ggml.token_type")?;
        if tokens.is_empty() || tokens.len() != token_types.len() {
            return Err(GgufError::InvalidTokenizer(
                "token and token-type arrays must have the same nonzero length",
            ));
        }
        Ok(Self {
            model: required_string(metadata, "tokenizer.ggml.model")?,
            pretokenizer: required_string(metadata, "tokenizer.ggml.pre")?,
            tokens,
            merges,
            token_types,
            bos_token_id: required_u32(metadata, "tokenizer.ggml.bos_token_id")?,
            eos_token_id: required_u32(metadata, "tokenizer.ggml.eos_token_id")?,
            padding_token_id: optional_u32(metadata, "tokenizer.ggml.padding_token_id")?,
        })
    }
}

/// Sequential reader over one encoded tensor payload. The reader stops exactly
/// at the tensor's encoded byte length and cannot consume the next tensor.
pub struct TensorDataReader {
    path: PathBuf,
    file: File,
    remaining: u64,
}

impl TensorDataReader {
    /// Read and dequantize the next complete block in this tensor payload.
    /// `None` means the bounded tensor range is exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::UnsupportedTensorType`] when no decoder exists,
    /// [`GgufError::TensorPayloadTruncated`] when the remaining range is not a
    /// complete block, or an I/O/dequantization error.
    pub fn read_dequantized_block(
        &mut self,
        value_type: u32,
    ) -> Result<Option<Vec<f32>>, GgufError> {
        let (_, block_bytes) = tensor_layout(value_type)?;
        if self.remaining == 0 {
            return Ok(None);
        }
        let block_bytes_u64 = block_bytes;
        if self.remaining < block_bytes_u64 {
            return Err(GgufError::TensorPayloadTruncated {
                value_type,
                expected: block_bytes_u64,
                remaining: self.remaining,
            });
        }
        let block_bytes =
            usize::try_from(block_bytes).map_err(|_| GgufError::TensorByteLengthOverflow)?;
        let mut encoded = vec![0; block_bytes];
        self.read_exact(&mut encoded)
            .map_err(|error| GgufError::io(&self.path, &error))?;
        dequantize_block(value_type, &encoded).map(Some)
    }

    #[must_use]
    pub const fn remaining(&self) -> u64 {
        self.remaining
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Read for TensorDataReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 || buffer.is_empty() {
            return Ok(0);
        }
        let requested = usize::try_from(self.remaining)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        let read = self.file.read(&mut buffer[..requested])?;
        self.remaining -= read as u64;
        Ok(read)
    }
}

/// Dequantize one complete GGML block into scalar `f32` values.
///
/// This deliberately operates on one block rather than allocating an entire
/// model tensor. Callers can stream or map a tensor and invoke it for each
/// block. The implementation covers the types present in the scalar and
/// common K-quant portions of the pinned Qwen3.8 artifact; unsupported IQ
/// variants remain explicit errors until their reference layouts are ported.
///
/// # Errors
///
/// Returns [`GgufError::UnsupportedTensorType`] for a type without a decoder,
/// or [`GgufError::InvalidTensorBlockLength`] when `encoded` is not exactly one
/// block for `value_type`.
pub fn dequantize_block(value_type: u32, encoded: &[u8]) -> Result<Vec<f32>, GgufError> {
    let (_, block_bytes) = tensor_layout(value_type)?;
    let expected = usize::try_from(block_bytes).map_err(|_| GgufError::TensorByteLengthOverflow)?;
    if encoded.len() != expected {
        return Err(GgufError::InvalidTensorBlockLength {
            value_type,
            expected,
            actual: encoded.len(),
        });
    }
    match value_type {
        0 => Ok(vec![f32::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3],
        ])]),
        1 => Ok(vec![f16_to_f32(u16::from_le_bytes([
            encoded[0], encoded[1],
        ]))]),
        8 => Ok(dequantize_q8_0(encoded)),
        11 => Ok(dequantize_q3_k(encoded)),
        12 => Ok(dequantize_q4_k(encoded)),
        13 => Ok(dequantize_q5_k(encoded)),
        14 => Ok(dequantize_q6_k(encoded)),
        20 => Ok(dequantize_iq4_nl(encoded)),
        21 => Ok(iq3_s::dequantize_block(encoded)),
        23 => Ok(dequantize_iq4_xs(encoded)),
        _ => Err(GgufError::UnsupportedTensorType(value_type)),
    }
}

const QK_K: usize = 256;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    match exponent {
        0 => sign * f32::from(fraction) * 2_f32.powi(-24),
        0x1f if fraction == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        exponent => {
            sign * (1.0 + f32::from(fraction) / 1024.0) * 2_f32.powi(i32::from(exponent) - 15)
        }
    }
}

fn read_f16(encoded: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([encoded[offset], encoded[offset + 1]]))
}

fn signed_scale(value: u8) -> i8 {
    value.wrapping_sub(32).cast_signed()
}

fn q4_k_scales(encoded: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0; 8];
    let mut minima = [0; 8];
    for index in 0..4 {
        let low_scale = encoded[index] & 0x3f;
        let high_scale = (encoded[8 + index] & 0x0f) | ((encoded[index] >> 2) & 0x30);
        let low_min = encoded[4 + index] & 0x3f;
        let high_min = (encoded[8 + index] >> 4) | ((encoded[4 + index] >> 2) & 0x30);
        scales[index] = low_scale;
        scales[index + 4] = high_scale;
        minima[index] = low_min;
        minima[index + 4] = high_min;
    }
    (scales, minima)
}

fn dequantize_q8_0(encoded: &[u8]) -> Vec<f32> {
    let scale = read_f16(encoded, 0);
    (0..32)
        .map(|index| scale * f32::from(i8::from_le_bytes([encoded[2 + index]])))
        .collect()
}

fn dequantize_q4_k(encoded: &[u8]) -> Vec<f32> {
    let scale = read_f16(encoded, 0);
    let minimum_scale = read_f16(encoded, 2);
    let (scales, minima) = q4_k_scales(&encoded[4..16]);
    let mut output = Vec::with_capacity(QK_K);
    for group in 0..8 {
        let data_offset = 16 + (group / 2) * 32;
        let shift = (group % 2) * 4;
        let group_scale = scale * f32::from(scales[group]);
        let group_minimum = minimum_scale * f32::from(minima[group]);
        for index in 0..32 {
            let quantized = (encoded[data_offset + index] >> shift) & 0x0f;
            output.push(group_scale * f32::from(quantized) - group_minimum);
        }
    }
    output
}

fn dequantize_q5_k(encoded: &[u8]) -> Vec<f32> {
    let scale = read_f16(encoded, 0);
    let minimum_scale = read_f16(encoded, 2);
    let (scales, minima) = q4_k_scales(&encoded[4..16]);
    let high_bits = &encoded[16..48];
    let low_bits = &encoded[48..];
    let mut output = Vec::with_capacity(QK_K);
    for group in 0..8 {
        let data_offset = (group / 2) * 32;
        let shift = (group % 2) * 4;
        let group_scale = scale * f32::from(scales[group]);
        let group_minimum = minimum_scale * f32::from(minima[group]);
        for index in 0..32 {
            let low = (low_bits[data_offset + index] >> shift) & 0x0f;
            let high = (high_bits[index] >> group) & 1;
            output.push(group_scale * f32::from(low | (high << 4)) - group_minimum);
        }
    }
    output
}

fn dequantize_q6_k(encoded: &[u8]) -> Vec<f32> {
    let low_bits = &encoded[..128];
    let high_bits = &encoded[128..192];
    let scales = &encoded[192..208];
    let scale = read_f16(encoded, 208);
    let mut output = Vec::with_capacity(QK_K);
    for group in 0..8 {
        let chunk = group / 4;
        let variant = group % 4;
        let low_offset = chunk * 64 + (variant % 2) * 32;
        let low_shift = if variant < 2 { 0 } else { 4 };
        let high_offset = chunk * 32;
        let high_shift = variant % 4;
        for half in 0..2 {
            let scale_index = group * 2 + half;
            let group_scale = scale * f32::from(i8::from_le_bytes([scales[scale_index]]));
            for index in 0..16 {
                let position = half * 16 + index;
                let low = (low_bits[low_offset + position] >> low_shift) & 0x0f;
                let high = (high_bits[high_offset + position] >> (high_shift * 2)) & 0x03;
                let quantized = i16::from(low | (high << 4)) - 32;
                output.push(group_scale * f32::from(quantized));
            }
        }
    }
    output
}

fn dequantize_q3_k(encoded: &[u8]) -> Vec<f32> {
    let high_bits = &encoded[..32];
    let low_bits = &encoded[32..96];
    let packed_scales = &encoded[96..108];
    let scale = read_f16(encoded, 108);
    let mut scales = [0_i8; 16];
    for index in 0..8 {
        let low_scale = (packed_scales[index] & 0x0f)
            | (((packed_scales[8 + index % 4] >> ((index / 4) * 2)) & 0x03) << 4);
        let high_slot = index + 8;
        let high_scale = (packed_scales[index] >> 4)
            | (((packed_scales[8 + high_slot % 4] >> ((high_slot / 4) * 2)) & 0x03) << 4);
        scales[index] = signed_scale(low_scale);
        scales[index + 8] = signed_scale(high_scale);
    }
    let mut output = Vec::with_capacity(QK_K);
    for (group, &packed_scale) in scales.iter().enumerate() {
        let chunk = group / 8;
        let variant = (group / 2) % 4;
        let half = group % 2;
        let low_offset = chunk * 32 + half * 16;
        let high_offset = half * 16;
        let group_scale = scale * f32::from(packed_scale);
        for index in 0..16 {
            let low = (low_bits[low_offset + index] >> (variant * 2)) & 0x03;
            let high = ((high_bits[high_offset + index] >> (group / 2)) & 1) ^ 1;
            let quantized = i16::from(low) - i16::from(high) * 4;
            output.push(group_scale * f32::from(quantized));
        }
    }
    output
}

fn dequantize_iq4_nl(encoded: &[u8]) -> Vec<f32> {
    const K_VALUES: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let scale = read_f16(encoded, 0);
    let mut output = Vec::with_capacity(32);
    for index in 0..16 {
        let byte = encoded[2 + index];
        output.push(scale * f32::from(K_VALUES[usize::from(byte & 0x0f)]));
    }
    for index in 0..16 {
        let byte = encoded[2 + index];
        output.push(scale * f32::from(K_VALUES[usize::from(byte >> 4)]));
    }
    output
}

fn dequantize_iq4_xs(encoded: &[u8]) -> Vec<f32> {
    const K_VALUES: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let scale = read_f16(encoded, 0);
    let high_scales = u16::from_le_bytes([encoded[2], encoded[3]]);
    let low_scales = &encoded[4..8];
    let quantized = &encoded[8..];
    let mut output = Vec::with_capacity(QK_K);
    for group in 0..8 {
        let low = (low_scales[group / 2] >> ((group % 2) * 4)) & 0x0f;
        let high = u8::try_from((high_scales >> (group * 2)) & 0x03)
            .expect("IQ4_XS high scale fits in two bits");
        let group_scale = scale * f32::from(signed_scale(low | (high << 4)));
        let data_offset = group * 16;
        for index in 0..16 {
            let byte = quantized[data_offset + index];
            output.push(group_scale * f32::from(K_VALUES[usize::from(byte & 0x0f)]));
        }
        for index in 0..16 {
            let byte = quantized[data_offset + index];
            output.push(group_scale * f32::from(K_VALUES[usize::from(byte >> 4)]));
        }
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35Config {
    context_length: u64,
    embedding_length: u32,
    feed_forward_length: u32,
    block_count: u32,
    attention_heads: u32,
    kv_heads: u32,
    key_length: u32,
    value_length: u32,
    full_attention_interval: u32,
    ssm_group_count: u32,
    ssm_inner_size: u32,
    ssm_state_size: u32,
    ssm_time_step_rank: u32,
    ssm_conv_kernel: u32,
    nextn_predict_layers: u32,
}

impl Qwen35Config {
    #[must_use]
    pub const fn context_length(&self) -> u64 {
        self.context_length
    }

    #[must_use]
    pub const fn embedding_length(&self) -> u32 {
        self.embedding_length
    }

    #[must_use]
    pub const fn feed_forward_length(&self) -> u32 {
        self.feed_forward_length
    }

    #[must_use]
    pub const fn block_count(&self) -> u32 {
        self.block_count
    }

    #[must_use]
    pub const fn attention_heads(&self) -> u32 {
        self.attention_heads
    }

    #[must_use]
    pub const fn kv_heads(&self) -> u32 {
        self.kv_heads
    }

    #[must_use]
    pub const fn key_length(&self) -> u32 {
        self.key_length
    }

    #[must_use]
    pub const fn value_length(&self) -> u32 {
        self.value_length
    }

    #[must_use]
    pub const fn full_attention_interval(&self) -> u32 {
        self.full_attention_interval
    }

    #[must_use]
    pub const fn ssm_group_count(&self) -> u32 {
        self.ssm_group_count
    }

    #[must_use]
    pub const fn ssm_inner_size(&self) -> u32 {
        self.ssm_inner_size
    }

    #[must_use]
    pub const fn ssm_state_size(&self) -> u32 {
        self.ssm_state_size
    }

    #[must_use]
    pub const fn ssm_time_step_rank(&self) -> u32 {
        self.ssm_time_step_rank
    }

    #[must_use]
    pub const fn ssm_conv_kernel(&self) -> u32 {
        self.ssm_conv_kernel
    }

    #[must_use]
    pub const fn nextn_predict_layers(&self) -> u32 {
        self.nextn_predict_layers
    }

    #[must_use]
    pub const fn language_layer_count(&self) -> Option<u32> {
        self.block_count.checked_sub(self.nextn_predict_layers)
    }

    fn from_metadata(metadata: &BTreeMap<String, MetadataValue>) -> Result<Self, GgufError> {
        Ok(Self {
            context_length: required_u64(metadata, "qwen35.context_length")?,
            embedding_length: required_u32(metadata, "qwen35.embedding_length")?,
            feed_forward_length: required_u32(metadata, "qwen35.feed_forward_length")?,
            block_count: required_u32(metadata, "qwen35.block_count")?,
            attention_heads: required_u32(metadata, "qwen35.attention.head_count")?,
            kv_heads: required_u32(metadata, "qwen35.attention.head_count_kv")?,
            key_length: required_u32(metadata, "qwen35.attention.key_length")?,
            value_length: required_u32(metadata, "qwen35.attention.value_length")?,
            full_attention_interval: required_u32(metadata, "qwen35.full_attention_interval")?,
            ssm_group_count: required_u32(metadata, "qwen35.ssm.group_count")?,
            ssm_inner_size: required_u32(metadata, "qwen35.ssm.inner_size")?,
            ssm_state_size: required_u32(metadata, "qwen35.ssm.state_size")?,
            ssm_time_step_rank: required_u32(metadata, "qwen35.ssm.time_step_rank")?,
            ssm_conv_kernel: required_u32(metadata, "qwen35.ssm.conv_kernel")?,
            nextn_predict_layers: required_u32(metadata, "qwen35.nextn_predict_layers")?,
        })
    }

    /// # Errors
    ///
    /// Returns [`GgufError::InvalidModelConfiguration`] when a required Qwen3.8
    /// hybrid dimension is zero or inconsistent.
    pub fn validate(&self) -> Result<(), GgufError> {
        let dimensions = [
            self.context_length,
            u64::from(self.embedding_length),
            u64::from(self.feed_forward_length),
            u64::from(self.block_count),
            u64::from(self.attention_heads),
            u64::from(self.kv_heads),
            u64::from(self.key_length),
            u64::from(self.value_length),
            u64::from(self.full_attention_interval),
            u64::from(self.ssm_group_count),
            u64::from(self.ssm_inner_size),
            u64::from(self.ssm_state_size),
            u64::from(self.ssm_time_step_rank),
            u64::from(self.ssm_conv_kernel),
        ];
        if dimensions.contains(&0) {
            return Err(GgufError::InvalidModelConfiguration(
                "Qwen3.8 dimensions must be non-zero",
            ));
        }
        let language_layers =
            self.language_layer_count()
                .ok_or(GgufError::InvalidModelConfiguration(
                    "MTP layer count exceeds total block count",
                ))?;
        if self.full_attention_interval > language_layers
            || language_layers % self.full_attention_interval != 0
            || !self.attention_heads.is_multiple_of(self.kv_heads)
            || self.ssm_time_step_rank.checked_mul(self.ssm_state_size) != Some(self.ssm_inner_size)
        {
            return Err(GgufError::InvalidModelConfiguration(
                "Qwen3.8 hybrid dimensions are inconsistent",
            ));
        }
        Ok(())
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
    MissingMetadata(String),
    MetadataTypeMismatch {
        key: String,
    },
    UnsupportedArchitecture(String),
    InvalidModelConfiguration(&'static str),
    InvalidTokenizer(&'static str),
    InvalidTensorBlockLength {
        value_type: u32,
        expected: usize,
        actual: usize,
    },
    TensorPayloadTruncated {
        value_type: u32,
        expected: u64,
        remaining: u64,
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
            Self::MissingMetadata(key) => write!(f, "missing GGUF metadata key {key:?}"),
            Self::MetadataTypeMismatch { key } => {
                write!(f, "GGUF metadata key {key:?} has an unexpected type")
            }
            Self::UnsupportedArchitecture(architecture) => {
                write!(f, "unsupported GGUF architecture {architecture:?}")
            }
            Self::InvalidModelConfiguration(reason) => {
                write!(f, "invalid Qwen3.8 model configuration: {reason}")
            }
            Self::InvalidTokenizer(reason) => {
                write!(f, "invalid GGUF tokenizer metadata: {reason}")
            }
            Self::InvalidTensorBlockLength {
                value_type,
                expected,
                actual,
            } => write!(
                f,
                "GGML tensor type {value_type} requires {expected} bytes per block, got {actual}"
            ),
            Self::TensorPayloadTruncated {
                value_type,
                expected,
                remaining,
            } => write!(
                f,
                "GGML tensor type {value_type} needs {expected} bytes for the next block, but only {remaining} remain"
            ),
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

fn required_string(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &'static str,
) -> Result<String, GgufError> {
    metadata
        .get(key)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
        })
}

fn required_string_array(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &'static str,
) -> Result<Vec<String>, GgufError> {
    let values = metadata
        .get(key)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
    let MetadataValue::Array(values) = values else {
        return Err(GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
        });
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: key.to_owned(),
                })
        })
        .collect()
}

fn required_i32_array(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &'static str,
) -> Result<Vec<i32>, GgufError> {
    let values = metadata
        .get(key)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
    let MetadataValue::Array(values) = values else {
        return Err(GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
        });
    };
    values
        .iter()
        .map(|value| {
            value
                .as_i32()
                .ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: key.to_owned(),
                })
        })
        .collect()
}

fn optional_u32(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &'static str,
) -> Result<Option<u32>, GgufError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: key.to_owned(),
                })
                .and_then(|value| {
                    u32::try_from(value).map_err(|_| GgufError::MetadataTypeMismatch {
                        key: key.to_owned(),
                    })
                })
        })
        .transpose()
}

fn required_u64(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &'static str,
) -> Result<u64, GgufError> {
    metadata
        .get(key)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?
        .as_u64()
        .ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
        })
}

fn required_u32(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &'static str,
) -> Result<u32, GgufError> {
    let value = required_u64(metadata, key)?;
    u32::try_from(value).map_err(|_| GgufError::MetadataTypeMismatch {
        key: key.to_owned(),
    })
}

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
        11 | 21 => Ok((256, 110)),
        12 => Ok((256, 144)),
        13 => Ok((256, 176)),
        14 => Ok((256, 210)),
        15 => Ok((256, 292)),
        16 | 35 => Ok((256, 66)),
        17 => Ok((256, 74)),
        18 => Ok((256, 98)),
        19 => Ok((256, 50)),
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
    fn streams_only_the_selected_tensor_payload() {
        let path = std::env::temp_dir().join(format!("engine-gguf-{}-2.gguf", std::process::id()));
        std::fs::write(&path, fixture()).expect("write fixture");
        let parsed = GgufFile::open(&path).expect("open fixture");
        let mut reader = parsed
            .open_tensor("token_embd.weight")
            .expect("tensor reader");
        let mut payload = Vec::new();
        reader.read_to_end(&mut payload).expect("tensor payload");
        assert_eq!(payload.len(), 64);
        assert_eq!(reader.remaining(), 0);
        assert_eq!(reader.read(&mut [0; 1]).expect("bounded read"), 0);

        let mut block_reader = parsed
            .open_tensor("token_embd.weight")
            .expect("tensor block reader");
        for _ in 0..32 {
            let block = block_reader
                .read_dequantized_block(1)
                .expect("dequantized block")
                .expect("block present");
            assert_eq!(block.len(), 1);
            assert_eq!(block[0].to_bits(), 0.0_f32.to_bits());
        }
        assert!(
            block_reader
                .read_dequantized_block(1)
                .expect("exhausted block reader")
                .is_none()
        );
        std::fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn dequantizes_supported_scalar_and_quantized_blocks() {
        let mut f32_block = [0_u8; 4];
        f32_block.copy_from_slice(&1.5_f32.to_le_bytes());
        assert_eq!(
            dequantize_block(0, &f32_block).expect("F32")[0].to_bits(),
            1.5_f32.to_bits()
        );

        let f16_block = 0x3c00_u16.to_le_bytes();
        assert_eq!(
            dequantize_block(1, &f16_block).expect("F16")[0].to_bits(),
            1.0_f32.to_bits()
        );

        let mut q8 = [0_u8; 34];
        q8[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q8[2] = 0x80;
        q8[3] = 0x7f;
        let decoded = dequantize_block(8, &q8).expect("Q8_0");
        assert_eq!(decoded[0].to_bits(), (-128.0_f32).to_bits());
        assert_eq!(decoded[1].to_bits(), 127.0_f32.to_bits());
        assert_eq!(decoded.len(), 32);

        let mut q4 = [0_u8; 144];
        q4[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q4[4..8].fill(1);
        q4[16] = 0x10;
        let decoded = dequantize_block(12, &q4).expect("Q4_K");
        assert_eq!(decoded[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(decoded[32].to_bits(), 1.0_f32.to_bits());
        assert_eq!(decoded.len(), 256);

        let mut q5 = [0_u8; 176];
        q5[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q5[4..8].fill(1);
        q5[16] = 1;
        q5[48] = 0x10;
        let decoded = dequantize_block(13, &q5).expect("Q5_K");
        assert_eq!(decoded[0].to_bits(), 16.0_f32.to_bits());
        assert_eq!(decoded[32].to_bits(), 1.0_f32.to_bits());
        assert_eq!(decoded.len(), 256);

        let mut q6 = [0_u8; 210];
        q6[192..208].fill(1);
        q6[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let decoded = dequantize_block(14, &q6).expect("Q6_K");
        assert!(
            decoded
                .iter()
                .all(|value| value.to_bits() == (-32.0_f32).to_bits())
        );

        let mut q3 = [0_u8; 110];
        q3[108..110].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let decoded = dequantize_block(11, &q3).expect("Q3_K");
        assert!(
            decoded
                .iter()
                .all(|value| value.to_bits() == 128.0_f32.to_bits())
        );

        let mut iq4_nl = [0_u8; 18];
        iq4_nl[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        iq4_nl[2] = 0x08;
        let decoded = dequantize_block(20, &iq4_nl).expect("IQ4_NL");
        assert_eq!(decoded[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(decoded[16].to_bits(), (-127.0_f32).to_bits());
        assert_eq!(decoded.len(), 32);

        let mut iq4_xs = [0_u8; 136];
        iq4_xs[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        iq4_xs[8] = 0x08;
        let decoded = dequantize_block(23, &iq4_xs).expect("IQ4_XS");
        assert_eq!(decoded[0].to_bits(), (-32.0_f32).to_bits());
        assert_eq!(decoded.len(), 256);

        let mut iq3_s = [0_u8; 110];
        iq3_s[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let decoded = dequantize_block(21, &iq3_s).expect("IQ3_S");
        assert!(
            decoded
                .iter()
                .all(|value| value.to_bits() == 1.0_f32.to_bits())
        );
        assert_eq!(decoded.len(), 256);
    }

    #[test]
    fn extracts_embedded_tokenizer_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            MetadataValue::String("gpt2".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.pre".to_owned(),
            MetadataValue::String("qwen35".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            MetadataValue::Array(vec![
                MetadataValue::String("a".to_owned()),
                MetadataValue::String("b".to_owned()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_owned(),
            MetadataValue::Array(vec![MetadataValue::String("a b".to_owned())]),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            MetadataValue::Array(vec![MetadataValue::I32(1), MetadataValue::I32(3)]),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_owned(),
            MetadataValue::U32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_owned(),
            MetadataValue::U32(2),
        );
        metadata.insert(
            "tokenizer.ggml.padding_token_id".to_owned(),
            MetadataValue::U32(0),
        );

        let tokenizer = GgufTokenizer::from_metadata(&metadata).expect("tokenizer metadata");
        assert_eq!(tokenizer.model(), "gpt2");
        assert_eq!(tokenizer.pretokenizer(), "qwen35");
        assert_eq!(tokenizer.tokens(), &["a", "b"]);
        assert_eq!(tokenizer.merges(), &["a b"]);
        assert_eq!(tokenizer.token_types(), &[1, 3]);
        assert_eq!(tokenizer.bos_token_id(), 1);
        assert_eq!(tokenizer.eos_token_id(), 2);
        assert_eq!(tokenizer.padding_token_id(), Some(0));
    }

    #[test]
    fn rejects_invalid_dequantization_blocks() {
        let error = dequantize_block(12, &[0; 143]).expect_err("short block");
        assert!(matches!(
            error,
            GgufError::InvalidTensorBlockLength {
                value_type: 12,
                expected: 144,
                actual: 143
            }
        ));
        assert!(matches!(
            dequantize_block(22, &[0; 82]),
            Err(GgufError::UnsupportedTensorType(22))
        ));
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

        let iq3_s_tensor = TensorInfo {
            name: "iq3_s".to_owned(),
            dimensions: vec![256],
            value_type: 21,
            offset: 0,
        };
        assert_eq!(iq3_s_tensor.byte_len().expect("IQ3_S byte length"), 110);
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
