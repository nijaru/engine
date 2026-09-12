//! Qwen model metadata, state requirements, and weight bindings.
use crate::{QwenConfig, QwenLayerKind};
use engine_core::{
    ConvolutionStateShape, DataType, DeviceId, KvStateSpec, ModelCapabilities, ModelDescription,
    ModelError, ModelId, ModelProvider, ModelRegion, ModelRegionId, ModelRegionKind, MtpCapability,
    Quantization, RecurrentMatrixShape, RecurrentStateSpec, StateRequirement, WeightBinding,
    WeightDescription, WeightFormat,
};
use engine_gguf::{GgufError, GgufFile, MetadataValue, TensorDataReader};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

fn config_from_gguf(file: &GgufFile) -> Result<QwenConfig, GgufError> {
    let architecture = file
        .metadata("general.architecture")
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| GgufError::MissingMetadata("general.architecture".to_owned()))?;
    if architecture != "qwen35" {
        return Err(GgufError::UnsupportedArchitecture(architecture.to_owned()));
    }
    let config = QwenConfig {
        context_length: required_u64(file, "qwen35.context_length")?,
        embedding_length: required_u32(file, "qwen35.embedding_length")?,
        feed_forward_length: required_u32(file, "qwen35.feed_forward_length")?,
        block_count: required_u32(file, "qwen35.block_count")?,
        attention_heads: required_u32(file, "qwen35.attention.head_count")?,
        kv_heads: required_u32(file, "qwen35.attention.head_count_kv")?,
        key_length: required_u32(file, "qwen35.attention.key_length")?,
        value_length: required_u32(file, "qwen35.attention.value_length")?,
        full_attention_interval: required_u32(file, "qwen35.full_attention_interval")?,
        ssm_group_count: required_u32(file, "qwen35.ssm.group_count")?,
        ssm_inner_size: required_u32(file, "qwen35.ssm.inner_size")?,
        ssm_state_size: required_u32(file, "qwen35.ssm.state_size")?,
        ssm_time_step_rank: required_u32(file, "qwen35.ssm.time_step_rank")?,
        ssm_conv_kernel: required_u32(file, "qwen35.ssm.conv_kernel")?,
        nextn_predict_layers: required_u32(file, "qwen35.nextn_predict_layers")?,
    };
    config
        .validate()
        .map_err(|error| GgufError::InvalidModelConfiguration(error.reason()))?;
    Ok(config)
}

fn required_u64(file: &GgufFile, key: &str) -> Result<u64, GgufError> {
    file.metadata(key)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?
        .as_u64()
        .ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
        })
}
fn required_u32(file: &GgufFile, key: &str) -> Result<u32, GgufError> {
    u32::try_from(required_u64(file, key)?).map_err(|_| GgufError::MetadataTypeMismatch {
        key: key.to_owned(),
    })
}

fn qwen35_model_description(
    config: &QwenConfig,
    kv_block_tokens: u32,
    model_id: ModelId,
    quantization: Quantization,
) -> Result<ModelDescription, GgufError> {
    let language_layers =
        config
            .language_layer_count()
            .ok_or(GgufError::InvalidModelConfiguration(
                "MTP layer count exceeds total block count",
            ))?;
    let full_layers = language_layers / config.full_attention_interval();
    let recurrent_layers = language_layers - full_layers;
    let full_spec = KvStateSpec::new(
        narrow_u16(full_layers, "full-attention layer count")?,
        narrow_u16(config.kv_heads(), "KV head count")?,
        narrow_u16(config.key_length(), "full-attention head dimension")?,
        kv_block_tokens,
        DataType::F16,
    )
    .map_err(|_| {
        GgufError::InvalidModelConfiguration("full-attention state dimensions are invalid")
    })?;
    let matrix = RecurrentMatrixShape::new(
        narrow_u16(config.ssm_time_step_rank(), "recurrent matrix count")?,
        narrow_u16(config.ssm_state_size(), "recurrent matrix row dimension")?,
        narrow_u16(
            config.ssm_inner_size() / config.ssm_time_step_rank(),
            "recurrent matrix column dimension",
        )?,
    )
    .ok_or(GgufError::InvalidModelConfiguration(
        "recurrent matrix dimensions are invalid",
    ))?;
    let convolution_channels = config
        .ssm_inner_size()
        .checked_add(
            config
                .ssm_group_count()
                .checked_mul(config.ssm_state_size())
                .and_then(|channels| channels.checked_mul(2))
                .ok_or(GgufError::InvalidModelConfiguration(
                    "recurrent convolution dimensions overflow",
                ))?,
        )
        .ok_or(GgufError::InvalidModelConfiguration(
            "recurrent convolution dimensions overflow",
        ))?;
    let convolution_history =
        config
            .ssm_conv_kernel()
            .checked_sub(1)
            .ok_or(GgufError::InvalidModelConfiguration(
                "recurrent convolution kernel has no history",
            ))?;
    let recurrent_spec = RecurrentStateSpec::new(
        narrow_u16(recurrent_layers, "recurrent layer count")?,
        matrix,
        ConvolutionStateShape::new(
            convolution_channels,
            narrow_u16(convolution_history, "recurrent convolution history")?,
        )
        .ok_or(GgufError::InvalidModelConfiguration(
            "recurrent convolution dimensions are invalid",
        ))?,
        DataType::F32,
        DataType::F32,
    )
    .map_err(|_| GgufError::InvalidModelConfiguration("recurrent state dimensions are invalid"))?;
    let mtp = MtpCapability::new(
        u8::try_from(config.nextn_predict_layers()).map_err(|_| {
            GgufError::InvalidModelConfiguration("MTP layer count exceeds the core capability")
        })?,
        u8::try_from(config.nextn_predict_layers()).map_err(|_| {
            GgufError::InvalidModelConfiguration("MTP token count exceeds the core capability")
        })?,
    );
    ModelDescription::new(
        model_id,
        "qwen3.8-hybrid",
        vec![
            ModelRegion::new(ModelRegionId::new(0), ModelRegionKind::Embedding),
            ModelRegion::new(ModelRegionId::new(1), ModelRegionKind::RecurrentAttention),
            ModelRegion::new(ModelRegionId::new(2), ModelRegionKind::FullAttention),
            ModelRegion::new(ModelRegionId::new(3), ModelRegionKind::FeedForward),
            ModelRegion::new(ModelRegionId::new(4), ModelRegionKind::OutputProjection),
        ],
        vec![
            StateRequirement::Recurrent(recurrent_spec),
            StateRequirement::FullAttentionKv(full_spec),
        ],
        ModelCapabilities::new(mtp, false),
        WeightDescription::new(WeightFormat::Gguf, quantization),
    )
    .map_err(|error| match error {
        ModelError::InvalidDescription(reason) => GgufError::InvalidModelConfiguration(reason),
        ModelError::PlanModelMismatch
        | ModelError::UnknownRegion(_)
        | ModelError::UndeclaredState(_) => {
            GgufError::InvalidModelConfiguration("unexpected model description validation failure")
        }
    })
}

fn narrow_u16(value: u32, name: &'static str) -> Result<u16, GgufError> {
    u16::try_from(value).map_err(|_| GgufError::InvalidModelConfiguration(name))
}

const QWEN35_RECURRENT_LAYER_TENSORS: &[&str] = &[
    "attn_gate.weight",
    "attn_norm.weight",
    "attn_qkv.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "ssm_a",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight",
];

const QWEN35_FULL_LAYER_TENSORS: &[&str] = &[
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
];

/// A provider for the text-only Qwen3.8 language artifact carried by GGUF.
/// It owns validated format metadata and the core model description; tensor
/// execution remains a separate backend/dispatcher responsibility.
/// Opening hashes the complete file in bounded memory to identify its contents.
/// The artifact must remain immutable while the provider and tensor readers live.
pub struct QwenGguf {
    file: GgufFile,
    config: QwenConfig,
    description: ModelDescription,
}

impl QwenGguf {
    /// Open and validate a Qwen3.8 GGUF artifact without materializing weights.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the artifact, model metadata, tensor ranges,
    /// or core model description is invalid.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, GgufError> {
        Self::open_with_kv_block_tokens(path, 16)
    }

    /// Open with an explicit full-attention KV allocation block size.
    /// This is a performance/layout choice, not a model semantic.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the block size, artifact, model metadata,
    /// tensor ranges, or core model description is invalid.
    pub fn open_with_kv_block_tokens(
        path: impl Into<PathBuf>,
        kv_block_tokens: u32,
    ) -> Result<Self, GgufError> {
        if kv_block_tokens == 0 {
            return Err(GgufError::InvalidModelConfiguration(
                "KV block size must be non-zero",
            ));
        }
        let file = GgufFile::open(path)?;
        let config = config_from_gguf(&file)?;
        for tensor in file.tensors() {
            file.tensor_data_range(tensor.name())?;
        }
        let identity = artifact_identity(file.path())?;
        let quantization = artifact_quantization(&file);
        let description =
            qwen35_model_description(&config, kv_block_tokens, identity, quantization)?;
        Ok(Self {
            file,
            config,
            description,
        })
    }

    #[must_use]
    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    #[must_use]
    pub const fn config(&self) -> &QwenConfig {
        &self.config
    }

    /// Open one validated tensor as a bounded canonical F32 stream. The
    /// provider still leaves physical allocation and execution to a backend.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the tensor is missing, unsupported, or its
    /// encoded range cannot be opened.
    pub fn open_tensor(&self, name: &str) -> Result<TensorDataReader, GgufError> {
        self.file.open_tensor(name)
    }

    /// Build a logical model/device binding for selected tensors without
    /// allocating or uploading their data. The backend materializes those
    /// streams and retains the physical buffers.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when a requested tensor cannot be opened or the
    /// names contain a duplicate.
    pub fn weight_binding(
        &self,
        device: DeviceId,
        names: &[&str],
    ) -> Result<WeightBinding, GgufError> {
        let specs = names
            .iter()
            .map(|name| self.open_tensor(name).map(|reader| reader.spec().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        WeightBinding::new(self.description.id().clone(), device, specs)
            .map_err(|error| GgufError::InvalidWeightBinding(error.to_string()))
    }

    /// Return the execution kind for one language layer. MTP blocks are not
    /// included in this accessor and are intentionally a separate capability.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::InvalidModelConfiguration`] when `layer` is not a
    /// language-layer index.
    pub fn layer_kind(&self, layer: u32) -> Result<QwenLayerKind, GgufError> {
        self.config
            .layer_kind(layer)
            .map_err(|error| GgufError::InvalidModelConfiguration(error.reason()))
    }

    /// Build a logical binding for all checkpoint tensors used by one language
    /// layer. This validates names and descriptors without allocating weight
    /// buffers or claiming that the layer is executable.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the layer index or any required tensor is
    /// invalid or missing.
    pub fn layer_weight_binding(
        &self,
        device: DeviceId,
        layer: u32,
    ) -> Result<WeightBinding, GgufError> {
        let kind = self.layer_kind(layer)?;
        let suffixes = match kind {
            QwenLayerKind::Recurrent => QWEN35_RECURRENT_LAYER_TENSORS,
            QwenLayerKind::FullAttention => QWEN35_FULL_LAYER_TENSORS,
        };
        let names = suffixes
            .iter()
            .map(|suffix| format!("blk.{layer}.{suffix}"))
            .collect::<Vec<_>>();
        let specs = names
            .iter()
            .map(|name| self.open_tensor(name).map(|reader| reader.spec().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        WeightBinding::new(self.description.id().clone(), device, specs)
            .map_err(|error| GgufError::InvalidWeightBinding(error.to_string()))
    }
}

impl ModelProvider for QwenGguf {
    fn description(&self) -> &ModelDescription {
        &self.description
    }
}

/// Hash the complete artifact in bounded memory, including actual weight bytes.
/// This deliberately pays a startup read instead of treating a model family or
/// file name as proof that two prepared artifacts contain the same weights.
fn artifact_identity(path: &Path) -> Result<ModelId, GgufError> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path).map_err(|error| GgufError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| GgufError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let hash = digest
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut text, byte| {
            use std::fmt::Write;
            write!(text, "{byte:02x}").expect("writing to String cannot fail");
            text
        });
    ModelId::new(format!("gguf:sha256:{hash}"))
        .map_err(|_| GgufError::InvalidModelConfiguration("invalid artifact identity"))
}

fn artifact_quantization(file: &GgufFile) -> Quantization {
    // GGUF file_type identifies the mixture recipe, not each tensor's encoding.
    // Unknown recipes remain explicit; per-tensor encodings are retained intact.
    quantization_for_recipe(
        file.metadata("general.file_type")
            .and_then(MetadataValue::as_u64),
    )
}

fn quantization_for_recipe(recipe: Option<u64>) -> Quantization {
    match recipe {
        Some(0 | 1 | 32) => Quantization::None,
        Some(15) => Quantization::GgufQ4Km,
        _ => Quantization::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn artifact_identity_tracks_bytes_instead_of_model_names() {
        let path =
            std::env::temp_dir().join(format!("engine-artifact-identity-{}", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let first = artifact_identity(&path).unwrap();
        assert_eq!(
            first.as_str(),
            "gguf:sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::write(&path, b"abd").unwrap();
        assert_ne!(first, artifact_identity(&path).unwrap());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn quantization_comes_from_artifact_recipe_metadata() {
        assert_eq!(quantization_for_recipe(None), Quantization::Other);
        for (recipe, expected) in [
            (15, Quantization::GgufQ4Km),
            (0, Quantization::None),
            (7, Quantization::Other),
        ] {
            assert_eq!(quantization_for_recipe(Some(recipe)), expected);
        }
    }

    #[test]
    fn builds_qwen35_description_with_distinct_state_families() {
        let mut config = QwenConfig {
            context_length: 262_144,
            embedding_length: 5120,
            feed_forward_length: 17_408,
            block_count: 65,
            attention_heads: 24,
            kv_heads: 4,
            key_length: 256,
            value_length: 256,
            full_attention_interval: 4,
            ssm_group_count: 16,
            ssm_inner_size: 6144,
            ssm_state_size: 128,
            ssm_time_step_rank: 48,
            ssm_conv_kernel: 4,
            nextn_predict_layers: 1,
        };
        let description = qwen35_model_description(
            &config,
            16,
            ModelId::new("test-artifact").unwrap(),
            Quantization::Other,
        )
        .expect("model description");
        assert_eq!(description.id().as_str(), "test-artifact");
        assert_eq!(description.regions().len(), 5);
        assert_eq!(description.state_requirements().len(), 2);
        let recurrent = description
            .state_requirements()
            .iter()
            .find_map(|requirement| match requirement {
                StateRequirement::Recurrent(spec) => Some(*spec),
                StateRequirement::FullAttentionKv(_) => None,
            })
            .expect("recurrent state");
        assert_eq!(recurrent.matrix().matrix_count(), 48);
        assert_eq!(recurrent.matrix().rows(), 128);
        assert_eq!(recurrent.matrix().columns(), 128);
        assert_eq!(recurrent.convolution().channels(), 10_240);
        assert_eq!(recurrent.convolution().history_tokens(), 3);
        assert!(
            description
                .state_requirements()
                .iter()
                .any(|requirement| matches!(requirement, StateRequirement::FullAttentionKv(_)))
        );
        assert!(
            description
                .state_requirements()
                .iter()
                .any(|requirement| matches!(requirement, StateRequirement::Recurrent(_)))
        );
        assert_eq!(
            description
                .capabilities()
                .mtp()
                .expect("MTP")
                .max_draft_tokens(),
            1
        );
        // The group product fits u32, but doubling it must fail without panic.
        config.ssm_group_count = 1 << 24;
        config.validate().unwrap();
        assert!(matches!(
            qwen35_model_description(
                &config,
                16,
                ModelId::new("overflow").unwrap(),
                Quantization::Other
            ),
            Err(GgufError::InvalidModelConfiguration(
                "recurrent convolution dimensions overflow"
            ))
        ));
    }
}

#[cfg(test)]
mod decoding_tests {
    use super::*;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }

    #[test]
    fn gguf_keys_normalize_into_the_model_definition() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.extend(17_u64.to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend(8_u32.to_le_bytes());
        string(&mut bytes, "qwen35");
        for (key, value) in [
            ("context_length", 262_144_u64),
            ("embedding_length", 5120),
            ("feed_forward_length", 17_408),
            ("block_count", 65),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 4),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
            ("full_attention_interval", 4),
            ("ssm.group_count", 16),
            ("ssm.inner_size", 6144),
            ("ssm.state_size", 128),
            ("ssm.time_step_rank", 48),
            ("ssm.conv_kernel", 4),
            ("nextn_predict_layers", 1),
            ("unused_field", 0),
        ] {
            string(&mut bytes, &format!("qwen35.{key}"));
            bytes.extend(10_u32.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        let path =
            std::env::temp_dir().join(format!("ribn-qwen-config-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let config = config_from_gguf(&file).unwrap();
        assert_eq!(config.embedding_length, 5120);
        assert_eq!(config.context_length, 262_144);
        assert_eq!(config.layer_kind(3), Ok(QwenLayerKind::FullAttention));
        std::fs::remove_file(path).unwrap();
    }
}
