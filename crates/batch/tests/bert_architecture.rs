use std::error::Error;
use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use ribn_batch::{BatchConfig, BatchExecutor, BatchRuntime, Job, JobOutput};
use ribn_foundation::{ParameterVersion, ScalarType};
use ribn_hf::{LocalModelPackage, LocalWeightSet, PackageError};
use ribn_safetensors::ArtifactError;

static NEXT_DIR: AtomicU64 = AtomicU64::new(1);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("ribn-bert-{}-{id}", process::id()));
        fs::create_dir(&path).expect("test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug)]
enum ModelError {
    InvalidConfig(String),
    Artifact(String),
    MissingParameter(String),
    UnsupportedTensor(String),
    InvalidInput(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid BERT config: {message}"),
            Self::Artifact(message) => write!(formatter, "weight artifact error: {message}"),
            Self::MissingParameter(name) => write!(formatter, "missing BERT parameter {name}"),
            Self::UnsupportedTensor(message) => {
                write!(formatter, "unsupported BERT tensor: {message}")
            }
            Self::InvalidInput(message) => write!(formatter, "invalid BERT input: {message}"),
        }
    }
}

impl Error for ModelError {}

#[derive(Clone, Debug, PartialEq)]
struct BertConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    layer_norm_eps: f32,
}

impl BertConfig {
    fn from_package(package: &LocalModelPackage) -> Result<Self, ModelError> {
        let config = package.config();
        if config["model_type"].as_str() != Some("bert") {
            return Err(ModelError::InvalidConfig(
                "model_type must be \"bert\"".to_owned(),
            ));
        }
        if config["hidden_act"].as_str().unwrap_or("gelu") != "gelu" {
            return Err(ModelError::InvalidConfig(
                "the reference path currently supports hidden_act=gelu".to_owned(),
            ));
        }
        if config["position_embedding_type"]
            .as_str()
            .unwrap_or("absolute")
            != "absolute"
        {
            return Err(ModelError::InvalidConfig(
                "the reference path currently supports absolute positions".to_owned(),
            ));
        }
        if config["is_decoder"].as_bool().unwrap_or(false)
            || config["add_cross_attention"].as_bool().unwrap_or(false)
        {
            return Err(ModelError::InvalidConfig(
                "decoder/cross-attention BERT is outside this encoder pressure test".to_owned(),
            ));
        }

        let parsed = Self {
            vocab_size: config_usize(config["vocab_size"].as_u64(), "vocab_size")?,
            hidden_size: config_usize(config["hidden_size"].as_u64(), "hidden_size")?,
            num_hidden_layers: config_usize(
                config["num_hidden_layers"].as_u64(),
                "num_hidden_layers",
            )?,
            num_attention_heads: config_usize(
                config["num_attention_heads"].as_u64(),
                "num_attention_heads",
            )?,
            intermediate_size: config_usize(
                config["intermediate_size"].as_u64(),
                "intermediate_size",
            )?,
            max_position_embeddings: config_usize(
                config["max_position_embeddings"].as_u64(),
                "max_position_embeddings",
            )?,
            type_vocab_size: config_usize(config["type_vocab_size"].as_u64(), "type_vocab_size")?,
            layer_norm_eps: config_f32(
                config["layer_norm_eps"].as_f64(),
                "layer_norm_eps",
                1.0e-12,
            )?,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.intermediate_size == 0
            || self.max_position_embeddings == 0
            || self.type_vocab_size == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.hidden_size > usize::from(u16::MAX)
            || !self.layer_norm_eps.is_finite()
            || self.layer_norm_eps <= 0.0
        {
            return Err(ModelError::InvalidConfig(
                "nonzero geometry, divisible attention heads, finite epsilon, and practical hidden width are required"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

fn config_usize(value: Option<u64>, key: &'static str) -> Result<usize, ModelError> {
    let value =
        value.ok_or_else(|| ModelError::InvalidConfig(format!("missing integer field {key}")))?;
    usize::try_from(value)
        .map_err(|_| ModelError::InvalidConfig(format!("field {key} does not fit usize")))
}

#[allow(clippy::cast_possible_truncation)]
fn config_f32(value: Option<f64>, key: &'static str, default: f32) -> Result<f32, ModelError> {
    let Some(value) = value else {
        return Ok(default);
    };
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(ModelError::InvalidConfig(format!(
            "field {key} is outside the supported f32 range"
        )));
    }
    Ok(value as f32)
}

struct ModelWeights<'a> {
    weights: LocalWeightSet<'a>,
}

impl<'a> ModelWeights<'a> {
    fn new(package: &'a LocalModelPackage) -> Self {
        Self {
            weights: package.weight_set(),
        }
    }

    fn load_required(
        &mut self,
        suffix: &str,
        expected_shape: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        self.load_optional(suffix, expected_shape)?
            .ok_or_else(|| ModelError::MissingParameter(suffix.to_owned()))
    }

    fn load_optional(
        &mut self,
        suffix: &str,
        expected_shape: &[usize],
    ) -> Result<Option<Vec<f32>>, ModelError> {
        for name in [format!("bert.{suffix}"), suffix.to_owned()] {
            let artifact = match self.weights.artifact(&name) {
                Ok(artifact) => artifact,
                Err(PackageError::UnknownParameter(_)) => continue,
                Err(error) => return Err(ModelError::Artifact(error.to_string())),
            };
            let tensor = match artifact.tensor(&name) {
                Ok(tensor) => tensor,
                Err(ArtifactError::MissingTensor(_)) => continue,
                Err(error) => return Err(ModelError::Artifact(error.to_string())),
            };
            return decode_f32_tensor(&name, &tensor, expected_shape).map(Some);
        }
        Ok(None)
    }

    fn artifact_count(&self) -> usize {
        self.weights.opened_shard_count()
    }
}

fn decode_f32_tensor(
    name: &str,
    tensor: &ribn_safetensors::ArtifactTensor<'_>,
    expected_shape: &[usize],
) -> Result<Vec<f32>, ModelError> {
    if tensor.scalar_type() != &ScalarType::F32 || tensor.shape() != expected_shape {
        return Err(ModelError::UnsupportedTensor(format!(
            "{name} expected F32 {expected_shape:?}, found {:?} {:?}",
            tensor.scalar_type(),
            tensor.shape()
        )));
    }
    let expected_elements = expected_shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension));
    let Some(expected_elements) = expected_elements else {
        return Err(ModelError::UnsupportedTensor(format!(
            "{name} shape overflows usize"
        )));
    };
    let (values, remainder) = tensor.data().as_chunks::<4>();
    if !remainder.is_empty() || values.len() != expected_elements {
        return Err(ModelError::UnsupportedTensor(format!(
            "{name} payload length does not match its shape"
        )));
    }
    Ok(values
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

struct Dense {
    input: usize,
    output: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Dense {
    fn load(
        cache: &mut ModelWeights<'_>,
        prefix: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            input,
            output,
            weight: cache.load_required(&format!("{prefix}.weight"), &[output, input])?,
            bias: cache.load_required(&format!("{prefix}.bias"), &[output])?,
        })
    }

    fn forward_rows(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len() % self.input, 0);
        let rows = input.len() / self.input;
        let mut output = Vec::with_capacity(rows.saturating_mul(self.output));
        for row in input.chunks_exact(self.input) {
            for output_index in 0..self.output {
                let start = output_index * self.input;
                let weights = &self.weight[start..start + self.input];
                let value = row
                    .iter()
                    .zip(weights)
                    .fold(self.bias[output_index], |sum, (input, weight)| {
                        input.mul_add(*weight, sum)
                    });
                output.push(value);
            }
        }
        output
    }
}

struct LayerNorm {
    width: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    epsilon: f32,
}

impl LayerNorm {
    fn load(
        cache: &mut ModelWeights<'_>,
        prefix: &str,
        width: usize,
        epsilon: f32,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            width,
            weight: cache.load_required(&format!("{prefix}.weight"), &[width])?,
            bias: cache.load_required(&format!("{prefix}.bias"), &[width])?,
            epsilon,
        })
    }

    fn apply_rows(&self, values: &mut [f32]) {
        let width = f32::from(u16::try_from(self.width).expect("validated BERT hidden width"));
        for row in values.chunks_exact_mut(self.width) {
            let mean = row.iter().copied().sum::<f32>() / width;
            let variance = row
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / width;
            let inverse = (variance + self.epsilon).sqrt().recip();
            for ((value, scale), bias) in row.iter_mut().zip(&self.weight).zip(&self.bias) {
                *value = (*value - mean).mul_add(inverse * *scale, *bias);
            }
        }
    }
}

struct Embeddings {
    word: Vec<f32>,
    position: Vec<f32>,
    token_type: Vec<f32>,
    norm: LayerNorm,
}

impl Embeddings {
    fn load(cache: &mut ModelWeights<'_>, config: &BertConfig) -> Result<Self, ModelError> {
        Ok(Self {
            word: cache.load_required(
                "embeddings.word_embeddings.weight",
                &[config.vocab_size, config.hidden_size],
            )?,
            position: cache.load_required(
                "embeddings.position_embeddings.weight",
                &[config.max_position_embeddings, config.hidden_size],
            )?,
            token_type: cache.load_required(
                "embeddings.token_type_embeddings.weight",
                &[config.type_vocab_size, config.hidden_size],
            )?,
            norm: LayerNorm::load(
                cache,
                "embeddings.LayerNorm",
                config.hidden_size,
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, input: &BertInput, config: &BertConfig) -> Result<Vec<f32>, ModelError> {
        let sequence = input.token_ids.len();
        if sequence == 0 || sequence > config.max_position_embeddings {
            return Err(ModelError::InvalidInput(format!(
                "sequence length {sequence} is outside 1..={}",
                config.max_position_embeddings
            )));
        }
        if let Some(type_ids) = &input.token_type_ids
            && type_ids.len() != sequence
        {
            return Err(ModelError::InvalidInput(
                "token_type_ids length must match token_ids".to_owned(),
            ));
        }

        let mut output = vec![0.0_f32; sequence.saturating_mul(config.hidden_size)];
        for (position, (&token_id, row)) in input
            .token_ids
            .iter()
            .zip(output.chunks_exact_mut(config.hidden_size))
            .enumerate()
        {
            let token = usize::try_from(token_id).map_err(|_| {
                ModelError::InvalidInput(format!("token id {token_id} does not fit usize"))
            })?;
            if token >= config.vocab_size {
                return Err(ModelError::InvalidInput(format!(
                    "token id {token_id} exceeds vocab_size {}",
                    config.vocab_size
                )));
            }
            let type_id = input
                .token_type_ids
                .as_ref()
                .map_or(0_u32, |ids| ids[position]);
            let type_index = usize::try_from(type_id).map_err(|_| {
                ModelError::InvalidInput(format!("token type id {type_id} does not fit usize"))
            })?;
            if type_index >= config.type_vocab_size {
                return Err(ModelError::InvalidInput(format!(
                    "token type id {type_id} exceeds type_vocab_size {}",
                    config.type_vocab_size
                )));
            }

            let word_start = token * config.hidden_size;
            let position_start = position * config.hidden_size;
            let type_start = type_index * config.hidden_size;
            for (hidden, destination) in row.iter_mut().enumerate() {
                *destination = self.word[word_start + hidden]
                    + self.position[position_start + hidden]
                    + self.token_type[type_start + hidden];
            }
        }
        self.norm.apply_rows(&mut output);
        Ok(output)
    }
}

struct BertLayer {
    query: Dense,
    key: Dense,
    value: Dense,
    attention_output: Dense,
    attention_norm: LayerNorm,
    intermediate: Dense,
    output: Dense,
    output_norm: LayerNorm,
}

impl BertLayer {
    fn load(
        cache: &mut ModelWeights<'_>,
        config: &BertConfig,
        layer: usize,
    ) -> Result<Self, ModelError> {
        let prefix = format!("encoder.layer.{layer}");
        Ok(Self {
            query: Dense::load(
                cache,
                &format!("{prefix}.attention.self.query"),
                config.hidden_size,
                config.hidden_size,
            )?,
            key: Dense::load(
                cache,
                &format!("{prefix}.attention.self.key"),
                config.hidden_size,
                config.hidden_size,
            )?,
            value: Dense::load(
                cache,
                &format!("{prefix}.attention.self.value"),
                config.hidden_size,
                config.hidden_size,
            )?,
            attention_output: Dense::load(
                cache,
                &format!("{prefix}.attention.output.dense"),
                config.hidden_size,
                config.hidden_size,
            )?,
            attention_norm: LayerNorm::load(
                cache,
                &format!("{prefix}.attention.output.LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            intermediate: Dense::load(
                cache,
                &format!("{prefix}.intermediate.dense"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            output: Dense::load(
                cache,
                &format!("{prefix}.output.dense"),
                config.intermediate_size,
                config.hidden_size,
            )?,
            output_norm: LayerNorm::load(
                cache,
                &format!("{prefix}.output.LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, hidden_states: &[f32], config: &BertConfig) -> Vec<f32> {
        let sequence = hidden_states.len() / config.hidden_size;
        let query = self.query.forward_rows(hidden_states);
        let key = self.key.forward_rows(hidden_states);
        let value = self.value.forward_rows(hidden_states);
        let context = attention_context(
            &query,
            &key,
            &value,
            sequence,
            config.hidden_size,
            config.num_attention_heads,
        );
        let mut attention = self.attention_output.forward_rows(&context);
        add_in_place(&mut attention, hidden_states);
        self.attention_norm.apply_rows(&mut attention);

        let mut intermediate = self.intermediate.forward_rows(&attention);
        for value in &mut intermediate {
            *value = gelu(*value);
        }
        let mut output = self.output.forward_rows(&intermediate);
        add_in_place(&mut output, &attention);
        self.output_norm.apply_rows(&mut output);
        output
    }
}

fn add_in_place(destination: &mut [f32], source: &[f32]) {
    debug_assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
}

#[allow(clippy::needless_range_loop)]
fn attention_context(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence: usize,
    hidden: usize,
    heads: usize,
) -> Vec<f32> {
    let head_width = hidden / heads;
    let head_width_f32 =
        f32::from(u16::try_from(head_width).expect("validated BERT attention width"));
    let scale = head_width_f32.sqrt().recip();
    let mut output = vec![0.0_f32; sequence.saturating_mul(hidden)];
    let mut scores = vec![0.0_f32; sequence];

    for query_position in 0..sequence {
        for head in 0..heads {
            let head_offset = head * head_width;
            for key_position in 0..sequence {
                let mut score = 0.0_f32;
                for hidden_index in 0..head_width {
                    let query_index = query_position * hidden + head_offset + hidden_index;
                    let key_index = key_position * hidden + head_offset + hidden_index;
                    score = query[query_index].mul_add(key[key_index], score);
                }
                scores[key_position] = score * scale;
            }
            softmax_in_place(&mut scores);
            for value_position in 0..sequence {
                for hidden_index in 0..head_width {
                    let output_index = query_position * hidden + head_offset + hidden_index;
                    let value_index = value_position * hidden + head_offset + hidden_index;
                    output[output_index] =
                        scores[value_position].mul_add(value[value_index], output[output_index]);
                }
            }
        }
    }
    output
}

fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        total += *value;
    }
    for value in values {
        *value /= total;
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + erf(value / std::f32::consts::SQRT_2))
}

#[allow(clippy::excessive_precision)]
fn erf(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let input = value.abs();
    let factor = 1.0 / input.mul_add(0.327_591_1, 1.0);
    let polynomial = (((1.061_405_4 * factor - 1.453_152_1) * factor + 1.421_413_8) * factor
        - 0.284_496_72)
        * factor
        + 0.254_829_6;
    sign * (1.0 - polynomial * factor * (-input * input).exp())
}

#[derive(Clone)]
struct BertInput {
    token_ids: Vec<u32>,
    token_type_ids: Option<Vec<u32>>,
}

impl BertInput {
    fn new(token_ids: impl Into<Vec<u32>>) -> Self {
        Self {
            token_ids: token_ids.into(),
            token_type_ids: None,
        }
    }
}

struct BertOutput {
    sequence: usize,
    hidden: usize,
    last_hidden_state: Vec<f32>,
    pooled_output: Option<Vec<f32>>,
}

struct BertReference {
    config: BertConfig,
    embeddings: Embeddings,
    layers: Vec<BertLayer>,
    pooler: Option<Dense>,
    version: ParameterVersion,
    max_batch_items: usize,
    max_batch_tokens: usize,
    loaded_artifacts: usize,
}

impl BertReference {
    fn load(
        package: &LocalModelPackage,
        version: ParameterVersion,
        max_batch_items: usize,
        max_batch_tokens: usize,
    ) -> Result<Self, ModelError> {
        if max_batch_items == 0 || max_batch_tokens == 0 {
            return Err(ModelError::InvalidConfig(
                "batch limits must be nonzero".to_owned(),
            ));
        }
        let config = BertConfig::from_package(package)?;
        let mut cache = ModelWeights::new(package);
        let embeddings = Embeddings::load(&mut cache, &config)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(BertLayer::load(&mut cache, &config, layer)?);
        }
        let pooler_weight = cache.load_optional(
            "pooler.dense.weight",
            &[config.hidden_size, config.hidden_size],
        )?;
        let pooler = if let Some(weight) = pooler_weight {
            Some(Dense {
                input: config.hidden_size,
                output: config.hidden_size,
                weight,
                bias: cache.load_required("pooler.dense.bias", &[config.hidden_size])?,
            })
        } else {
            None
        };
        let loaded_artifacts = cache.artifact_count();
        Ok(Self {
            config,
            embeddings,
            layers,
            pooler,
            version,
            max_batch_items,
            max_batch_tokens,
            loaded_artifacts,
        })
    }

    fn encode(&self, input: &BertInput) -> Result<BertOutput, ModelError> {
        let mut hidden_states = self.embeddings.forward(input, &self.config)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.config);
        }
        let pooled_output = self.pooler.as_ref().map(|pooler| {
            pooler
                .forward_rows(&hidden_states[..self.config.hidden_size])
                .into_iter()
                .map(f32::tanh)
                .collect()
        });
        Ok(BertOutput {
            sequence: input.token_ids.len(),
            hidden: self.config.hidden_size,
            last_hidden_state: hidden_states,
            pooled_output,
        })
    }
}

impl BatchExecutor for BertReference {
    type Input = BertInput;
    type Output = BertOutput;
    type Error = ModelError;

    fn parameter_version(&self) -> ParameterVersion {
        self.version
    }

    fn max_batch_items(&self) -> usize {
        self.max_batch_items
    }

    fn select_batch(&self, candidates: &[&Self::Input]) -> usize {
        let mut selected = 0_usize;
        let mut tokens = 0_usize;
        for input in candidates {
            let Some(next) = tokens.checked_add(input.token_ids.len()) else {
                break;
            };
            if selected > 0 && next > self.max_batch_tokens {
                break;
            }
            selected += 1;
            tokens = next;
            if tokens >= self.max_batch_tokens {
                break;
            }
        }
        selected
    }

    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
        batch
            .into_iter()
            .map(|job| {
                let request = job.request();
                Ok(JobOutput::new(request, self.encode(job.input())?))
            })
            .collect()
    }
}

struct TensorFixture {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn tensor(name: impl Into<String>, shape: &[usize], values: Vec<f32>) -> TensorFixture {
    TensorFixture {
        name: name.into(),
        shape: shape.to_vec(),
        values,
    }
}

fn zeros(elements: usize) -> Vec<f32> {
    vec![0.0; elements]
}

fn ones(elements: usize) -> Vec<f32> {
    vec![1.0; elements]
}

fn identity(width: usize) -> Vec<f32> {
    let mut values = zeros(width.saturating_mul(width));
    for diagonal in 0..width {
        values[diagonal * width + diagonal] = 1.0;
    }
    values
}

#[allow(clippy::too_many_lines)]
fn bert_fixture_tensors() -> Vec<TensorFixture> {
    let hidden = 4;
    let intermediate = 4;
    let mut tensors = vec![
        tensor(
            "bert.embeddings.word_embeddings.weight",
            &[3, hidden],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        tensor(
            "bert.embeddings.position_embeddings.weight",
            &[8, hidden],
            zeros(8 * hidden),
        ),
        tensor(
            "bert.embeddings.token_type_embeddings.weight",
            &[2, hidden],
            zeros(2 * hidden),
        ),
        tensor("bert.embeddings.LayerNorm.weight", &[hidden], ones(hidden)),
        tensor("bert.embeddings.LayerNorm.bias", &[hidden], zeros(hidden)),
    ];

    let layer = "bert.encoder.layer.0";
    tensors.extend([
        tensor(
            format!("{layer}.attention.self.query.weight"),
            &[hidden, hidden],
            zeros(hidden * hidden),
        ),
        tensor(
            format!("{layer}.attention.self.query.bias"),
            &[hidden],
            zeros(hidden),
        ),
        tensor(
            format!("{layer}.attention.self.key.weight"),
            &[hidden, hidden],
            zeros(hidden * hidden),
        ),
        tensor(
            format!("{layer}.attention.self.key.bias"),
            &[hidden],
            zeros(hidden),
        ),
        tensor(
            format!("{layer}.attention.self.value.weight"),
            &[hidden, hidden],
            identity(hidden),
        ),
        tensor(
            format!("{layer}.attention.self.value.bias"),
            &[hidden],
            zeros(hidden),
        ),
        tensor(
            format!("{layer}.attention.output.dense.weight"),
            &[hidden, hidden],
            identity(hidden),
        ),
        tensor(
            format!("{layer}.attention.output.dense.bias"),
            &[hidden],
            zeros(hidden),
        ),
        tensor(
            format!("{layer}.attention.output.LayerNorm.weight"),
            &[hidden],
            ones(hidden),
        ),
        tensor(
            format!("{layer}.attention.output.LayerNorm.bias"),
            &[hidden],
            zeros(hidden),
        ),
        tensor(
            format!("{layer}.intermediate.dense.weight"),
            &[intermediate, hidden],
            zeros(intermediate * hidden),
        ),
        tensor(
            format!("{layer}.intermediate.dense.bias"),
            &[intermediate],
            zeros(intermediate),
        ),
        tensor(
            format!("{layer}.output.dense.weight"),
            &[hidden, intermediate],
            zeros(hidden * intermediate),
        ),
        tensor(
            format!("{layer}.output.dense.bias"),
            &[hidden],
            zeros(hidden),
        ),
        tensor(
            format!("{layer}.output.LayerNorm.weight"),
            &[hidden],
            ones(hidden),
        ),
        tensor(
            format!("{layer}.output.LayerNorm.bias"),
            &[hidden],
            zeros(hidden),
        ),
        tensor(
            "bert.pooler.dense.weight",
            &[hidden, hidden],
            identity(hidden),
        ),
        tensor("bert.pooler.dense.bias", &[hidden], zeros(hidden)),
    ]);
    tensors
}

fn safetensors_fixture(tensors: &[TensorFixture]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header = String::from("{");
    for (index, tensor) in tensors.iter().enumerate() {
        let start = data.len();
        for value in &tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let end = data.len();
        let shape = tensor
            .shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        if index > 0 {
            header.push(',');
        }
        write!(
            header,
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            tensor.name, shape, start, end
        )
        .expect("fixture header");
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

fn write_bert_package(root: &Path) {
    fs::write(
        root.join("config.json"),
        br#"{
            "architectures":["BertForPreTraining"],
            "model_type":"bert",
            "vocab_size":3,
            "hidden_size":4,
            "num_hidden_layers":1,
            "num_attention_heads":2,
            "intermediate_size":4,
            "hidden_act":"gelu",
            "max_position_embeddings":8,
            "type_vocab_size":2,
            "layer_norm_eps":1e-12,
            "position_embedding_type":"absolute",
            "is_decoder":false,
            "add_cross_attention":false
        }"#,
    )
    .expect("config");
    fs::write(
        root.join("model.safetensors"),
        safetensors_fixture(&bert_fixture_tensors()),
    )
    .expect("weights");
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).abs() <= 2.0e-5,
            "{actual} differs from {expected}"
        );
    }
}

#[test]
fn actual_bert_encoder_semantics_load_from_hf_package_and_run_non_ar() {
    let dir = TestDir::new();
    write_bert_package(dir.path());
    let package = LocalModelPackage::open(dir.path()).expect("package");
    let model = BertReference::load(&package, ParameterVersion::new(41), 4, 8).expect("BERT");
    assert_eq!(model.loaded_artifacts, 1);

    let mut runtime = BatchRuntime::new(
        model,
        BatchConfig {
            max_queued_requests: 4,
        },
    )
    .expect("runtime");
    let first = runtime
        .submit(BertInput::new(vec![0, 1]))
        .expect("first request");
    let second = runtime
        .submit(BertInput::new(vec![2]))
        .expect("second request");
    assert!(runtime.step().expect("BERT batch"));

    let first_output = runtime.pop_completed().expect("first output");
    assert_eq!(first_output.request(), first);
    assert_eq!(first_output.parameter_version(), ParameterVersion::new(41));
    assert_eq!(first_output.output().sequence, 2);
    assert_eq!(first_output.output().hidden, 4);
    assert_close(
        &first_output.output().last_hidden_state,
        &[
            1.632_993_2,
            0.0,
            -0.816_496_6,
            -0.816_496_6,
            0.0,
            1.632_993_2,
            -0.816_496_6,
            -0.816_496_6,
        ],
    );
    assert_close(
        first_output
            .output()
            .pooled_output
            .as_deref()
            .expect("pooler output"),
        &[0.926_486_7, 0.0, -0.673_158_5, -0.673_158_5],
    );

    let second_output = runtime.pop_completed().expect("second output");
    assert_eq!(second_output.request(), second);
}

#[test]
fn bert_sequence_lengths_drive_executor_batch_selection() {
    let dir = TestDir::new();
    write_bert_package(dir.path());
    let package = LocalModelPackage::open(dir.path()).expect("package");
    let model = BertReference::load(&package, ParameterVersion::new(42), 4, 3).expect("BERT");
    let mut runtime = BatchRuntime::new(
        model,
        BatchConfig {
            max_queued_requests: 4,
        },
    )
    .expect("runtime");

    runtime
        .submit(BertInput::new(vec![0, 1]))
        .expect("first request");
    runtime
        .submit(BertInput::new(vec![1, 2]))
        .expect("second request");
    assert!(runtime.step().expect("first batch"));
    assert_eq!(runtime.queued(), 1);
    assert!(runtime.pop_completed().is_some());
    assert!(runtime.pop_completed().is_none());
    assert!(runtime.step().expect("second batch"));
    assert_eq!(runtime.queued(), 0);
}
