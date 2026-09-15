//! Device execution for the encoder.
//!
//! One encoder owns its prepared weights and a CUDA stream. A request is submitted
//! as an owned [`EncoderSubmission`]: the device buffers it needs stay alive with
//! it, and its recorded completion event is the single ordering point covering
//! every kernel and copy of that request. Reading a result therefore requires
//! completion, which is exactly the producer-side dependency a downstream consumer
//! must respect before it touches the storage.
//!
//! Projections are dense GEMMs and run on cuBLAS. The remaining per-row and
//! elementwise work runs on `engine-nvidia`'s encoder primitives.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream};
use engine_nvidia::{CudaEncoderKernelError, CudaEncoderOps};
use ribn_hf::LocalModelPackage;

use crate::config::{BertConfig, ConfigError};
use crate::weights::{Dense as HostDense, HostWeights, LayerNorm as HostLayerNorm, WeightError};

/// One encoder request: token ids, their segment ids and a per-position key mask.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncoderRequest {
    pub token_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    /// One flag per position; a zero flag removes that position as a key.
    pub attention_mask: Vec<i32>,
}

impl EncoderRequest {
    /// A single-segment request whose every position is attendable.
    #[must_use]
    pub fn single_segment(token_ids: Vec<u32>) -> Self {
        Self {
            token_type_ids: vec![0; token_ids.len()],
            attention_mask: vec![1; token_ids.len()],
            token_ids,
        }
    }
}

/// Encoder results for one request.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderOutput {
    /// `[sequence, hidden]` row-major hidden states after the last layer.
    pub last_hidden_state: Vec<f32>,
    /// Pooled `[hidden]` vector from the first position.
    pub pooled: Vec<f32>,
}

/// Why an encoder request could not be prepared or executed.
#[derive(Debug)]
pub enum EncoderError {
    Config(ConfigError),
    Weights(WeightError),
    Package(String),
    Kernels(CudaEncoderKernelError),
    Driver(String),
    InvalidRequest(String),
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => write!(f, "encoder configuration: {error}"),
            Self::Weights(error) => write!(f, "encoder parameters: {error}"),
            Self::Package(message) => write!(f, "encoder package: {message}"),
            Self::Kernels(error) => write!(f, "encoder kernels: {error}"),
            Self::Driver(message) => write!(f, "encoder driver: {message}"),
            Self::InvalidRequest(message) => write!(f, "encoder request: {message}"),
        }
    }
}

impl std::error::Error for EncoderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Weights(error) => Some(error),
            Self::Kernels(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ConfigError> for EncoderError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<WeightError> for EncoderError {
    fn from(error: WeightError) -> Self {
        Self::Weights(error)
    }
}

impl From<CudaEncoderKernelError> for EncoderError {
    fn from(error: CudaEncoderKernelError) -> Self {
        Self::Kernels(error)
    }
}

/// A prepared BERT encoder on one CUDA device.
pub struct CudaBertEncoder {
    config: BertConfig,
    ops: CudaEncoderOps,
    blas: CudaBlas,
    weights: DeviceWeights,
}

impl CudaBertEncoder {
    /// Resolve a package, validate its configuration and prepare its weights.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for an unusable configuration, a missing or
    /// mis-shaped parameter, or a CUDA/NVRTC/cuBLAS failure.
    pub fn load(package_root: impl AsRef<Path>, device_index: usize) -> Result<Self, EncoderError> {
        let package = LocalModelPackage::open(package_root.as_ref())
            .map_err(|error| EncoderError::Package(error.to_string()))?;
        let config = BertConfig::from_json(package.config())?;
        let host = HostWeights::load(&package, &config)?;
        Self::from_host(config, &host, device_index)
    }

    /// Prepare an already-loaded host parameter set on the device.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for a configuration/parameter mismatch or a CUDA,
    /// NVRTC or cuBLAS failure.
    pub fn from_host(
        config: BertConfig,
        host: &HostWeights,
        device_index: usize,
    ) -> Result<Self, EncoderError> {
        config.validate()?;
        let ops = CudaEncoderOps::new(device_index).map_err(EncoderError::Kernels)?;
        let stream = Arc::clone(ops.stream());
        let blas = CudaBlas::new(Arc::clone(&stream))
            .map_err(|error| EncoderError::Driver(error.to_string()))?;
        let upload = |values: &[f32]| -> Result<CudaSlice<f32>, EncoderError> {
            stream
                .clone_htod(values)
                .map_err(|error| EncoderError::Driver(error.to_string()))
        };
        let upload_dense = |dense: &HostDense| -> Result<DeviceDense, EncoderError> {
            Ok(DeviceDense {
                input: dense.input,
                output: dense.output,
                weight: upload(&dense.weight)?,
                bias: upload(&dense.bias)?,
            })
        };
        let upload_norm = |norm: &HostLayerNorm| -> Result<DeviceLayerNorm, EncoderError> {
            Ok(DeviceLayerNorm {
                weight: upload(&norm.weight)?,
                bias: upload(&norm.bias)?,
            })
        };
        let weights = DeviceWeights {
            word_embeddings: upload(&host.word_embeddings)?,
            position_embeddings: upload(&host.position_embeddings)?,
            token_type_embeddings: upload(&host.token_type_embeddings)?,
            embedding_norm: upload_norm(&host.embedding_norm)?,
            layers: host
                .layers
                .iter()
                .map(|layer| -> Result<DeviceLayer, EncoderError> {
                    Ok(DeviceLayer {
                        query: upload_dense(&layer.query)?,
                        key: upload_dense(&layer.key)?,
                        value: upload_dense(&layer.value)?,
                        attention_output: upload_dense(&layer.attention_output)?,
                        attention_norm: upload_norm(&layer.attention_norm)?,
                        intermediate: upload_dense(&layer.intermediate)?,
                        output: upload_dense(&layer.output)?,
                        output_norm: upload_norm(&layer.output_norm)?,
                    })
                })
                .collect::<Result<Vec<_>, EncoderError>>()?,
            pooler: upload_dense(&host.pooler)?,
        };
        Ok(Self {
            config,
            ops,
            blas,
            weights,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &BertConfig {
        &self.config
    }

    /// The stream every submission of this encoder runs on.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        self.ops.stream()
    }

    /// Bytes one prepared request holds on the device until it is released.
    ///
    /// This is the request's real envelope: staging plus every activation the
    /// forward pass keeps alive. It is what a shared pool must cover before the
    /// work is admitted.
    ///
    /// # Errors
    /// Returns [`EncoderError::InvalidRequest`] for a sequence this encoder cannot
    /// execute.
    pub fn request_bytes(&self, sequence: usize) -> Result<u64, EncoderError> {
        let geometry = self.geometry(sequence)?;
        Ok(geometry.workspace_bytes)
    }

    /// Enqueue one request and return its owned device state.
    ///
    /// The returned submission owns every device buffer the request uses and a
    /// completion event recorded after the last of them. Nothing is read back
    /// here, so the caller decides when to wait.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for an invalid request or a CUDA/cuBLAS failure.
    pub fn submit(&self, request: &EncoderRequest) -> Result<EncoderSubmission<'_>, EncoderError> {
        let geometry = self.geometry(request.token_ids.len())?;
        validate_ids(request, &self.config)?;
        let stream = self.stream();
        let upload_tokens = |values: &[u32]| -> Result<CudaSlice<u32>, EncoderError> {
            stream
                .clone_htod(values)
                .map_err(|error| EncoderError::Driver(error.to_string()))
        };
        let mask: Vec<i32> = request.attention_mask.clone();
        let mut submission = EncoderSubmission {
            encoder: self,
            geometry,
            tokens: upload_tokens(&request.token_ids)?,
            types: upload_tokens(&request.token_type_ids)?,
            mask: stream
                .clone_htod(&mask)
                .map_err(|error| EncoderError::Driver(error.to_string()))?,
            hidden: alloc(stream, geometry.hidden_values)?,
            query: alloc(stream, geometry.hidden_values)?,
            key: alloc(stream, geometry.hidden_values)?,
            value: alloc(stream, geometry.hidden_values)?,
            context: alloc(stream, geometry.hidden_values)?,
            projection: alloc(stream, geometry.hidden_values)?,
            intermediate: alloc(stream, geometry.intermediate_values)?,
            first_row: alloc(stream, geometry.hidden)?,
            pooled: alloc(stream, geometry.hidden)?,
            completion: None,
        };
        submission.run()?;
        submission.completion = Some(
            stream
                .record_event(None)
                .map_err(|error| EncoderError::Driver(error.to_string()))?,
        );
        Ok(submission)
    }

    /// Run one request to completion and read its results.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for an invalid request, an execution failure, or a
    /// readback failure.
    pub fn encode(&self, request: &EncoderRequest) -> Result<EncoderOutput, EncoderError> {
        let mut submission = self.submit(request)?;
        submission.synchronize()?;
        submission.read()
    }

    /// Validate a request's shape without touching the device.
    fn geometry(&self, sequence: usize) -> Result<Geometry, EncoderError> {
        if sequence == 0 {
            return Err(EncoderError::InvalidRequest(
                "sequence length must be positive".to_owned(),
            ));
        }
        if sequence > self.config.max_position_embeddings {
            return Err(EncoderError::InvalidRequest(format!(
                "sequence length {sequence} exceeds the model's {} position embeddings",
                self.config.max_position_embeddings
            )));
        }
        let hidden = self.config.hidden_size;
        let hidden_values = sequence * hidden;
        let intermediate_values = sequence * self.config.intermediate_size;
        // Staging plus the activations the forward pass keeps alive.
        let float_values = 6 * hidden_values + intermediate_values + 2 * hidden;
        let bytes = float_values * size_of::<f32>() + 3 * sequence * size_of::<u32>();
        Ok(Geometry {
            sequence,
            hidden,
            heads: self.config.num_attention_heads,
            hidden_values,
            intermediate_values,
            workspace_bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
        })
    }

    /// Dense projection `weight · input + bias` for `rows` rows.
    ///
    /// The checkpoint stores `weight` row-major as `[output, input]`, which is
    /// column-major `[input, output]`: the projection is therefore a `k = input`
    /// product against the transpose of that view, producing `[output, rows]`
    /// column-major, which is `[rows, output]` row-major.
    fn project(
        &self,
        dense: &DeviceDense,
        input: &CudaSlice<f32>,
        rows: usize,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), EncoderError> {
        let (m, n, k) = (dense.output, rows, dense.input);
        let config = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: i32::try_from(m).map_err(|_| EncoderError::Driver("width".to_owned()))?,
            n: i32::try_from(n).map_err(|_| EncoderError::Driver("rows".to_owned()))?,
            k: i32::try_from(k).map_err(|_| EncoderError::Driver("inner".to_owned()))?,
            alpha: 1.0,
            lda: i32::try_from(k).map_err(|_| EncoderError::Driver("inner".to_owned()))?,
            ldb: i32::try_from(k).map_err(|_| EncoderError::Driver("inner".to_owned()))?,
            beta: 0.0,
            ldc: i32::try_from(m).map_err(|_| EncoderError::Driver("width".to_owned()))?,
        };
        // Safety: every slice belongs to this encoder's stream and context, and the
        // strides above describe exactly the buffers' element counts.
        unsafe {
            self.blas
                .gemm(config, &dense.weight, input, output)
                .map_err(|error| EncoderError::Driver(error.to_string()))?;
        }
        self.ops.add_row_bias(output, &dense.bias)?;
        Ok(())
    }
}

/// One request's device state, owned until the submission is dropped.
pub struct EncoderSubmission<'a> {
    encoder: &'a CudaBertEncoder,
    geometry: Geometry,
    tokens: CudaSlice<u32>,
    types: CudaSlice<u32>,
    mask: CudaSlice<i32>,
    hidden: CudaSlice<f32>,
    query: CudaSlice<f32>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    context: CudaSlice<f32>,
    projection: CudaSlice<f32>,
    intermediate: CudaSlice<f32>,
    first_row: CudaSlice<f32>,
    pooled: CudaSlice<f32>,
    completion: Option<CudaEvent>,
}

impl EncoderSubmission<'_> {
    /// Bytes this submission holds on the device.
    #[must_use]
    pub const fn device_bytes(&self) -> u64 {
        self.geometry.workspace_bytes
    }

    /// Whether every kernel and copy of this submission has completed.
    ///
    /// # Errors
    /// Returns [`EncoderError::Driver`] for a device fault rather than reporting a
    /// fault as "still running".
    pub fn is_complete(&self) -> Result<bool, EncoderError> {
        let Some(event) = &self.completion else {
            return Ok(false);
        };
        match unsafe { cudarc::driver::result::event::query(event.cu_event()) } {
            Ok(()) => Ok(true),
            Err(error) if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                Ok(false)
            }
            Err(error) => Err(EncoderError::Driver(error.to_string())),
        }
    }

    /// Block until this submission's device work and copies have completed.
    ///
    /// # Errors
    /// Returns [`EncoderError::Driver`] when the stream reports a device failure.
    pub fn synchronize(&mut self) -> Result<(), EncoderError> {
        self.encoder
            .stream()
            .synchronize()
            .map_err(|error| EncoderError::Driver(error.to_string()))
    }

    /// Read this submission's results.
    ///
    /// The caller must have established completion first; a read that races the
    /// device would return storage the device may still be writing.
    ///
    /// # Errors
    /// Returns [`EncoderError::Driver`] when a copy fails.
    pub fn read(&self) -> Result<EncoderOutput, EncoderError> {
        let stream = self.encoder.stream();
        let hidden = stream
            .clone_dtoh(&self.hidden)
            .map_err(|error| EncoderError::Driver(error.to_string()))?;
        let pooled = stream
            .clone_dtoh(&self.pooled)
            .map_err(|error| EncoderError::Driver(error.to_string()))?;
        Ok(EncoderOutput {
            last_hidden_state: hidden,
            pooled,
        })
    }

    /// Enqueue the whole forward pass on this encoder's stream.
    fn run(&mut self) -> Result<(), EncoderError> {
        let encoder = self.encoder;
        let config = encoder.config;
        let geometry = self.geometry;
        let ops = &encoder.ops;

        ops.embedding_sum(
            &self.tokens,
            &self.types,
            &encoder.weights.word_embeddings,
            &encoder.weights.position_embeddings,
            &encoder.weights.token_type_embeddings,
            geometry.hidden,
            &mut self.hidden,
        )?;
        ops.layer_norm_rows(
            &self.hidden,
            &encoder.weights.embedding_norm.weight,
            &encoder.weights.embedding_norm.bias,
            geometry.hidden,
            config.layer_norm_eps,
            &mut self.query,
        )?;
        std::mem::swap(&mut self.hidden, &mut self.query);

        for layer in &encoder.weights.layers {
            encoder.project(
                &layer.query,
                &self.hidden,
                geometry.sequence,
                &mut self.query,
            )?;
            encoder.project(&layer.key, &self.hidden, geometry.sequence, &mut self.key)?;
            encoder.project(
                &layer.value,
                &self.hidden,
                geometry.sequence,
                &mut self.value,
            )?;
            ops.attention_context(
                &self.query,
                &self.key,
                &self.value,
                &self.mask,
                geometry.hidden,
                geometry.heads,
                &mut self.context,
            )?;
            encoder.project(
                &layer.attention_output,
                &self.context,
                geometry.sequence,
                &mut self.projection,
            )?;
            ops.add_in_place(&mut self.projection, &self.hidden)?;
            ops.layer_norm_rows(
                &self.projection,
                &layer.attention_norm.weight,
                &layer.attention_norm.bias,
                geometry.hidden,
                config.layer_norm_eps,
                &mut self.hidden,
            )?;

            encoder.project(
                &layer.intermediate,
                &self.hidden,
                geometry.sequence,
                &mut self.intermediate,
            )?;
            ops.gelu_erf(&mut self.intermediate)?;
            encoder.project(
                &layer.output,
                &self.intermediate,
                geometry.sequence,
                &mut self.projection,
            )?;
            ops.add_in_place(&mut self.projection, &self.hidden)?;
            ops.layer_norm_rows(
                &self.projection,
                &layer.output_norm.weight,
                &layer.output_norm.bias,
                geometry.hidden,
                config.layer_norm_eps,
                &mut self.hidden,
            )?;
        }

        // The pooler reads the first position only, so copy that row and project it
        // as a single-row input instead of running the head over every position.
        encoder
            .stream()
            .memcpy_dtod(&self.hidden.slice(..geometry.hidden), &mut self.first_row)
            .map_err(|error| EncoderError::Driver(error.to_string()))?;
        encoder.project(
            &encoder.weights.pooler,
            &self.first_row,
            1,
            &mut self.pooled,
        )?;
        ops.tanh_apply(&mut self.pooled)?;
        Ok(())
    }
}

impl fmt::Debug for EncoderSubmission<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncoderSubmission")
            .field("sequence", &self.geometry.sequence)
            .field("device_bytes", &self.geometry.workspace_bytes)
            .field("complete", &self.is_complete().unwrap_or(false))
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    sequence: usize,
    hidden: usize,
    heads: usize,
    hidden_values: usize,
    intermediate_values: usize,
    workspace_bytes: u64,
}

struct DeviceDense {
    input: usize,
    output: usize,
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct DeviceLayerNorm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct DeviceLayer {
    query: DeviceDense,
    key: DeviceDense,
    value: DeviceDense,
    attention_output: DeviceDense,
    attention_norm: DeviceLayerNorm,
    intermediate: DeviceDense,
    output: DeviceDense,
    output_norm: DeviceLayerNorm,
}

struct DeviceWeights {
    word_embeddings: CudaSlice<f32>,
    position_embeddings: CudaSlice<f32>,
    token_type_embeddings: CudaSlice<f32>,
    embedding_norm: DeviceLayerNorm,
    layers: Vec<DeviceLayer>,
    pooler: DeviceDense,
}

fn alloc(stream: &Arc<CudaStream>, values: usize) -> Result<CudaSlice<f32>, EncoderError> {
    stream
        .alloc_zeros::<f32>(values)
        .map_err(|error| EncoderError::Driver(error.to_string()))
}

fn validate_ids(request: &EncoderRequest, config: &BertConfig) -> Result<(), EncoderError> {
    if request.token_type_ids.len() != request.token_ids.len()
        || request.attention_mask.len() != request.token_ids.len()
    {
        return Err(EncoderError::InvalidRequest(
            "token, token-type and mask lengths must agree".to_owned(),
        ));
    }
    for token in &request.token_ids {
        if usize::try_from(*token).map_or(true, |token| token >= config.vocab_size) {
            return Err(EncoderError::InvalidRequest(format!(
                "token id {token} is outside the model's vocabulary"
            )));
        }
    }
    for token_type in &request.token_type_ids {
        if usize::try_from(*token_type).map_or(true, |kind| kind >= config.type_vocab_size) {
            return Err(EncoderError::InvalidRequest(format!(
                "token type id {token_type} is outside the model's type vocabulary"
            )));
        }
    }
    Ok(())
}
