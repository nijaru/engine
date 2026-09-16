//! Device execution for the encoder.
//!
//! One encoder owns its prepared weights and a CUDA stream. A request is submitted
//! as an owned [`EncoderSubmission`]: the device buffers it needs stay alive with
//! it, and its recorded completion event is the single ordering point covering
//! every kernel and copy of that request. A submission therefore outlives any
//! borrow of the encoder and can be handed to a consumer that must respect that
//! producer dependency before it reads or reuses the storage.
//!
//! Projections are dense GEMMs and run on cuBLAS. The remaining per-row and
//! elementwise work runs on `engine-nvidia`'s encoder primitives.
//!
//! Which shapes this encoder can execute, and how many device bytes each of them
//! holds, are host decisions in [`crate::request`]; this module materializes them
//! rather than re-deriving them.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream};
use engine_nvidia::{CudaEncoderKernelError, CudaEncoderOps};
use ribn_hf::LocalModelPackage;

use crate::config::{BertConfig, ConfigError};
use crate::request::{EncoderConstraint, EncoderRequest, EncoderShape};
use crate::weights::{Dense as HostDense, HostWeights, LayerNorm as HostLayerNorm, WeightError};

/// Requests one accepted range may contain when the deployment does not say.
///
/// This is deployment policy, not a model property: it caps how much of one batch
/// the encoder enqueues at once. The shared pool remains the byte bound, and the
/// runtime shrinks the accepted range further when it cannot reserve the envelopes.
pub const DEFAULT_MAX_BATCH_ITEMS: usize = 16;

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
    /// The request itself can never run on this encoder, whatever is released.
    InvalidRequest(EncoderConstraint),
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => write!(f, "encoder configuration: {error}"),
            Self::Weights(error) => write!(f, "encoder parameters: {error}"),
            Self::Package(message) => write!(f, "encoder package: {message}"),
            Self::Kernels(error) => write!(f, "encoder kernels: {error}"),
            Self::Driver(message) => write!(f, "encoder driver: {message}"),
            Self::InvalidRequest(constraint) => write!(f, "encoder request: {constraint}"),
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

impl From<EncoderConstraint> for EncoderError {
    fn from(constraint: EncoderConstraint) -> Self {
        Self::InvalidRequest(constraint)
    }
}

/// A prepared BERT encoder on one CUDA device.
pub struct CudaBertEncoder {
    config: BertConfig,
    max_batch_items: usize,
    prepared: Arc<Prepared>,
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
            max_batch_items: DEFAULT_MAX_BATCH_ITEMS,
            prepared: Arc::new(Prepared { ops, blas, weights }),
        })
    }

    /// Set how many requests one accepted range may contain.
    ///
    /// Zero is not usable: the batching runtime refuses an executor that cannot run
    /// one item.
    #[must_use]
    pub fn with_max_batch_items(mut self, items: usize) -> Self {
        self.max_batch_items = items;
        self
    }

    #[must_use]
    pub const fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Requests one accepted range may contain.
    #[must_use]
    pub const fn max_batch_items(&self) -> usize {
        self.max_batch_items
    }

    /// The stream every submission of this encoder runs on.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        self.prepared.ops.stream()
    }

    /// Bytes one request of `sequence` positions holds on the device until released.
    ///
    /// This is the request's real envelope: staging plus every activation the
    /// forward pass keeps alive. It is what a shared pool must cover before the work
    /// is admitted, and [`EncoderSubmission::device_bytes`] reports the same
    /// quantity from the allocations that actually exist.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for a sequence this encoder cannot execute.
    pub fn request_bytes(&self, sequence: usize) -> Result<u64, EncoderError> {
        Ok(crate::request::shape(&self.config, sequence)?.device_bytes())
    }

    /// Enqueue one request and return its owned device state.
    ///
    /// The returned submission owns every device buffer the request uses, the
    /// prepared resources they were launched against, and a completion event
    /// recorded after the last of them. Nothing is read back here, so the caller
    /// decides when to wait.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for an invalid request or a CUDA/cuBLAS failure.
    pub fn submit(&self, request: &EncoderRequest) -> Result<EncoderSubmission, EncoderError> {
        let shape = crate::request::shape_of(&self.config, request)?;
        let stream = self.stream();
        let upload_tokens = |values: &[u32]| -> Result<CudaSlice<u32>, EncoderError> {
            stream
                .clone_htod(values)
                .map_err(|error| EncoderError::Driver(error.to_string()))
        };
        let mask: Vec<i32> = request.attention_mask.clone();
        let mut submission = EncoderSubmission {
            layer_norm_eps: self.config.layer_norm_eps,
            prepared: Arc::clone(&self.prepared),
            shape,
            tokens: upload_tokens(&request.token_ids)?,
            types: upload_tokens(&request.token_type_ids)?,
            mask: stream
                .clone_htod(&mask)
                .map_err(|error| EncoderError::Driver(error.to_string()))?,
            hidden: alloc(stream, shape.hidden_values())?,
            query: alloc(stream, shape.hidden_values())?,
            key: alloc(stream, shape.hidden_values())?,
            value: alloc(stream, shape.hidden_values())?,
            context: alloc(stream, shape.hidden_values())?,
            projection: alloc(stream, shape.hidden_values())?,
            intermediate: alloc(stream, shape.intermediate_values())?,
            first_row: alloc(stream, shape.hidden())?,
            pooled: alloc(stream, shape.hidden())?,
            completion: None,
        };
        let run = submission.run().and_then(|()| {
            submission.completion = Some(
                stream
                    .record_event(None)
                    .map_err(|error| EncoderError::Driver(error.to_string()))?,
            );
            Ok(())
        });
        match run {
            Ok(()) => Ok(submission),
            Err(error) => {
                // A partially enqueued request owns device storage the stream may
                // still be writing, so drain before the buffers are dropped. The
                // executor-owned retirement handshake replaces this best-effort
                // step; until then a device fault here leaves the storage's pool
                // charge with the caller rather than releasing it early.
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Run one request to completion and read its results.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for an invalid request, an execution failure, or a
    /// readback failure.
    pub fn encode(&self, request: &EncoderRequest) -> Result<EncoderOutput, EncoderError> {
        let submission = self.submit(request)?;
        submission.synchronize()?;
        submission.read()
    }
}

/// Prepared device state shared by every submission of one encoder.
///
/// A submission outlives any borrow of its encoder, because it is handed to a
/// consumer that may read it long after the submitting call returned. The kernel
/// functions, cuBLAS handle and weights it launches against are therefore shared
/// rather than borrowed, and it keeps them alive for as long as it owns storage.
struct Prepared {
    ops: CudaEncoderOps,
    blas: CudaBlas,
    weights: DeviceWeights,
}

impl Prepared {
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
pub struct EncoderSubmission {
    layer_norm_eps: f32,
    prepared: Arc<Prepared>,
    shape: EncoderShape,
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

impl EncoderSubmission {
    /// The concrete shape this submission was launched with.
    #[must_use]
    pub const fn shape(&self) -> EncoderShape {
        self.shape
    }

    /// Bytes this submission actually holds on the device.
    ///
    /// Counted from the buffers that exist rather than from the prediction, so a
    /// device test can prove that the envelope a pool was charged for equals the
    /// storage it covers.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        let floats = self.hidden.len()
            + self.query.len()
            + self.key.len()
            + self.value.len()
            + self.context.len()
            + self.projection.len()
            + self.intermediate.len()
            + self.first_row.len()
            + self.pooled.len();
        let ids = self.tokens.len() + self.types.len() + self.mask.len();
        let bytes = floats.checked_mul(size_of::<f32>()).and_then(|bytes| {
            ids.checked_mul(size_of::<u32>())
                .and_then(|ids| bytes.checked_add(ids))
        });
        u64::try_from(bytes.unwrap_or(usize::MAX)).unwrap_or(u64::MAX)
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
    pub fn synchronize(&self) -> Result<(), EncoderError> {
        self.prepared
            .ops
            .stream()
            .synchronize()
            .map_err(|error| EncoderError::Driver(error.to_string()))
    }

    /// Read this submission's results.
    ///
    /// The caller must have established completion first ([`Self::is_complete`] or
    /// [`Self::synchronize`]); a read that races the device would return storage the
    /// device may still be writing.
    ///
    /// # Errors
    /// Returns [`EncoderError::Driver`] when a copy fails.
    pub fn read(&self) -> Result<EncoderOutput, EncoderError> {
        let stream = self.prepared.ops.stream();
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
        let prepared = Arc::clone(&self.prepared);
        let shape = self.shape;
        let ops = &prepared.ops;

        ops.embedding_sum(
            &self.tokens,
            &self.types,
            &prepared.weights.word_embeddings,
            &prepared.weights.position_embeddings,
            &prepared.weights.token_type_embeddings,
            shape.hidden(),
            &mut self.hidden,
        )?;
        ops.layer_norm_rows(
            &self.hidden,
            &prepared.weights.embedding_norm.weight,
            &prepared.weights.embedding_norm.bias,
            shape.hidden(),
            self.layer_norm_eps,
            &mut self.query,
        )?;
        std::mem::swap(&mut self.hidden, &mut self.query);

        for layer in &prepared.weights.layers {
            prepared.project(
                &layer.query,
                &self.hidden,
                shape.sequence(),
                &mut self.query,
            )?;
            prepared.project(&layer.key, &self.hidden, shape.sequence(), &mut self.key)?;
            prepared.project(
                &layer.value,
                &self.hidden,
                shape.sequence(),
                &mut self.value,
            )?;
            ops.attention_context(
                &self.query,
                &self.key,
                &self.value,
                &self.mask,
                shape.hidden(),
                shape.heads(),
                &mut self.context,
            )?;
            prepared.project(
                &layer.attention_output,
                &self.context,
                shape.sequence(),
                &mut self.projection,
            )?;
            ops.add_in_place(&mut self.projection, &self.hidden)?;
            ops.layer_norm_rows(
                &self.projection,
                &layer.attention_norm.weight,
                &layer.attention_norm.bias,
                shape.hidden(),
                self.layer_norm_eps,
                &mut self.hidden,
            )?;

            prepared.project(
                &layer.intermediate,
                &self.hidden,
                shape.sequence(),
                &mut self.intermediate,
            )?;
            ops.gelu_erf(&mut self.intermediate)?;
            prepared.project(
                &layer.output,
                &self.intermediate,
                shape.sequence(),
                &mut self.projection,
            )?;
            ops.add_in_place(&mut self.projection, &self.hidden)?;
            ops.layer_norm_rows(
                &self.projection,
                &layer.output_norm.weight,
                &layer.output_norm.bias,
                shape.hidden(),
                self.layer_norm_eps,
                &mut self.hidden,
            )?;
        }

        // The pooler reads the first position only, so copy that row and project it
        // as a single-row input instead of running the head over every position.
        let stream = prepared.ops.stream();
        stream
            .memcpy_dtod(&self.hidden.slice(..shape.hidden()), &mut self.first_row)
            .map_err(|error| EncoderError::Driver(error.to_string()))?;
        prepared.project(
            &prepared.weights.pooler,
            &self.first_row,
            1,
            &mut self.pooled,
        )?;
        ops.tanh_apply(&mut self.pooled)?;
        Ok(())
    }
}

impl fmt::Debug for EncoderSubmission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncoderSubmission")
            .field("sequence", &self.shape.sequence())
            .field("device_bytes", &self.device_bytes())
            .field("complete", &self.is_complete().unwrap_or(false))
            .finish_non_exhaustive()
    }
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
