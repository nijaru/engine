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
use std::sync::{Arc, Mutex, PoisonError};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream};
use engine_nvidia::{CudaEncoderKernelError, CudaEncoderOps};
use ribn_foundation::PoolLease;
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
    /// The device failed and completion could not be established, so the encoder
    /// keeps quarantined work and refuses new submissions until it can.
    Faulted(String),
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
            Self::Faulted(message) => write!(
                f,
                "encoder is faulted and retains quarantined device work: {message}"
            ),
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
    retirement: Mutex<Retirement>,
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
            retirement: Mutex::new(Retirement::default()),
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
    /// A failure that leaves device work running keeps that storage in the encoder's
    /// retirement list, because no caller may free storage it cannot prove complete.
    /// The caller that holds a shared-pool reservation should use
    /// [`Self::enqueue`] instead, so the charge covering that storage travels with
    /// it.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for an invalid request, a CUDA/cuBLAS failure, or a
    /// faulted encoder.
    pub fn submit(&self, request: &EncoderRequest) -> Result<EncoderSubmission, EncoderError> {
        match self.enqueue(request) {
            Ok(submission) => Ok(submission),
            Err(failure) => Err(self.retire_failure(failure, None)),
        }
    }

    /// Enqueue one request, reporting what the attempt left behind on failure.
    ///
    /// The caller owns the reservation covering this request's storage, so it — not
    /// this method — pairs the two when the failure returned storage that may still
    /// be running.
    pub(crate) fn enqueue(
        &self,
        request: &EncoderRequest,
    ) -> Result<EncoderSubmission, EnqueueFailure> {
        if let Some(fault) = self.fault() {
            return Err(EnqueueFailure::refused(EncoderError::Faulted(fault)));
        }
        let shape = crate::request::shape_of(&self.config, request).map_err(|constraint| {
            EnqueueFailure::refused(EncoderError::InvalidRequest(constraint))
        })?;
        let stream = self.stream();
        // Buffers released here are safe: a dropped device buffer is either freed in
        // stream order or freed after an explicit drain, so nothing is returned to the
        // allocator while the device can still write it. The charge is a different
        // question, and that is what the retirement list answers.
        let upload_tokens = |values: &[u32]| -> Result<CudaSlice<u32>, EnqueueFailure> {
            stream
                .clone_htod(values)
                .map_err(|error| EnqueueFailure::refused(EncoderError::Driver(error.to_string())))
        };
        let mask: Vec<i32> = request.attention_mask.clone();
        let mut submission = EncoderSubmission {
            layer_norm_eps: self.config.layer_norm_eps,
            prepared: Arc::clone(&self.prepared),
            shape,
            tokens: upload_tokens(&request.token_ids)?,
            types: upload_tokens(&request.token_type_ids)?,
            mask: stream.clone_htod(&mask).map_err(|error| {
                EnqueueFailure::refused(EncoderError::Driver(error.to_string()))
            })?,
            hidden: alloc(stream, shape.hidden_values()).map_err(EnqueueFailure::refused)?,
            query: alloc(stream, shape.hidden_values()).map_err(EnqueueFailure::refused)?,
            key: alloc(stream, shape.hidden_values()).map_err(EnqueueFailure::refused)?,
            value: alloc(stream, shape.hidden_values()).map_err(EnqueueFailure::refused)?,
            context: alloc(stream, shape.hidden_values()).map_err(EnqueueFailure::refused)?,
            projection: alloc(stream, shape.hidden_values()).map_err(EnqueueFailure::refused)?,
            intermediate: alloc(stream, shape.intermediate_values())
                .map_err(EnqueueFailure::refused)?,
            first_row: alloc(stream, shape.hidden()).map_err(EnqueueFailure::refused)?,
            pooled: alloc(stream, shape.hidden()).map_err(EnqueueFailure::refused)?,
            completion: None,
        };
        // From here the submission is complete and owns live device state: a failure
        // returns it so its owner can keep it until completion is proven.
        if let Err(error) = submission.run() {
            return Err(EnqueueFailure::partial(error, submission));
        }
        #[cfg(test)]
        if tests::take_injected_enqueue_failure() {
            // A real device only reaches this path on an OOM or a device fault, which a
            // qualification run cannot schedule; the injection keeps the retirement
            // path itself testable on device.
            return Err(EnqueueFailure::partial(
                EncoderError::Driver("injected post-enqueue failure".to_owned()),
                submission,
            ));
        }
        #[cfg(test)]
        if tests::take_injected_unrecorded_event() {
            // The forward pass is enqueued, but no completion event exists. That is the
            // state a consumer must treat as "completion not established" instead of
            // assuming the result is ready; a healthy device leaves this window too
            // short to schedule on demand.
            return Ok(submission);
        }
        match stream.record_event(None) {
            Ok(event) => {
                submission.completion = Some(event);
                Ok(submission)
            }
            Err(error) => Err(EnqueueFailure::partial(
                EncoderError::Driver(error.to_string()),
                submission,
            )),
        }
    }

    /// Keep storage a failed enqueue may still be running, with its reservation.
    ///
    /// A proven drain is the only thing that releases either, so this drains
    /// opportunistically and records a fault when it cannot. The caller gets its own
    /// error back unchanged.
    pub(crate) fn retire_failure(
        &self,
        failure: EnqueueFailure,
        lease: Option<PoolLease>,
    ) -> EncoderError {
        let EnqueueFailure { error, storage } = failure;
        if let Some(submission) = storage {
            self.quarantine(*submission, lease);
            let _ = self.drain_retirement();
        }
        error
    }

    /// Establish completion of every quarantined request and release what is proven
    /// done.
    ///
    /// A successful drain is the only evidence that permits release. While it fails,
    /// the encoder keeps the storage and the charge and refuses new submissions:
    /// releasing a charge early would let another owner reserve bytes the device is
    /// still holding, and the storage behind an outstanding stream-ordered free has
    /// not returned to the allocator either.
    ///
    /// # Errors
    /// Returns the device error that prevented completion.
    pub fn drain_retirement(&self) -> Result<usize, EncoderError> {
        let (pending, faulted) = {
            let retirement = self.retirement();
            (retirement.quarantined.len(), retirement.fault.is_some())
        };
        if pending == 0 && !faulted {
            return Ok(0);
        }
        #[cfg(test)]
        if tests::take_injected_drain_failure() {
            let message = "injected drain failure".to_owned();
            self.retirement().fault = Some(message.clone());
            return Err(EncoderError::Driver(message));
        }
        if let Err(error) = self.stream().synchronize() {
            let message = error.to_string();
            self.retirement().fault = Some(message);
            return Err(EncoderError::Driver(error.to_string()));
        }
        let mut retirement = self.retirement();
        let drained = retirement.quarantined.len();
        retirement.quarantined.clear();
        retirement.fault = None;
        Ok(drained)
    }

    /// Requests whose device work is still held by this encoder.
    #[must_use]
    pub fn quarantined_requests(&self) -> usize {
        self.retirement().quarantined.len()
    }

    /// Device bytes this encoder still holds for work that may not have completed.
    #[must_use]
    pub fn quarantined_bytes(&self) -> u64 {
        self.retirement()
            .quarantined
            .iter()
            .map(|entry| entry.submission.device_bytes())
            .sum()
    }

    /// Shared-pool bytes this encoder still holds for quarantined work.
    ///
    /// A quarantined request keeps the reservation that was granted for it, so this
    /// equals its device bytes whenever a shared pool granted one. It is zero on the
    /// standalone submission path, which has no pool to account to.
    #[must_use]
    pub fn quarantined_charge(&self) -> u64 {
        self.retirement()
            .quarantined
            .iter()
            .filter_map(|entry| entry.lease.as_ref())
            .fold(0_u64, |total, lease| total.saturating_add(lease.bytes()))
    }

    /// Whether the device failed and this encoder still holds quarantined work.
    #[must_use]
    pub fn is_faulted(&self) -> bool {
        self.retirement().fault.is_some()
    }

    /// The fault that stopped this encoder from accepting work, when it has one.
    #[must_use]
    pub fn fault(&self) -> Option<String> {
        self.retirement().fault.clone()
    }

    /// Keep one live submission, and the charge covering it, until a proven drain.
    pub(crate) fn quarantine(&self, submission: EncoderSubmission, lease: Option<PoolLease>) {
        self.retirement()
            .quarantined
            .push(Quarantined { submission, lease });
    }

    fn retirement(&self) -> std::sync::MutexGuard<'_, Retirement> {
        // The retirement list holds no arithmetic that can panic while locked, so
        // recovering the guard keeps one failed device from becoming a permanent
        // panic for every later release attempt.
        self.retirement
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
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

/// Device work the encoder may still hold after a failed or uncommittable
/// submission.
///
/// Release waits on a proven stream drain, not on the fact that an error was
/// reported: the storage may still be running, and the reservoir covering it is not
/// free until the device has reached the stream-ordered free of that storage.
#[derive(Default)]
struct Retirement {
    quarantined: Vec<Quarantined>,
    /// Set when a drain failed; new submissions are refused until one succeeds.
    fault: Option<String>,
}

/// One quarantined request: its device storage and the charge covering it.
///
/// The charge is absent only on the standalone submission path, which has no shared
/// pool to account to. Whenever a reservation exists, the two stay here together.
struct Quarantined {
    submission: EncoderSubmission,
    lease: Option<PoolLease>,
}

/// Why one request could not be enqueued, and what the encoder still owns.
pub(crate) struct EnqueueFailure {
    error: EncoderError,
    /// Device storage the failed attempt already created. It may still be running,
    /// so its owner must keep it — with the reservation covering it — until a proven
    /// drain. Boxed because a submission owns a whole activation workspace and this
    /// travels as an error.
    storage: Option<Box<EncoderSubmission>>,
}

impl EnqueueFailure {
    /// A failure that left no owned storage: validation, allocation or upload.
    fn refused(error: EncoderError) -> Self {
        Self {
            error,
            storage: None,
        }
    }

    /// A failure after the request's device state existed and was already enqueued.
    fn partial(error: EncoderError, storage: EncoderSubmission) -> Self {
        Self {
            error,
            storage: Some(Box::new(storage)),
        }
    }

    /// Whether the attempt created device storage that is still owned.
    #[must_use]
    pub(crate) const fn owns_storage(&self) -> bool {
        self.storage.is_some()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ribn_foundation::BytePool;

    /// Countdown of submissions whose enqueue is deliberately failed after the
    /// forward pass was already launched.
    ///
    /// Only a device fault or an allocation failure reaches this path in production,
    /// and neither can be scheduled on demand, so the retirement cycle around it is
    /// injected rather than assumed.
    static INJECTED_ENQUEUE_FAILURES: AtomicUsize = AtomicUsize::new(0);

    /// Countdown of drains that are deliberately reported as unprovable.
    static INJECTED_DRAIN_FAILURES: AtomicUsize = AtomicUsize::new(0);

    /// Countdown of requests whose completion event is deliberately not recorded, so
    /// the consumer sees an unestablished completion.
    static INJECTED_UNRECORDED_EVENTS: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn take_injected_enqueue_failure() -> bool {
        take(&INJECTED_ENQUEUE_FAILURES)
    }

    pub(super) fn take_injected_drain_failure() -> bool {
        take(&INJECTED_DRAIN_FAILURES)
    }

    pub(super) fn take_injected_unrecorded_event() -> bool {
        take(&INJECTED_UNRECORDED_EVENTS)
    }

    fn take(counter: &AtomicUsize) -> bool {
        counter
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
    }

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../batch/tests/fixtures/bert-tiny")
    }

    fn encoder() -> CudaBertEncoder {
        CudaBertEncoder::load(fixture(), 0).expect("prepare encoder")
    }

    /// A quarantine that cannot be released must keep the charge covering it, refuse
    /// new work, and return to service once a drain proves completion.
    #[test]
    #[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
    fn an_unprovable_drain_keeps_the_charge_and_refuses_new_work() {
        let encoder = encoder();
        let request = EncoderRequest::single_segment(vec![4, 1, 9, 3]);
        let envelope = encoder.request_bytes(request.sequence()).expect("envelope");
        let pool = BytePool::new(envelope).shared();
        let lease = pool.reserve(envelope).expect("reservation");

        // The forward pass runs, then the submission fails and the drain that would
        // release it cannot be proven.
        INJECTED_ENQUEUE_FAILURES.store(1, Ordering::SeqCst);
        INJECTED_DRAIN_FAILURES.store(1, Ordering::SeqCst);
        let failure = encoder
            .enqueue(&request)
            .expect_err("injected post-enqueue failure");
        assert!(failure.owns_storage());
        let error = encoder.retire_failure(failure, Some(lease));
        assert!(matches!(error, EncoderError::Driver(_)));

        assert_eq!(encoder.quarantined_requests(), 1);
        assert_eq!(encoder.quarantined_bytes(), envelope);
        assert_eq!(
            encoder.quarantined_charge(),
            envelope,
            "the charge covering quarantined storage must stay with it"
        );
        assert_eq!(
            pool.granted(),
            envelope,
            "a rejected work submission must not free bytes the device still covers"
        );
        assert!(encoder.is_faulted());
        assert!(
            matches!(encoder.submit(&request), Err(EncoderError::Faulted(_))),
            "a faulted encoder refuses new work"
        );
        assert!(
            encoder.drain_retirement().is_ok(),
            "the injected drain failure was spent, so the device can be proven complete"
        );

        assert_eq!(encoder.quarantined_requests(), 0);
        assert_eq!(encoder.quarantined_charge(), 0);
        assert_eq!(pool.granted(), 0, "a proven drain releases the charge");
        assert!(!encoder.is_faulted());
        assert_eq!(encoder.drain_retirement().expect("idle drain"), 0);
        assert!(
            encoder.submit(&request).is_ok(),
            "the encoder is usable again"
        );
    }
    /// A result whose completion is not established must reach its consumer as such:
    /// the runtime hands it over without waiting, and the consumer has to await the
    /// dependency before reading.
    #[test]
    #[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
    fn an_unestablished_completion_is_handed_over_without_stalling_the_runtime() {
        use crate::executor::EncoderCompletion;
        use ribn_batch::{BatchConfig, BatchRuntime, StepOutcome};

        let encoder = encoder();
        let request = EncoderRequest::single_segment(vec![4, 1, 9, 3]);
        let envelope = encoder.request_bytes(request.sequence()).expect("envelope");
        let pool = BytePool::new(2 * envelope).shared();
        let mut runtime = BatchRuntime::new(
            encoder,
            Arc::clone(&pool),
            BatchConfig {
                max_waiting_requests: 4,
                max_retained_results: 4,
            },
        )
        .expect("runtime");

        INJECTED_UNRECORDED_EVENTS.store(1, Ordering::SeqCst);
        runtime.submit(request.clone()).expect("first request");
        runtime.submit(request).expect("second request");
        // Both requests are admitted and enqueued without waiting for the device.
        assert_eq!(
            runtime.step().expect("step"),
            StepOutcome::Executed { results: 2 }
        );
        assert_eq!(pool.granted(), 2 * envelope);

        let completion =
            EncoderCompletion::from_completed(runtime.pop_completed().expect("first entry"));
        let EncoderCompletion::Result(result) = completion else {
            panic!("an executing request is not a rejection");
        };
        assert!(
            !result.is_complete().expect("completion query"),
            "the consumer must see that completion is not established"
        );
        // The producer dependency is what makes the storage readable, and the charge
        // covering it is held until the consumer releases the result.
        result.synchronize().expect("synchronize");
        assert!(result.is_complete().expect("completion query"));
        let output = result.read().expect("read");
        assert!(output.pooled.iter().all(|value| value.is_finite()));
        assert_eq!(runtime.executor().quarantined_requests(), 0);
        drop(result);
        assert_eq!(pool.granted(), envelope);
        drop(runtime.pop_completed().expect("second entry"));
        assert_eq!(pool.granted(), 0);
    }
}
