//! Binding the device encoder to the non-autoregressive batching runtime.
//!
//! The runtime owns reservation and readiness; this module supplies the two things
//! only the model can supply: the concrete accepted range (from request shapes) and
//! the real device envelope each accepted request will hold. Execution enqueues
//! without waiting, so a result leaves [`ribn_batch::BatchRuntime`] device-resident
//! and incomplete, and the consumer receives it with its pool charge and its
//! producer completion dependency attached.
//!
//! The executor is also where a shared-pool reservation and its storage stay
//! together. Each accepted request arrives with the charge covering it; a request
//! whose enqueue fails keeps that charge with the storage it already created, so the
//! runtime never receives a reservation back to release on its own.
//!
//! `engine-bert` may depend on `ribn-batch` for the same reason `engine-qwen`
//! depends on `ribn`: a model crate implements the runtime's executor seam instead
//! of owning a second scheduler.

use ribn_batch::{
    BatchExecutor, BatchSelection, Completed, EnqueueError, Job, JobOutput, Rejection, RequestId,
    Terminal,
};
use ribn_foundation::{ParameterVersion, PoolLease};

use crate::cuda::{CudaBertEncoder, EncoderError, EncoderOutput, EncoderSubmission};
use crate::request::{self, EncoderConstraint, EncoderRequest};

/// Prepared weights never change while one encoder is loaded, so its parameter
/// version is constant. Replacing them constructs a new prepared encoder.
const ENCODER_PARAMETER_VERSION: ParameterVersion = ParameterVersion::new(0);

impl BatchExecutor for CudaBertEncoder {
    type Input = EncoderRequest;
    type Output = EncoderSubmission;
    type Error = EncoderError;
    type Constraint = EncoderConstraint;

    fn parameter_version(&self) -> ParameterVersion {
        ENCODER_PARAMETER_VERSION
    }

    fn max_batch_items(&self) -> usize {
        self.max_batch_items()
    }

    /// Accept the prefix whose shapes this encoder can execute.
    ///
    /// The range comes from concrete shapes, not from a per-row budget: the runtime
    /// then reserves each accepted request's real envelope from the shared pool and
    /// shrinks the range further to the largest prefix the pool can cover.
    fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
        request::select(self.config(), candidates)
    }

    /// The real device envelope of one request, or zero when its shape is
    /// permanently infeasible.
    ///
    /// An infeasible head is rejected by [`Self::select_batch`] before any
    /// reservation is attempted, so the zero here never covers executing work.
    fn retained_bytes(&self, input: &Self::Input) -> u64 {
        self.request_bytes(input.sequence()).unwrap_or(0)
    }

    /// Enqueue every accepted request without waiting for the device.
    ///
    /// Each job carries the reservation covering its storage. A request that fails
    /// before its device state exists releases its reservation here, because no
    /// storage covers it; a request that fails after keeps both, and the runtime is
    /// told the outcome is uncertain rather than clean.
    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, EnqueueError<Self::Error>> {
        let mut enqueued = Vec::with_capacity(batch.len());
        for job in batch {
            let (request, input, lease) = job.into_parts();
            match self.enqueue(&input) {
                Ok(submission) => enqueued.push(JobOutput::new(request, submission, lease)),
                Err(failure) => {
                    // Anything already on the device keeps its storage and charge; the
                    // failing request keeps its reservation only if it created storage.
                    let uncertain = !enqueued.is_empty() || failure.owns_storage();
                    let error = self.retire_failure(failure, Some(lease));
                    self.retire(enqueued);
                    let _ = self.drain_retirement();
                    return Err(if uncertain {
                        EnqueueError::Uncertain(error)
                    } else {
                        EnqueueError::Refused(error)
                    });
                }
            }
        }
        Ok(enqueued)
    }

    /// Keep output the runtime cannot commit, with the charge covering it.
    ///
    /// The runtime calls this when the returned results do not match its work, so
    /// nothing may be delivered. Storage and charge wait here for a proven drain
    /// instead of returning to the pool while the device may still write them.
    fn retire(&mut self, outputs: Vec<JobOutput<Self::Output>>) {
        for output in outputs {
            let (_, submission, lease) = output.into_parts();
            self.quarantine(submission, Some(lease));
        }
        let _ = self.drain_retirement();
    }
}

/// One terminal encoder completion.
///
/// This is the handoff surface for the encoder path: a request that executed is
/// returned as an [`EncoderResult`] that owns its device storage and the pool charge
/// covering it together, and a request that never executed is returned as its own
/// rejection, which carries neither.
#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "the executing variant is the common case, and boxing it would add an \
              allocation to every device result"
)]
pub enum EncoderCompletion {
    /// The request executed and its device-resident result is ready to be awaited.
    Result(EncoderResult),
    /// The request was refused before any device work; nothing is retained.
    Rejected {
        request: RequestId,
        rejection: Rejection<EncoderConstraint>,
    },
}

impl EncoderCompletion {
    /// Take a runtime completion, keeping the result with its charge.
    #[must_use]
    pub fn from_completed(completed: Completed<EncoderSubmission, EncoderConstraint>) -> Self {
        let request = completed.request();
        let (outcome, lease) = completed.into_parts();
        match outcome {
            Terminal::Output(submission) => Self::Result(EncoderResult {
                request,
                submission,
                lease,
            }),
            Terminal::Rejected(rejection) => Self::Rejected { request, rejection },
        }
    }
}

/// One completed encoder request handed to its consumer.
///
/// The device-resident result and the pool charge covering it travel together, so a
/// consumer cannot release the bytes while the storage is still readable. The
/// producer's completion dependency stays explicit: [`Self::is_complete`] or
/// [`Self::synchronize`] must establish completion before [`Self::read`], and the
/// storage is only free for reuse once this value is dropped.
#[derive(Debug)]
pub struct EncoderResult {
    request: RequestId,
    submission: EncoderSubmission,
    lease: Option<PoolLease>,
}

impl EncoderResult {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn submission(&self) -> &EncoderSubmission {
        &self.submission
    }

    /// Whether every kernel and copy of this request has completed.
    ///
    /// # Errors
    /// Returns [`EncoderError`] for a device fault.
    pub fn is_complete(&self) -> Result<bool, EncoderError> {
        self.submission.is_complete()
    }

    /// Block until this request's device work has completed.
    ///
    /// # Errors
    /// Returns [`EncoderError`] when the stream reports a device failure.
    pub fn synchronize(&self) -> Result<(), EncoderError> {
        self.submission.synchronize()
    }

    /// Read the results, after completion has been established.
    ///
    /// # Errors
    /// Returns [`EncoderError`] when a copy fails.
    pub fn read(&self) -> Result<EncoderOutput, EncoderError> {
        self.submission.read()
    }

    /// Bytes this result holds on the device until it is dropped.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.submission.device_bytes()
    }

    /// The pool charge covering this result's storage.
    #[must_use]
    pub const fn lease(&self) -> Option<&PoolLease> {
        self.lease.as_ref()
    }

    /// Pool bytes this result keeps reserved until it is dropped.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.lease.as_ref().map_or(0, PoolLease::bytes)
    }
}
