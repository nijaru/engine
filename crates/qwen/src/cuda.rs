use std::path::PathBuf;

use ribn::{
    Admission, BatchItem, ExecutionError, ExecutorInfo, GenerationExecutor, SequenceId,
    StepCompletion, SubmissionId, TokenRequest,
};

use crate::execution::model_error;
use crate::loading::{self, MemoryReport, QwenLoadOptions, Resources};

/// Experimental prepared Qwen GGUF/CUDA text implementation.
///
/// The existing CUDA kernels are reused, but integration with the new scheduler
/// still requires hardware qualification. This is not an automatically qualified
/// execution variant, a generic model loader, or a multimodal implementation.
pub struct QwenCuda {
    info: ExecutorInfo,
    memory: MemoryReport,
    resources: Option<Resources>,
}

impl QwenCuda {
    /// Load weights, prepare kernels, and reserve logical sequence capacity.
    ///
    /// # Errors
    /// Reports unsupported artifacts, invalid options, memory budget failure,
    /// or CUDA preparation errors before an engine can accept requests.
    pub fn load_gguf(
        path: impl Into<PathBuf>,
        options: QwenLoadOptions,
    ) -> Result<Self, ExecutionError> {
        let (resources, memory) = loading::load(path.into(), options)?;
        Ok(Self {
            info: resources.execution.info().clone(),
            resources: Some(resources),
            memory,
        })
    }

    #[must_use]
    pub const fn memory_report(&self) -> MemoryReport {
        self.memory
    }

    fn resources(&mut self) -> Result<&mut Resources, ExecutionError> {
        self.resources
            .as_mut()
            .ok_or_else(|| ExecutionError::new("Qwen device owner is unavailable"))
    }
}

impl GenerationExecutor for QwenCuda {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }
    fn admit(
        &mut self,
        id: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        self.resources()?.execution.admit(id, request)
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        self.resources()?.execution.submit(batch)
    }
    fn poll(&mut self, id: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        self.resources()?.execution.poll(id)
    }
    fn release(&mut self, id: SequenceId) -> Result<(), ExecutionError> {
        self.resources()?.execution.release(id)
    }
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        let resources = self.resources()?;
        resources.stream.synchronize().map_err(model_error)?;
        resources.execution.drain_after_barrier()
    }
}

impl Drop for QwenCuda {
    fn drop(&mut self) {
        if self.resources.is_some()
            && self.synchronize().is_err()
            && let Some(resources) = self.resources.take()
        {
            // No barrier proof: retain the single owner instead of releasing
            // allocations, pinned output, or streams still visible to the GPU.
            std::mem::forget(resources);
        }
    }
}
