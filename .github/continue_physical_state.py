from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


# Give every backend a narrow inference-state lifecycle hook. Backends without
# physical request state keep the default no-op.
p = Path("crates/core/src/backend.rs")
s = p.read_text()
s = replace_once(
    s,
    """    /// Submit a multi-request batch without requiring host synchronization
    /// with its completion.
""",
    """    /// Release backend-owned physical resources associated with one logical
    /// inference-state set. Stateless backends may keep the default no-op.
    ///
    /// This is a lifecycle operation, not model execution. Callers must not
    /// release state while a submission still owns it.
    ///
    /// # Errors
    ///
    /// Returns a backend error when physical state cannot be released safely.
    fn release_inference_state(
        &mut self,
        _state: &InferenceStateSet,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Submit a multi-request batch without requiring host synchronization
    /// with its completion.
""",
    "backend release hook",
)
p.write_text(s)

# Runtime orders physical release before logical identity/capacity release.
p = Path("crates/core/src/runtime.rs")
s = p.read_text()
s = replace_once(
    s,
    """    /// Release every logical state allocation owned by one finished request.
    ///
    /// Physical backend state remains a backend concern; this closes the core
    /// allocation lifecycle so request reclamation cannot leak `StateManager`
    /// capacity.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::State`] when a state handle is invalid or its
    /// manager cannot release it.
    pub fn release_state_set(&mut self, state: &InferenceStateSet) -> Result<(), RuntimeError> {
        for value in state.states() {
            self.state_manager.release(value.handle().clone())?;
        }
        Ok(())
    }
""",
    """    /// Release backend-owned physical state, then release the logical state
    /// identities and capacity owned by one finished request.
    ///
    /// Keeping this ordering preserves the logical handles needed by a backend
    /// to find its physical allocations. If physical release fails, logical
    /// identity is retained rather than silently orphaning device memory.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Backend`] when physical state release fails or
    /// [`RuntimeError::State`] when a logical state handle cannot be released.
    pub fn release_state_set(&mut self, state: &InferenceStateSet) -> Result<(), RuntimeError> {
        self.backend
            .release_inference_state(state)
            .map_err(RuntimeError::Backend)?;
        for value in state.states() {
            self.state_manager.release(value.handle().clone())?;
        }
        Ok(())
    }
""",
    "runtime ordered release",
)
p.write_text(s)

# NVIDIA forwards lifecycle to the concrete dispatcher, just as it forwards
# execution. The dispatcher remains the owner of backend-specific resources.
p = Path("crates/core/src/nvidia.rs")
s = p.read_text()
s = replace_once(
    s,
    """    /// Dispatch one scheduler-selected multi-request batch.
""",
    """    /// Release dispatcher-owned physical state associated with one logical
    /// inference-state set. Stateless/reference dispatchers use the no-op.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the dispatcher cannot release the state.
    fn release_inference_state(
        &mut self,
        _state: &InferenceStateSet,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Dispatch one scheduler-selected multi-request batch.
""",
    "nvidia dispatcher release hook",
)
s = replace_once(
    s,
    """    fn submit(
        &mut self,
        plan: &ExecutionPlan,
""",
    """    fn release_inference_state(
        &mut self,
        state: &InferenceStateSet,
    ) -> Result<(), BackendError> {
        self.dispatcher.release_inference_state(state)
    }

    fn submit(
        &mut self,
        plan: &ExecutionPlan,
""",
    "nvidia backend release forwarding",
)
# Make the existing batch-aware test dispatcher observable for lifecycle tests.
s = replace_once(
    s,
    """    struct BatchAwareDispatcher {
        batch_calls: usize,
        segment_calls: usize,
    }
""",
    """    struct BatchAwareDispatcher {
        batch_calls: usize,
        segment_calls: usize,
        release_calls: usize,
    }
""",
    "batch-aware release count",
)
needle = """        fn dispatch_batch(
            &mut self,
            _plan: &ExecutionPlan,
            batch: &ExecutionBatch,
            _weights: &WeightBinding,
            states: &mut [InferenceStateSet],
        ) -> Result<Vec<ExecutionOutcome>, BackendError> {
            if batch.len() != states.len() {
                return Err(BackendError::StateCountMismatch);
            }
            self.batch_calls += 1;
            Ok((0..batch.len())
                .map(|_| ExecutionOutcome::new(ExecutionMetrics::new(7, 0, 0)))
                .collect())
        }
"""
replacement = needle[:-2] + """

        fn release_inference_state(
            &mut self,
            _state: &InferenceStateSet,
        ) -> Result<(), BackendError> {
            self.release_calls += 1;
            Ok(())
        }
    }
"""
s = replace_once(s, needle, replacement, "batch-aware release implementation")
marker = """    #[test]
    fn backend_forwards_the_whole_scheduler_batch_to_the_dispatcher() {
"""
test = """    #[test]
    fn backend_forwards_state_release_to_the_dispatcher() {
        let device = DeviceId::new(0);
        let capabilities = BackendCapabilities::new(
            BackendId::new("cuda").expect("backend ID"),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(vec![DataType::F16], vec![], false, true),
        );
        let mut backend = NvidiaBackend::new(capabilities, BatchAwareDispatcher::default())
            .expect("CUDA backend");
        let state = InferenceStateSet::new(Vec::new()).expect("state");

        backend
            .release_inference_state(&state)
            .expect("release state");
        assert_eq!(backend.dispatcher().release_calls, 1);
    }

"""
if test not in s:
    s = replace_once(s, marker, test + marker, "nvidia release test")
p.write_text(s)

# CUDA physical state is keyed by stable logical StateId values, not request
# identity, so the same request can be rescheduled and state remains reusable.
p = Path("crates/nvidia/src/state.rs")
s = p.read_text()
s = replace_once(s, "use std::fmt;\n", "use std::collections::HashMap;\nuse std::fmt;\n", "HashMap import")
s = replace_once(
    s,
    """use engine_core::{
    DataType, DeviceId, InferenceStateSet, KvState, KvStateSpec, RecurrentState,
    RecurrentStateSpec, StateLocation,
};
""",
    """use engine_core::{
    DataType, DeviceId, InferenceStateSet, KvState, KvStateSpec, RecurrentState,
    RecurrentStateSpec, StateId, StateLocation,
};
""",
    "StateId import",
)
marker = """/// Backend-owned physical counterpart to a core [`InferenceStateSet`]. It is a
"""
registry = """#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaStateKey(Vec<StateId>);

impl CudaStateKey {
    fn from_state_set(state: &InferenceStateSet) -> Self {
        let mut ids = state
            .states()
            .iter()
            .map(|value| value.handle().id())
            .collect::<Vec<_>>();
        ids.sort_unstable_by_key(|id| id.get());
        Self(ids)
    }
}

/// Persistent backend ownership for physical CUDA request state.
///
/// Logical [`StateId`] values are the identity boundary. Request IDs are not
/// used because scheduling/lifecycle identity and model state identity are
/// deliberately separate in core.
pub struct CudaStateRegistry {
    stream: Arc<CudaStream>,
    states: HashMap<CudaStateKey, CudaHybridState>,
}

impl CudaStateRegistry {
    #[must_use]
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            states: HashMap::new(),
        }
    }

    /// Materialize physical state on first use and reuse it on later steps.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when CUDA allocation fails or the persistent
    /// physical prefix no longer matches the authoritative logical prefix.
    pub fn get_or_create(
        &mut self,
        state: &InferenceStateSet,
    ) -> Result<&mut CudaHybridState, CudaStateError> {
        let key = CudaStateKey::from_state_set(state);
        let expected = state.token_position().unwrap_or(0);
        if !self.states.contains_key(&key) {
            let physical = CudaHybridState::from_state_set(self.stream.clone(), state)?;
            self.states.insert(key.clone(), physical);
        }
        let physical = self.states.get_mut(&key).ok_or(CudaStateError::RegistryInvariant)?;
        if physical.token_position() != expected {
            return Err(CudaStateError::PositionMismatch {
                logical: expected,
                physical: physical.token_position(),
            });
        }
        Ok(physical)
    }

    /// Drop any physical allocation associated with this logical state set.
    /// Missing state is a valid no-op for requests cancelled before first use.
    pub fn release(&mut self, state: &InferenceStateSet) -> bool {
        self.states
            .remove(&CudaStateKey::from_state_set(state))
            .is_some()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

"""
if registry not in s:
    s = replace_once(s, marker, registry + marker, "CUDA state registry")
s = replace_once(
    s,
    """    PositionRegression {
        current: u32,
        requested: u32,
    },
""",
    """    PositionRegression {
        current: u32,
        requested: u32,
    },
    PositionMismatch {
        logical: u32,
        physical: u32,
    },
    RegistryInvariant,
""",
    "CUDA state registry errors",
)
s = replace_once(
    s,
    """            Self::PositionRegression { current, requested } => write!(
                f,
                "physical state position cannot move from {current} back to {requested}"
            ),
""",
    """            Self::PositionRegression { current, requested } => write!(
                f,
                "physical state position cannot move from {current} back to {requested}"
            ),
            Self::PositionMismatch { logical, physical } => write!(
                f,
                "logical state is at position {logical}, but persistent CUDA state is at {physical}"
            ),
            Self::RegistryInvariant => f.write_str("CUDA state registry lost a materialized entry"),
""",
    "CUDA state registry error display",
)
p.write_text(s)

p = Path("crates/nvidia/src/lib.rs")
s = p.read_text()
s = replace_once(
    s,
    """pub use state::{
    CudaHybridState, CudaKvState, CudaRecurrentState, CudaStateBuffer, CudaStateError,
};
""",
    """pub use state::{
    CudaHybridState, CudaKvState, CudaRecurrentState, CudaStateBuffer, CudaStateError,
    CudaStateRegistry,
};
""",
    "CUDA registry export",
)
p.write_text(s)

# First serving dispatcher: correctness-oriented and explicitly sequential
# inside the whole-batch backend boundary. It preserves the scheduler/runtime
# batch contract while later kernels replace the inner implementation.
p = Path("crates/nvidia/src/serving.rs")
p.write_text(r'''use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaStream;
use engine_core::{
    BackendError, ExecutionBatch, ExecutionMetrics, ExecutionOutcome, ExecutionPhase,
    ExecutionPlan, ExecutionSegment, InferenceStateSet, NvidiaDispatcher, WeightBinding,
};

use crate::decode::CudaQwen35Decode;
use crate::state::CudaStateRegistry;

/// Correctness-first Qwen3.8 CUDA serving dispatcher.
///
/// The scheduler submits a whole multi-request batch. This implementation
/// initially executes its entries sequentially with one batch-1 Qwen executor;
/// that is intentional. It validates persistent state and serving semantics
/// before true batched kernels or overlapping streams are introduced.
pub struct CudaQwen35ServingDispatcher {
    executor: CudaQwen35Decode,
    states: CudaStateRegistry,
}

impl CudaQwen35ServingDispatcher {
    #[must_use]
    pub fn new(executor: CudaQwen35Decode, stream: Arc<CudaStream>) -> Self {
        Self {
            executor,
            states: CudaStateRegistry::new(stream),
        }
    }

    #[must_use]
    pub const fn state_registry(&self) -> &CudaStateRegistry {
        &self.states
    }

    fn dispatch_segment(
        &mut self,
        segment: &ExecutionSegment,
        state: &InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError> {
        if let Some(sampling) = segment.sampling()
            && sampling.temperature() != 0.0
        {
            return Err(BackendError::Unsupported(
                "non-greedy Qwen3.8 CUDA sampling on the correctness path",
            ));
        }

        let physical = self
            .states
            .get_or_create(state)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        let started = Instant::now();
        let mut output_token = None;

        match segment.phase() {
            ExecutionPhase::Prefill => {
                let tokens = segment
                    .token_input()
                    .and_then(engine_core::ExecutionTokenInput::prompt_slice)
                    .ok_or_else(|| {
                        BackendError::ExecutionFailed(
                            "Qwen prefill segment requires prompt token input".to_owned(),
                        )
                    })?;
                for (offset, &token) in tokens.iter().enumerate() {
                    let offset = u32::try_from(offset).map_err(|_| {
                        BackendError::ExecutionFailed("prefill offset overflowed".to_owned())
                    })?;
                    let position = segment.state_position().checked_add(offset).ok_or_else(|| {
                        BackendError::ExecutionFailed("prefill position overflowed".to_owned())
                    })?;
                    let chosen = self
                        .executor
                        .decode_step(physical, token, position)
                        .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                    let next = position.checked_add(1).ok_or_else(|| {
                        BackendError::ExecutionFailed("prefill position overflowed".to_owned())
                    })?;
                    physical
                        .advance_to(next)
                        .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                    if segment.requests_sampling() && offset + 1 == segment.token_count() {
                        output_token = Some(chosen);
                    }
                }
            }
            ExecutionPhase::Decode => {
                let token = segment
                    .token_input()
                    .and_then(engine_core::ExecutionTokenInput::decode_token)
                    .ok_or_else(|| {
                        BackendError::ExecutionFailed(
                            "Qwen decode segment requires one decode token".to_owned(),
                        )
                    })?;
                let chosen = self
                    .executor
                    .decode_step(physical, token, segment.state_position())
                    .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                let next = segment.state_position().checked_add(1).ok_or_else(|| {
                    BackendError::ExecutionFailed("decode position overflowed".to_owned())
                })?;
                physical
                    .advance_to(next)
                    .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                if segment.requests_sampling() {
                    output_token = Some(chosen);
                }
            }
            ExecutionPhase::SpecDraft
            | ExecutionPhase::SpecVerify
            | ExecutionPhase::Encoder
            | ExecutionPhase::MoEExpert => {
                return Err(BackendError::Unsupported(
                    "this Qwen3.8 CUDA serving execution phase",
                ));
            }
        }

        let elapsed_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let outcome = ExecutionOutcome::new(ExecutionMetrics::new(elapsed_nanos, 0, 0));
        Ok(match output_token {
            Some(token) => outcome.with_output_token(token),
            None => outcome,
        })
    }
}

impl NvidiaDispatcher for CudaQwen35ServingDispatcher {
    fn dispatch(
        &mut self,
        _plan: &ExecutionPlan,
        segment: &ExecutionSegment,
        _weights: &WeightBinding,
        state: &mut InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError> {
        self.dispatch_segment(segment, state)
    }

    fn dispatch_batch(
        &mut self,
        _plan: &ExecutionPlan,
        batch: &ExecutionBatch,
        _weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Vec<ExecutionOutcome>, BackendError> {
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        batch
            .segments()
            .iter()
            .zip(states)
            .map(|(segment, state)| self.dispatch_segment(segment, state))
            .collect()
    }

    fn release_inference_state(
        &mut self,
        state: &InferenceStateSet,
    ) -> Result<(), BackendError> {
        self.states.release(state);
        Ok(())
    }
}
''')

p = Path("crates/nvidia/src/lib.rs")
s = p.read_text()
s = replace_once(
    s,
    """#[cfg(feature = "cuda")]
mod staging;
#[cfg(feature = "cuda")]
mod state;
""",
    """#[cfg(feature = "cuda")]
mod staging;
#[cfg(feature = "cuda")]
mod state;
#[cfg(feature = "cuda")]
mod serving;
""",
    "serving module",
)
s += "\n#[cfg(feature = \"cuda\")]\npub use serving::CudaQwen35ServingDispatcher;\n"
p.write_text(s)
