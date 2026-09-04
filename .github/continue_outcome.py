from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


# A backend execution needs to return both measurements and any sampled token.
p = Path("crates/core/src/execution.rs")
s = p.read_text()
marker = """#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionEvent {
"""
outcome = """#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionOutcome {
    metrics: ExecutionMetrics,
    output_token: Option<u32>,
}

impl ExecutionOutcome {
    #[must_use]
    pub const fn new(metrics: ExecutionMetrics) -> Self {
        Self {
            metrics,
            output_token: None,
        }
    }

    #[must_use]
    pub const fn with_output_token(mut self, token: u32) -> Self {
        self.output_token = Some(token);
        self
    }

    #[must_use]
    pub const fn metrics(self) -> ExecutionMetrics {
        self.metrics
    }

    #[must_use]
    pub const fn output_token(self) -> Option<u32> {
        self.output_token
    }
}

"""
if outcome not in s:
    s = replace_once(s, marker, outcome + marker, "execution outcome insertion")
p.write_text(s)

p = Path("crates/core/src/lib.rs")
s = p.read_text()
s = replace_once(
    s,
    "    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionPhase,\n",
    "    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionOutcome,\n    ExecutionPhase,\n",
    "outcome export",
)
p.write_text(s)

p = Path("crates/core/src/nvidia.rs")
s = p.read_text()
s = replace_once(
    s,
    """use crate::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionPlan,
    ExecutionSegment,
};
""",
    """use crate::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionOutcome,
    ExecutionPlan, ExecutionSegment,
};
""",
    "nvidia imports",
)
s = s.replace(
    ") -> Result<ExecutionMetrics, BackendError>;",
    ") -> Result<ExecutionOutcome, BackendError>;",
    1,
)
s = s.replace(
    ") -> Result<Vec<ExecutionMetrics>, BackendError> {",
    ") -> Result<Vec<ExecutionOutcome>, BackendError> {",
    1,
)
s = replace_once(
    s,
    """        let metrics = self
            .dispatcher
            .dispatch_batch(plan, batch, plan.weights(), states)?;
        if metrics.len() != batch.len() {
            return Err(BackendError::ExecutionFailed(format!(
                "NVIDIA dispatcher returned {} metric records for {} request segments",
                metrics.len(),
                batch.len()
            )));
        }

        let events = batch
            .segments()
            .iter()
            .zip(metrics)
            .map(|(segment, metrics)| {
                ExecutionEvent::new(
                    segment.request(),
                    plan.policy_version(),
                    segment.phase(),
                    segment.token_count(),
                    metrics,
                )
                .ok_or_else(|| {
                    BackendError::ExecutionFailed("segment contained no tokens".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
""",
    """        let outcomes = self
            .dispatcher
            .dispatch_batch(plan, batch, plan.weights(), states)?;
        if outcomes.len() != batch.len() {
            return Err(BackendError::ExecutionFailed(format!(
                "NVIDIA dispatcher returned {} outcomes for {} request segments",
                outcomes.len(),
                batch.len()
            )));
        }

        let events = batch
            .segments()
            .iter()
            .zip(outcomes)
            .map(|(segment, outcome)| {
                let output_token = outcome.output_token();
                if segment.requests_sampling() != output_token.is_some() {
                    return Err(BackendError::ExecutionFailed(format!(
                        "NVIDIA dispatcher output-token presence did not match sampling for request {}",
                        segment.request().get()
                    )));
                }
                let event = ExecutionEvent::new(
                    segment.request(),
                    plan.policy_version(),
                    segment.phase(),
                    segment.token_count(),
                    outcome.metrics(),
                )
                .ok_or_else(|| {
                    BackendError::ExecutionFailed("segment contained no tokens".to_owned())
                })?;
                Ok(match output_token {
                    Some(token) => event.with_output_token(token),
                    None => event,
                })
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
""",
    "nvidia outcome mapping",
)
# Test dispatcher return signatures and values.
s = s.replace(
    ") -> Result<ExecutionMetrics, BackendError> {\n            Ok(ExecutionMetrics::new(12, 4, 8))",
    ") -> Result<ExecutionOutcome, BackendError> {\n            Ok(ExecutionOutcome::new(ExecutionMetrics::new(12, 4, 8)))",
    1,
)
s = s.replace(
    ") -> Result<ExecutionMetrics, BackendError> {\n            self.segment_calls += 1;\n            Ok(ExecutionMetrics::new(99, 0, 0))",
    ") -> Result<ExecutionOutcome, BackendError> {\n            self.segment_calls += 1;\n            Ok(ExecutionOutcome::new(ExecutionMetrics::new(99, 0, 0)))",
    1,
)
s = s.replace(
    ") -> Result<Vec<ExecutionMetrics>, BackendError> {",
    ") -> Result<Vec<ExecutionOutcome>, BackendError> {",
    1,
)
s = s.replace(
    ".map(|_| ExecutionMetrics::new(7, 0, 0))",
    ".map(|_| ExecutionOutcome::new(ExecutionMetrics::new(7, 0, 0)))",
    1,
)
# Make sampling import available for the contract test.
s = replace_once(
    s,
    "    use crate::request::RequestId;\n",
    "    use crate::request::{RequestId, SamplingParams};\n",
    "sampling import",
)
marker = """    #[test]
    fn backend_forwards_the_whole_scheduler_batch_to_the_dispatcher() {
"""
test = """    #[test]
    fn backend_rejects_missing_requested_output_token() {
        let device = DeviceId::new(0);
        let backend_id = BackendId::new("cuda").expect("backend ID");
        let capabilities = BackendCapabilities::new(
            backend_id.clone(),
            device,
            BackendKind::Cuda,
            24 * 1024 * 1024 * 1024,
            BackendFeatures::new(vec![DataType::F16], vec![], false, true),
        );
        let mut backend = NvidiaBackend::new(capabilities, TestDispatcher).expect("CUDA backend");
        let model = ModelId::new("test-model").expect("model ID");
        let plan = ExecutionPlan::new(
            model.clone(),
            backend_id,
            device,
            PolicyVersion::new(1).expect("policy version"),
            vec![ExecutionStage::new(
                ModelRegionId::new(0),
                ExecutionPhase::Decode,
            )],
            Vec::new(),
            WeightBinding::empty(model, device),
        )
        .expect("plan");
        let segment = ExecutionSegment::new(
            RequestId::new(1).expect("request ID"),
            ExecutionPhase::Decode,
            1,
            1,
            0,
            Vec::new(),
        )
        .expect("segment")
        .with_sampling(SamplingParams::greedy(None));
        let batch = ExecutionBatch::new(vec![segment]).expect("batch");
        let mut states = vec![InferenceStateSet::new(Vec::new()).expect("state")];

        assert!(matches!(
            backend.submit(&plan, &batch, &mut states),
            Err(BackendError::ExecutionFailed(_))
        ));
    }

"""
if test not in s:
    s = replace_once(s, marker, test + marker, "nvidia contract test")
p.write_text(s)

# Runtime's reference dispatcher adopts the generic outcome type.
p = Path("crates/core/src/runtime.rs")
s = p.read_text()
s = replace_once(
    s,
    "    use crate::execution::{ExecutionMetrics, ExecutionPhase, ExecutionStage};\n",
    "    use crate::execution::{ExecutionMetrics, ExecutionOutcome, ExecutionPhase, ExecutionStage};\n",
    "runtime outcome import",
)
s = replace_once(
    s,
    """        ) -> Result<ExecutionMetrics, BackendError> {
            Ok(ExecutionMetrics::new(20, 0, 0))
        }
""",
    """        ) -> Result<ExecutionOutcome, BackendError> {
            Ok(ExecutionOutcome::new(ExecutionMetrics::new(20, 0, 0)))
        }
""",
    "runtime dispatcher outcome",
)
p.write_text(s)

# CUDA reference execution remains metrics-only but now returns an outcome.
p = Path("crates/nvidia/src/cuda.rs")
s = p.read_text()
s = replace_once(
    s,
    "    BackendError, DataType, ExecutionMetrics, ExecutionPlan, ExecutionSegment, F32BlockStream,\n",
    "    BackendError, DataType, ExecutionMetrics, ExecutionOutcome, ExecutionPlan, ExecutionSegment,\n    F32BlockStream,\n",
    "cuda outcome import",
)
s = replace_once(
    s,
    """    ) -> Result<ExecutionMetrics, BackendError> {
        if segment.batch_size() != 1 || segment.token_count() != 1 {
""",
    """    ) -> Result<ExecutionOutcome, BackendError> {
        if segment.batch_size() != 1 || segment.token_count() != 1 {
""",
    "cuda dispatcher signature",
)
s = replace_once(
    s,
    """        self.run_linear_layer(weights)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
""",
    """        self.run_linear_layer(weights)
            .map(ExecutionOutcome::new)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))
""",
    "cuda outcome wrapping",
)
p.write_text(s)
