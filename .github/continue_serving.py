from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


# Export the token-input boundary from core.
p = Path("crates/core/src/lib.rs")
s = p.read_text()
s = replace_once(
    s,
    "    ExecutionPlan, ExecutionSegment, ExecutionStage, PlanError,\n",
    "    ExecutionPlan, ExecutionSegment, ExecutionStage, ExecutionTokenInput, PlanError,\n",
    "lib export",
)
p.write_text(s)

# Fix the rustfmt-only failure in the previous commit.
p = Path("crates/core/src/execution.rs")
s = p.read_text()
s = replace_once(
    s,
    """    pub fn prompt(
        tokens: Arc<[u32]>,
        start: u32,
        token_count: u32,
    ) -> Result<Self, PlanError> {
""",
    """    pub fn prompt(tokens: Arc<[u32]>, start: u32, token_count: u32) -> Result<Self, PlanError> {
""",
    "execution rustfmt",
)
p.write_text(s)

# Connect persistent serving requests to execution token payloads.
p = Path("crates/core/src/serving_runtime.rs")
s = p.read_text()
s = replace_once(s, "use std::fmt;\n", "use std::fmt;\nuse std::sync::Arc;\n", "Arc import")
s = replace_once(
    s,
    "use crate::execution::{ExecutionBatch, ExecutionPlan, ExecutionSegment, PlanError};\n",
    """use crate::execution::{
    ExecutionBatch, ExecutionPlan, ExecutionSegment, ExecutionTokenInput, PlanError,
};
""",
    "execution imports",
)
s = replace_once(
    s,
    """    plan: ExecutionPlan,
    submissions: HashMap<BackendSubmissionId, RuntimeSubmission>,
}
""",
    """    plan: ExecutionPlan,
    submissions: HashMap<BackendSubmissionId, RuntimeSubmission>,
    prompt_tokens: HashMap<RequestId, Arc<[u32]>>,
    next_tokens: HashMap<RequestId, u32>,
}
""",
    "runtime fields",
)
s = replace_once(
    s,
    """            plan,
            submissions: HashMap::new(),
        })
""",
    """            plan,
            submissions: HashMap::new(),
            prompt_tokens: HashMap::new(),
            next_tokens: HashMap::new(),
        })
""",
    "runtime init",
)
s = replace_once(
    s,
    """    pub fn admit(
        &mut self,
        request: RequestSpec,
        state: InferenceStateSet,
        prompt_tokens: u32,
    ) -> Result<crate::serving::RequestSlotId, ServingRuntimeError> {
        Ok(self.scheduler.admit(request, state, prompt_tokens)?)
    }
""",
    """    pub fn admit(
        &mut self,
        request: RequestSpec,
        state: InferenceStateSet,
        prompt_tokens: Arc<[u32]>,
    ) -> Result<crate::serving::RequestSlotId, ServingRuntimeError> {
        if prompt_tokens.is_empty() {
            return Err(ServingRuntimeError::InvalidPrompt);
        }
        let prompt_len =
            u32::try_from(prompt_tokens.len()).map_err(|_| ServingRuntimeError::InvalidPrompt)?;
        let request_id = request.id();
        let slot = self.scheduler.admit(request, state, prompt_len)?;
        self.prompt_tokens.insert(request_id, prompt_tokens);
        Ok(slot)
    }
""",
    "admit",
)
s = replace_once(
    s,
    """    pub fn reclaim_next(&mut self) -> Result<Option<ActiveRequestSlot>, ServingRuntimeError> {
        Ok(self.scheduler.reclaim_next()?)
    }
""",
    """    pub fn reclaim_next(&mut self) -> Result<Option<ActiveRequestSlot>, ServingRuntimeError> {
        let reclaimed = self.scheduler.reclaim_next()?;
        if let Some(slot) = &reclaimed {
            let request = slot.request().id();
            self.prompt_tokens.remove(&request);
            self.next_tokens.remove(&request);
        }
        Ok(reclaimed)
    }
""",
    "reclaim",
)
s = replace_once(
    s,
    """                Ok(Some(completed)) => {
                    let (event, states) = completed.into_parts();
                    self.scheduler.complete_submission(id, &event, states)?;
                    completed_count += 1;
                }
""",
    """                Ok(Some(completed)) => {
                    let (event, states) = completed.into_parts();
                    self.scheduler.complete_submission(id, &event, states)?;
                    for completed in event.events() {
                        let request = completed.request();
                        if let Some(token) = completed.output_token() {
                            self.next_tokens.insert(request, token);
                        }
                        if completed.phase() == crate::execution::ExecutionPhase::Prefill {
                            let prompt_complete = self
                                .scheduler
                                .slot_for_request(request)
                                .and_then(|slot| self.scheduler.slots().get(slot))
                                .is_some_and(|slot| {
                                    slot.progress().prompt_processed()
                                        == slot.progress().prompt_tokens()
                                });
                            if prompt_complete {
                                self.prompt_tokens.remove(&request);
                            }
                        }
                    }
                    completed_count += 1;
                }
""",
    "completion token handoff",
)
s = replace_once(
    s,
    """        let segments = work
            .iter()
            .map(|item| {
                let mut segment = ExecutionSegment::new(
                    item.request(),
                    item.phase(),
                    1,
                    item.token_count(),
                    item.state_position(),
                    self.plan.state_requirements().to_vec(),
                )?;
                if item.requests_output() {
                    let slot = self
                        .scheduler
                        .slots()
                        .get(item.slot())
                        .ok_or(SchedulerError::StaleWork)?;
                    segment = segment.with_sampling(slot.request().semantics().sampling());
                }
                Ok(segment)
            })
            .collect::<Result<Vec<_>, ServingRuntimeError>>()?;
        Ok(ExecutionBatch::new(segments)?)
""",
    """        let segments = work
            .iter()
            .map(|item| {
                let mut segment = ExecutionSegment::new(
                    item.request(),
                    item.phase(),
                    1,
                    item.token_count(),
                    item.state_position(),
                    self.plan.state_requirements().to_vec(),
                )?;
                match item.phase() {
                    crate::execution::ExecutionPhase::Prefill => {
                        let prompt = self
                            .prompt_tokens
                            .get(&item.request())
                            .ok_or(ServingRuntimeError::PromptMissing(item.request()))?
                            .clone();
                        let input = ExecutionTokenInput::prompt(
                            prompt,
                            item.state_position(),
                            item.token_count(),
                        )?;
                        segment = segment.with_token_input(input)?;
                    }
                    crate::execution::ExecutionPhase::Decode => {
                        let token = self
                            .next_tokens
                            .get(&item.request())
                            .copied()
                            .ok_or(ServingRuntimeError::DecodeTokenMissing(item.request()))?;
                        segment = segment.with_token_input(ExecutionTokenInput::decode(token))?;
                    }
                    crate::execution::ExecutionPhase::SpecDraft
                    | crate::execution::ExecutionPhase::SpecVerify
                    | crate::execution::ExecutionPhase::Encoder
                    | crate::execution::ExecutionPhase::MoEExpert => {}
                }
                if item.requests_output() {
                    let slot = self
                        .scheduler
                        .slots()
                        .get(item.slot())
                        .ok_or(SchedulerError::StaleWork)?;
                    segment = segment.with_sampling(slot.request().semantics().sampling());
                }
                Ok(segment)
            })
            .collect::<Result<Vec<_>, ServingRuntimeError>>()?;
        Ok(ExecutionBatch::new(segments)?)
""",
    "execution batch",
)
s = replace_once(
    s,
    """pub enum ServingRuntimeError {
    PolicyMismatch,
    SubmissionCollision(BackendSubmissionId),
""",
    """pub enum ServingRuntimeError {
    PolicyMismatch,
    InvalidPrompt,
    PromptMissing(RequestId),
    DecodeTokenMissing(RequestId),
    SubmissionCollision(BackendSubmissionId),
""",
    "error variants",
)
s = replace_once(
    s,
    """            Self::PolicyMismatch => {
                f.write_str("scheduler and execution plan use different policy versions")
            }
            Self::SubmissionCollision(id) => {
""",
    """            Self::PolicyMismatch => {
                f.write_str("scheduler and execution plan use different policy versions")
            }
            Self::InvalidPrompt => {
                f.write_str("serving prompt must contain a representable token sequence")
            }
            Self::PromptMissing(request) => {
                write!(
                    f,
                    "prompt tokens for request {} are unavailable",
                    request.get()
                )
            }
            Self::DecodeTokenMissing(request) => {
                write!(
                    f,
                    "next decode token for request {} is unavailable",
                    request.get()
                )
            }
            Self::SubmissionCollision(id) => {
""",
    "error display",
)

# Test backend records the actual token payload crossing the runtime/backend boundary.
s = replace_once(
    s,
    """    struct DelayedBackend {
        capabilities: BackendCapabilities,
        next_submission: u64,
        pending: HashMap<BackendSubmissionId, (u8, ExecutionBatchEvent)>,
        fail_submit: bool,
    }
""",
    """    struct DelayedBackend {
        capabilities: BackendCapabilities,
        next_submission: u64,
        pending: HashMap<BackendSubmissionId, (u8, ExecutionBatchEvent)>,
        submitted_inputs: Vec<Vec<ExecutionTokenInput>>,
        fail_submit: bool,
    }
""",
    "test backend fields",
)
s = replace_once(
    s,
    """                capabilities,
                next_submission: 1,
                pending: HashMap::new(),
                fail_submit,
            }
""",
    """                capabilities,
                next_submission: 1,
                pending: HashMap::new(),
                submitted_inputs: Vec::new(),
                fail_submit,
            }
""",
    "test backend init",
)
s = replace_once(
    s,
    """            if self.fail_submit {
                return Err(BackendError::ExecutionFailed(
                    "test submit failure".to_owned(),
                ));
            }
            let id = BackendSubmissionId::new(self.next_submission)
""",
    """            if self.fail_submit {
                return Err(BackendError::ExecutionFailed(
                    "test submit failure".to_owned(),
                ));
            }
            let inputs = batch
                .segments()
                .iter()
                .map(|segment| {
                    segment.token_input().cloned().ok_or_else(|| {
                        BackendError::ExecutionFailed(
                            "test serving segment lacked token input".to_owned(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.submitted_inputs.push(inputs);
            let id = BackendSubmissionId::new(self.next_submission)
""",
    "test backend inputs",
)
s = replace_once(
    s,
    """    fn state() -> InferenceStateSet {
        InferenceStateSet::new(Vec::new()).expect("state")
    }
""",
    """    fn state() -> InferenceStateSet {
        InferenceStateSet::new(Vec::new()).expect("state")
    }

    fn prompt(tokens: &[u32]) -> Arc<[u32]> {
        Arc::from(tokens)
    }
""",
    "prompt test helper",
)

# Existing tests used zero prompt length to synthesize decode-only work. Give them
# real one-token prompts now that ServingRuntime owns concrete token payloads.
s = re.sub(
    r"serving\.admit\(request\((\d+)\), state\(\), 0\)",
    lambda m: f"serving.admit(request({m.group(1)}), state(), prompt(&[{10 + int(m.group(1))}]))",
    s,
)

marker = """    #[test]
    fn cancelling_one_request_does_not_cancel_peer_in_same_submission() {
"""
test = """    #[test]
    fn sampled_prefill_token_becomes_the_next_decode_input() {
        let mut serving = fixture(false);
        serving
            .admit(request(1), state(), prompt(&[11, 12]))
            .expect("request");

        serving.submit_ready_batch().expect("prefill submit");
        assert_eq!(
            serving.runtime().backend().submitted_inputs[0][0].prompt_slice(),
            Some(&[11, 12][..])
        );
        assert_eq!(serving.poll_completions().expect("pending poll"), 0);
        assert_eq!(
            serving.poll_completions().expect("prefill completion"),
            1
        );

        serving.submit_ready_batch().expect("decode submit");
        assert_eq!(
            serving.runtime().backend().submitted_inputs[1][0].decode_token(),
            Some(7)
        );
    }

"""
if marker not in s:
    raise SystemExit("test insertion marker not found")
if test not in s:
    s = s.replace(marker, test + marker, 1)

p.write_text(s)
