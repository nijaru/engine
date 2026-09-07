use engine_core::{
    BackendSubmissionId, ExecutionBatchEvent, ExecutionEvent, ExecutionMetrics, ExecutionPhase,
    InferenceStateSet, ModelId, PolicySnapshot, PolicyVersion, RequestId, RequestSemantics,
    RequestSpec, SamplingParams, SchedulerConfig, ServingScheduler, SpeculationPolicy,
    StateTierPreference, ThinkingMode,
};

fn state() -> InferenceStateSet {
    InferenceStateSet::new(Vec::new()).expect("empty state")
}

fn request() -> RequestSpec {
    RequestSpec::new(
        RequestId::new(1).expect("request ID"),
        ModelId::new("test/model").expect("model ID"),
        RequestSemantics::new(2, SamplingParams::greedy(None), ThinkingMode::Off)
            .expect("request semantics"),
    )
}

fn completion(work: engine_core::ScheduledWork, output_token: Option<u32>) -> ExecutionBatchEvent {
    let event = ExecutionEvent::new(
        work.request(),
        PolicyVersion::new(1).expect("policy version"),
        work.phase(),
        work.token_count(),
        ExecutionMetrics::new(1, 0, 0),
    )
    .expect("execution event");
    ExecutionBatchEvent::new(vec![match output_token {
        Some(token) => event.with_output_token(token),
        None => event,
    }])
    .expect("batch event")
}

#[test]
fn only_the_final_prefill_chunk_requests_an_output_token() {
    let version = PolicyVersion::new(1).expect("policy version");
    let policy = PolicySnapshot::new(
        version,
        1,
        3,
        StateTierPreference::Automatic,
        SpeculationPolicy::Disabled,
    )
    .expect("policy");
    let config = SchedulerConfig::new(1, 0, 3).expect("scheduler config");
    let mut scheduler = ServingScheduler::new(policy, config);
    scheduler
        .admit(request(), state(), 5)
        .expect("admit request");

    let first = scheduler.schedule().expect("first schedule");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].phase(), ExecutionPhase::Prefill);
    assert_eq!(first[0].token_count(), 3);
    assert!(!first[0].requests_output());
    let mut first_states = scheduler
        .prepare_submission(&first)
        .expect("prepare first chunk");
    let first_submission = BackendSubmissionId::new(1).expect("submission ID");
    scheduler
        .confirm_submission(first.clone(), first_submission)
        .expect("confirm first chunk");
    scheduler
        .complete_submission(
            first_submission,
            &completion(first[0], None),
            &mut first_states,
        )
        .expect("complete first chunk");

    let second = scheduler.schedule().expect("second schedule");
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].phase(), ExecutionPhase::Prefill);
    assert_eq!(second[0].token_count(), 2);
    assert_eq!(second[0].state_position(), 3);
    assert!(second[0].requests_output());
    let mut second_states = scheduler
        .prepare_submission(&second)
        .expect("prepare final chunk");
    let second_submission = BackendSubmissionId::new(2).expect("submission ID");
    scheduler
        .confirm_submission(second.clone(), second_submission)
        .expect("confirm final chunk");
    scheduler
        .complete_submission(
            second_submission,
            &completion(second[0], Some(7)),
            &mut second_states,
        )
        .expect("complete final chunk");

    let slot_id = scheduler
        .slot_for_request(RequestId::new(1).expect("request ID"))
        .expect("request slot");
    let slot = scheduler.slots().get(slot_id).expect("request slot");
    assert_eq!(slot.progress().prompt_processed(), 5);
    assert_eq!(slot.progress().generated_tokens(), 1);
    assert_eq!(slot.progress().decode_processed(), 0);

    let decode = scheduler.schedule().expect("decode schedule");
    assert_eq!(decode.len(), 1);
    assert_eq!(decode[0].phase(), ExecutionPhase::Decode);
    assert_eq!(decode[0].state_position(), 5);
    assert!(decode[0].requests_output());
}
