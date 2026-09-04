use engine_core::{
    BackendSubmissionId, ExecutionPhase, InferenceStateSet, ModelId, RequestId, RequestLifecycle,
    RequestSemantics, RequestSlotError, RequestSlots, RequestSpec, SamplingParams, ThinkingMode,
};

fn request(id: u64) -> RequestSpec {
    let semantics = RequestSemantics::new(8, SamplingParams::greedy(Some(42)), ThinkingMode::Off)
        .expect("request semantics");
    RequestSpec::new(
        RequestId::new(id).expect("request ID"),
        ModelId::new("test/model").expect("model ID"),
        semantics,
    )
}

fn empty_state() -> InferenceStateSet {
    InferenceStateSet::new(Vec::new()).expect("state")
}

#[test]
fn reused_slots_invalidate_stale_generation_ids() {
    let mut slots = RequestSlots::default();
    let first = slots
        .insert(request(1), empty_state(), 4)
        .expect("first slot");
    slots
        .get_mut(first)
        .expect("first request")
        .request_cancel()
        .expect("cancel");
    slots.remove(first).expect("remove first request");

    let second = slots
        .insert(request(2), empty_state(), 4)
        .expect("second slot");

    assert_eq!(first.index(), second.index());
    assert_ne!(first.generation(), second.generation());
    assert!(slots.get(first).is_none());
    assert_eq!(
        slots.get(second).expect("second request").request().id(),
        RequestId::new(2).expect("request ID")
    );
}

#[test]
fn in_flight_cancellation_waits_for_backend_completion() {
    let mut slots = RequestSlots::default();
    let id = slots.insert(request(1), empty_state(), 4).expect("slot");
    let submission = BackendSubmissionId::new(1).expect("submission");

    let slot = slots.get_mut(id).expect("request");
    slot.make_runnable().expect("admit");
    let state = slot.prepare_submission().expect("prepare");
    assert_eq!(slot.lifecycle(), RequestLifecycle::Submitting);
    assert!(slot.state().is_none());
    slot.confirm_submission(submission).expect("submit");
    slot.request_cancel().expect("cancel request");
    assert_eq!(slot.lifecycle(), RequestLifecycle::Cancelling(submission));

    assert!(matches!(
        slots.remove(id),
        Err(RequestSlotError::RequestStillActive)
    ));

    let slot = slots.get_mut(id).expect("request");
    slot.complete_step(submission, ExecutionPhase::Decode, 1, state)
        .expect("complete cancelled work");
    assert_eq!(slot.lifecycle(), RequestLifecycle::Cancelled);
    assert!(slot.state().is_some());
    slots.remove(id).expect("reclaim cancelled request");
}
