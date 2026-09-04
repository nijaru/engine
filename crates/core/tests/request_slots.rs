use engine_core::{
    InferenceStateSet, ModelId, RequestId, RequestLifecycle, RequestSemantics, RequestSlots,
    RequestSpec, SamplingParams, ThinkingMode,
};

fn request(id: u64) -> RequestSpec {
    let semantics = RequestSemantics::new(
        8,
        SamplingParams::greedy(Some(42)),
        ThinkingMode::Off,
    )
    .expect("request semantics");
    RequestSpec::new(
        RequestId::new(id).expect("request ID"),
        ModelId::new("test/model").expect("model ID"),
        semantics,
    )
}

#[test]
fn reused_slots_invalidate_stale_generation_ids() {
    let mut slots = RequestSlots::default();
    let first = slots
        .insert(
            request(1),
            InferenceStateSet::new(Vec::new()).expect("state"),
            4,
        )
        .expect("first slot");
    slots
        .get_mut(first)
        .expect("first request")
        .transition(RequestLifecycle::Cancelled)
        .expect("cancel");
    slots.remove(first).expect("remove first request");

    let second = slots
        .insert(
            request(2),
            InferenceStateSet::new(Vec::new()).expect("state"),
            4,
        )
        .expect("second slot");

    assert_eq!(first.index(), second.index());
    assert_ne!(first.generation(), second.generation());
    assert!(slots.get(first).is_none());
    assert_eq!(
        slots.get(second).expect("second request").request().id(),
        RequestId::new(2).expect("request ID")
    );
}
