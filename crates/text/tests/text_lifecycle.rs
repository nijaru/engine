//! Text-facade lifecycle tests that need a real device and the pinned GGUF.
//!
//! These cover the paths that no host-only test can reach, because `TextModel`
//! loads a Qwen CUDA executor. The host-side contract they depend on lives in
//! `crates/runtime/tests/abandoned_requests.rs`; this file checks the facade built
//! on top of it.
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! cargo test -p ribn-text --features cuda --test text_lifecycle -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use ribn::GenerationOptions;
use ribn_text::{FinishReason, LoadOptions, TextInput, TextModel, TextRequest};

const DEFAULT_MODEL: &str = "/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf";

fn load() -> TextModel {
    let path = std::env::var("ENGINE_QWEN_GGUF").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
    let mut options = LoadOptions::default();
    // Small budgets keep these lifecycle tests independent of the qualification
    // runs: they exercise request ownership, not throughput.
    options.context_tokens = options.context_tokens.min(512);
    options.max_sequences = options.max_sequences.min(2);
    TextModel::load(path, options).expect("load pinned Qwen GGUF")
}

fn options() -> GenerationOptions {
    GenerationOptions {
        max_output_tokens: 8,
        ..GenerationOptions::default()
    }
}

/// Dropping an unfinished stream must not leave a mailbox that a later batch
/// either trips over or cannot get past. Before the fix this panicked inside
/// `generate_batch`, because the engine's global drain delivered the abandoned
/// request's terminal event and the batch indexed its own request map with it.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn dropping_a_stream_does_not_poison_a_later_batch() {
    let mut model = load();

    // Start streaming, consume one event, then abandon the request mid-flight.
    {
        let mut stream = model
            .stream(TextInput::Prompt("Explain a mutex.".to_owned()), options())
            .expect("start stream");
        let first = stream.next().expect("one streamed event");
        assert!(first.is_ok(), "stream started: {first:?}");
        drop(stream);
    }

    let responses = model
        .generate_batch(vec![
            TextRequest::new(
                TextInput::Prompt("Name three colors.".to_owned()),
                options(),
            ),
            TextRequest::new(TextInput::Prompt("Say hello.".to_owned()), options()),
        ])
        .expect("batch after an abandoned stream");
    assert_eq!(responses.len(), 2);
    for response in &responses {
        assert!(
            !response.tokens.is_empty(),
            "every batch member generated: {response:?}"
        );
    }
    assert_ne!(
        responses[0].text, responses[1].text,
        "results stay correlated with their own request"
    );

    // And the model still serves a plain request afterwards.
    let response = model
        .generate("Capital of France?", options())
        .expect("single request after the batch");
    assert!(!response.tokens.is_empty());
}

/// Repeated abandonment must not permanently consume the engine's output budget,
/// which is what an undrained mailbox would do until capacity ran out.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn repeated_abandonment_stays_bounded() {
    let mut model = load();
    for attempt in 0..6 {
        let mut stream = model
            .stream(TextInput::Prompt(format!("Attempt {attempt}")), options())
            .expect("start stream");
        let _ = stream.next();
        drop(stream);
    }

    let response = model
        .generate("Still serving?", options())
        .expect("model serves after repeated abandonment");
    assert!(
        !response.tokens.is_empty(),
        "an abandoned mailbox must not wedge the output budget"
    );
}

/// An input that fails to prepare must fail the batch *before* anything is
/// submitted, so no member is left running with no owner and the model stays
/// usable.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn an_unpreparable_batch_input_fails_before_submitting_anything() {
    let mut model = load();
    for attempt in 0..3 {
        let error = model
            .generate_batch(vec![
                TextRequest::new(TextInput::Prompt("first".to_owned()), options()),
                // Empty chat is rejected while preparing inputs, before enqueue.
                TextRequest::new(TextInput::Chat(Vec::new()), options()),
            ])
            .expect_err("an empty chat input must fail the batch");
        eprintln!("attempt {attempt} rejected with: {error}");
    }

    let responses = model
        .generate_batch(vec![TextRequest::new(
            TextInput::Prompt("second attempt".to_owned()),
            options(),
        )])
        .expect("retry after rejected batches");
    assert_eq!(responses.len(), 1);
    assert!(!responses[0].tokens.is_empty());
}

/// A request whose input the executor rejects fails only itself: the other batch
/// member still completes, and the model stays usable. This is the engine's
/// request-local failure behavior seen through the facade, and it is why the
/// facade must not treat a `Finished` reason as a batch-level error.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn an_invalid_token_fails_only_its_own_batch_member() {
    let mut model = load();
    let responses = model
        .generate_batch(vec![
            TextRequest::new(TextInput::Prompt("first".to_owned()), options()),
            TextRequest::new(TextInput::Tokens(vec![u32::MAX]), options()),
        ])
        .expect("an invalid token is rejected per request, not per batch");
    assert_eq!(responses.len(), 2);
    assert!(!responses[0].tokens.is_empty(), "valid member generated");
    assert!(
        matches!(responses[1].reason, FinishReason::Failed(_)),
        "invalid member reports a local failure: {:?}",
        responses[1].reason
    );
    assert!(responses[1].tokens.is_empty());

    let response = model
        .generate("still serving", options())
        .expect("model usable after a per-request failure");
    assert!(!response.tokens.is_empty());
}
