//! Text-facade lifecycle tests that need a real device and the pinned GGUF.
//!
//! These cover the paths that only a real prepared model reaches. Facade
//! behavior that does not need a device is covered on the host in
//! `tests/facade.rs`; these tests check that the same contracts hold over the
//! CUDA-backed processor and executor.
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
//! cargo test -p ribn-text --features cuda --test text_lifecycle -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

use ribn::GenerationOptions;
use ribn_text::{FinishReason, LoadOptions, TextInput, TextModel, TextOwner, TextRequest};

const DEFAULT_MODEL: &str = "/home/nick/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf";

struct Loaded {
    owner: TextOwner,
    model: TextModel,
}

fn load() -> Loaded {
    let path = std::env::var("ENGINE_QWEN_GGUF").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
    let mut options = LoadOptions::default();
    // Small budgets keep these lifecycle tests independent of the qualification
    // runs: they exercise request ownership, not throughput.
    options.context_tokens = options.context_tokens.min(512);
    options.max_sequences = options.max_sequences.min(2);
    let (owner, _memory) = TextOwner::load(path, options).expect("load pinned Qwen GGUF");
    let model = owner.model().clone();
    Loaded { owner, model }
}

fn options() -> GenerationOptions {
    GenerationOptions {
        max_output_tokens: 8,
        ..GenerationOptions::default()
    }
}

/// Dropping an unfinished stream must not leave a mailbox that a later batch
/// either trips over or cannot get past. Before the fix this panicked inside the
/// batch path, because the engine's global drain delivered the abandoned
/// request's terminal event to the wrong consumer.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn dropping_a_stream_does_not_poison_a_later_batch() {
    let loaded = load();

    // Start streaming, consume one event, then abandon the request mid-flight.
    {
        let mut stream = loaded
            .model
            .stream_blocking(TextInput::prompt("Explain a mutex."), options())
            .expect("start stream");
        let first = stream.next().expect("one streamed event");
        assert!(first.is_ok(), "stream started: {first:?}");
        drop(stream);
    }

    let results = loaded.model.generate_batch(vec![
        TextRequest::new(TextInput::prompt("Name three colors."), options()),
        TextRequest::new(TextInput::prompt("Say hello."), options()),
    ]);
    assert_eq!(results.len(), 2);
    let responses = results
        .into_iter()
        .map(|result| result.expect("batch member settled"))
        .collect::<Vec<_>>();
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
    let response = loaded
        .model
        .generate_blocking(TextInput::prompt("Capital of France?"), options())
        .expect("single request after the batch");
    assert!(!response.tokens.is_empty());
}

/// Repeated abandonment must not permanently consume the engine's output budget,
/// which is what an undrained mailbox would do until capacity ran out.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn repeated_abandonment_stays_bounded() {
    let loaded = load();
    for attempt in 0..6 {
        let mut stream = loaded
            .model
            .stream_blocking(TextInput::prompt(format!("Attempt {attempt}")), options())
            .expect("start stream");
        let _ = stream.next();
        drop(stream);
    }

    let response = loaded
        .model
        .generate_blocking(TextInput::prompt("Still serving?"), options())
        .expect("model serves after repeated abandonment");
    assert!(
        !response.tokens.is_empty(),
        "an abandoned mailbox must not wedge the output budget"
    );
}

/// An input that cannot be prepared settles only its own item. Peers already in
/// the admission window stay owned and complete, which is why the collect helper
/// returns ordered per-item outcomes rather than one batch-wide error.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn an_unpreparable_batch_input_settles_only_its_own_item() {
    let loaded = load();
    for attempt in 0..3 {
        let results = loaded.model.generate_batch(vec![
            TextRequest::new(TextInput::prompt("first"), options()),
            // Empty chat is rejected while preparing inputs.
            TextRequest::new(TextInput::Chat(Vec::new()), options()),
            TextRequest::new(TextInput::prompt("third"), options()),
        ]);
        assert_eq!(results.len(), 3);
        assert!(
            results[0].as_ref().is_ok_and(|r| !r.tokens.is_empty()),
            "first item still generated: {:?}",
            results[0]
        );
        assert!(
            results[1].is_err(),
            "the empty chat is rejected for its own item"
        );
        assert!(
            results[2].as_ref().is_ok_and(|r| !r.tokens.is_empty()),
            "item after the rejected one still generated: {:?}",
            results[2]
        );
        eprintln!("attempt {attempt} rejected item: {:?}", results[1]);
    }

    let response = loaded
        .model
        .generate_blocking(TextInput::prompt("second attempt"), options())
        .expect("retry after rejected batches");
    assert!(!response.tokens.is_empty());
}

/// A request whose input the executor rejects fails only itself: the other batch
/// member still completes, and the model stays usable. This is the engine's
/// request-local failure behavior seen through the facade, and it is why the
/// facade must not treat a `Finished` reason as a batch-level error.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn an_invalid_token_fails_only_its_own_batch_member() {
    let loaded = load();
    let results = loaded.model.generate_batch(vec![
        TextRequest::new(TextInput::prompt("first"), options()),
        TextRequest::new(TextInput::Tokens(vec![u32::MAX]), options()),
    ]);
    assert_eq!(results.len(), 2);
    let first = results[0]
        .as_ref()
        .expect("valid member is a settled response");
    assert!(!first.tokens.is_empty(), "valid member generated");
    let second = results[1]
        .as_ref()
        .expect("an invalid token is a request-local terminal, not a text failure");
    assert!(
        matches!(second.reason, FinishReason::Failed(_)),
        "invalid member reports a local failure: {:?}",
        second.reason
    );
    assert!(second.tokens.is_empty());

    let response = loaded
        .model
        .generate_blocking(TextInput::prompt("still serving"), options())
        .expect("model usable after a per-request failure");
    assert!(!response.tokens.is_empty());
}

/// Explicit cancellation ends one request while its peers keep generating, and
/// the owner still serves afterwards.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn cancellation_is_request_local() {
    let loaded = load();
    let mut cancelled = loaded
        .model
        .stream_blocking(
            TextInput::prompt("Write a very long paragraph about rivers."),
            GenerationOptions {
                max_output_tokens: 64,
                ..GenerationOptions::default()
            },
        )
        .expect("start stream");
    let _ = cancelled.next();
    cancelled.cancel();
    let mut reason = None;
    for event in cancelled {
        if let ribn_text::TextEvent::Finished {
            reason: finished, ..
        } = event.expect("cancelled stream stays readable")
        {
            reason = Some(finished);
        }
    }
    assert_eq!(reason, Some(FinishReason::Cancelled));

    let healthy = loaded
        .model
        .generate_blocking(TextInput::prompt("Name a river."), options())
        .expect("peer request after cancellation");
    assert!(!healthy.tokens.is_empty());

    drop(loaded.owner);
}

/// Repeated identical requests must be deterministic and coherent. Greedy
/// sampling with one immutable prompt has no legitimate reason to change, so a
/// divergence or a repetition loop means shared state, cross-request delivery or
/// a stale continuation slipped through the gates that only assert arrivals.
#[test]
#[ignore = "requires the pinned Qwen GGUF and a CUDA device"]
fn repeated_requests_are_deterministic_and_coherent() {
    let loaded = load();
    let prompt = "Name one primary color.";
    let mut responses = Vec::new();
    for index in 0..4 {
        let response = loaded
            .model
            .generate_blocking(
                TextInput::prompt(prompt),
                GenerationOptions {
                    max_output_tokens: 16,
                    ..GenerationOptions::default()
                },
            )
            .expect("generate");
        eprintln!(
            "request {index}: prompt_tokens={} completion={} text={:?}",
            response.usage.prompt_tokens, response.usage.completion_tokens, response.text
        );
        responses.push(response);
    }

    for (index, response) in responses.iter().enumerate() {
        assert!(
            !response.text.trim().is_empty(),
            "request {index} produced empty text"
        );
        // Model output quality on a given prompt is not a pipeline contract, but a
        // single repeated token id means nothing was sampled at all.
        let distinct = response
            .tokens
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(
            distinct > 1,
            "request {index} sampled one repeated token: {:?}",
            response.tokens
        );
    }

    let first = responses[0].text.trim();
    for (index, response) in responses.iter().enumerate().skip(1) {
        assert_eq!(
            response.text.trim(),
            first,
            "identical request {index} diverged from request 0"
        );
    }
}
