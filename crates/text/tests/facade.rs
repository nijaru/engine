//! Host tests for the owned text facade.
//!
//! These drive the real `GgufProcessor`, the real token driver and the real text
//! facade over a scripted fixture executor. Only the device is substituted, so
//! preprocessing, delivery, batching and shutdown semantics under test are the
//! production ones.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use engine_gguf::{GgufTokenizer, MetadataValue, byte_token_symbol};
use ribn::driver::{Driver, DriverConfig, DriverError};
use ribn::{
    Admission, BatchItem, Engine, EngineConfig, ExecutionError, ExecutorInfo, FinishReason,
    GenerationExecutor, GenerationLimits, GenerationOptions, RequestId, SchedulePolicy, SequenceId,
    StepCompletion, StepKind, SubmissionId, TokenRequest,
};
use ribn_text::{
    GgufProcessor, Message, ProcessorLimits, TextConfig, TextError, TextInput, TextModel,
    TextOwner, TextProcessor,
};

/// The fixture vocabulary maps token id `n` to raw byte `n`, so tests can script
/// exact output bytes. Token `0` doubles as the stop token.
const EOS: u32 = 0;

fn limits() -> ProcessorLimits {
    ProcessorLimits {
        max_input_bytes: 4096,
        max_rendered_bytes: 4096,
        max_prompt_tokens: 64,
        max_decoded_token_bytes: 16,
    }
}

fn processor(limits: ProcessorLimits) -> Arc<dyn TextProcessor> {
    let tokens = (0..=255_u8)
        .map(|byte| MetadataValue::String(byte_token_symbol(byte).to_string()))
        .collect();
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_owned(),
        MetadataValue::String("gpt2".to_owned()),
    );
    metadata.insert(
        "tokenizer.ggml.pre".to_owned(),
        MetadataValue::String("qwen35".to_owned()),
    );
    metadata.insert(
        "tokenizer.ggml.tokens".to_owned(),
        MetadataValue::Array(tokens),
    );
    metadata.insert(
        "tokenizer.ggml.merges".to_owned(),
        MetadataValue::Array(Vec::new()),
    );
    metadata.insert(
        "tokenizer.ggml.token_type".to_owned(),
        MetadataValue::Array((0..256).map(|_| MetadataValue::I32(1)).collect()),
    );
    metadata.insert(
        "tokenizer.ggml.bos_token_id".to_owned(),
        MetadataValue::U32(254),
    );
    metadata.insert(
        "tokenizer.ggml.eos_token_id".to_owned(),
        MetadataValue::U32(EOS),
    );
    metadata.insert(
        "tokenizer.ggml.padding_token_id".to_owned(),
        MetadataValue::U32(253),
    );
    metadata.insert(
        "tokenizer.chat_template".to_owned(),
        MetadataValue::String("{% for m in messages %}{{ m.content }}{% endfor %}".to_owned()),
    );
    let tokenizer = GgufTokenizer::from_metadata(&metadata).expect("fixture vocabulary");
    Arc::new(GgufProcessor::new(tokenizer, limits))
}

#[derive(Default)]
struct Plan {
    /// Output tokens per first prompt token.
    scripts: Mutex<HashMap<u32, Vec<u32>>>,
    /// While set, the fixture reports pending work for decode rows only, so a
    /// request can be observed after its prefill without completing.
    hold: AtomicBool,
}

impl Plan {
    fn script(&self, prompt: u8, tokens: &[u32]) {
        self.scripts
            .lock()
            .unwrap()
            .insert(u32::from(prompt), tokens.to_vec());
    }

    fn holding(&self, hold: bool) {
        self.hold.store(hold, Ordering::SeqCst);
    }
}

struct Fixture {
    info: ExecutorInfo,
    plan: Arc<Plan>,
    states: HashMap<SequenceId, State>,
    batch: Option<Vec<BatchItem>>,
    next: u64,
}

struct State {
    script: VecDeque<u32>,
    prompt_tokens: u32,
}

impl Fixture {
    fn new(plan: Arc<Plan>) -> Self {
        Self {
            info: ExecutorInfo {
                name: "text fixture".into(),
                limits: GenerationLimits {
                    context_tokens: 4096,
                    max_sequences: 4,
                    max_batch_tokens: 8,
                    max_decode_tokens: 1,
                },
            },
            plan,
            states: HashMap::new(),
            batch: None,
            next: 0,
        }
    }
}

impl GenerationExecutor for Fixture {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _request: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let prompt = request.tokens.first().copied().unwrap_or_default();
        let script = self
            .plan
            .scripts
            .lock()
            .unwrap()
            .get(&prompt)
            .cloned()
            .unwrap_or_default();
        self.states.insert(
            sequence,
            State {
                script: script.into(),
                prompt_tokens: u32::try_from(request.tokens.len()).unwrap_or(u32::MAX),
            },
        );
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        self.batch = Some(batch.to_vec());
        self.next += 1;
        Ok(SubmissionId::new(self.next))
    }

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let holding = self.plan.hold.load(Ordering::SeqCst);
        if holding
            && self
                .batch
                .as_ref()
                .is_some_and(|batch| batch.iter().any(|item| item.kind == StepKind::Decode))
        {
            return Ok(None);
        }
        let Some(batch) = self.batch.take() else {
            return Ok(None);
        };
        let mut rows = Vec::with_capacity(batch.len());
        for item in batch {
            let state = self
                .states
                .get_mut(&item.sequence)
                .expect("admitted sequence");
            let prefix = item.prefix + item.token_budget;
            let outputs = match item.kind {
                StepKind::Prefill => u32::from(prefix == state.prompt_tokens),
                StepKind::Decode => 1,
            };
            let tokens = (0..outputs)
                .map(|_| state.script.pop_front().unwrap_or(EOS))
                .collect();
            rows.push(StepCompletion {
                sequence: item.sequence,
                prefix,
                tokens,
            });
        }
        Ok(Some(rows))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.states.remove(&sequence);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.batch = None;
        self.states.clear();
        Ok(())
    }
}

struct Harness {
    owner: TextOwner,
    model: TextModel,
    plan: Arc<Plan>,
}

impl Harness {
    fn start(limits: ProcessorLimits, config: TextConfig) -> Self {
        Self::with_events(limits, config, 32)
    }

    /// The global event budget divided by the per-request mailbox limit sets the
    /// driver's request permits, so it also bounds how many requests can be live.
    fn with_events(limits: ProcessorLimits, config: TextConfig, events: usize) -> Self {
        let plan = Arc::new(Plan::default());
        let engine = Engine::new(
            Fixture::new(plan.clone()),
            EngineConfig {
                max_active_requests: 4,
                max_queued_requests: 4,
                max_queued_input_tokens: 4096,
                max_buffered_events: events,
                max_events_per_request: 4,
            },
            SchedulePolicy {
                max_batch_tokens: 8,
                ..SchedulePolicy::default()
            },
        )
        .expect("fixture engine");
        let driver = DriverConfig::for_engine(&engine);
        let (shutdown, handle) = Driver::spawn(engine, driver).expect("driver spawn");
        let owner =
            TextOwner::new(processor(limits), shutdown, handle, config).expect("text owner");
        let model = owner.model().clone();
        Self { owner, model, plan }
    }

    fn default_start() -> Self {
        Self::start(
            limits(),
            TextConfig {
                preprocessing_workers: 2,
                batch_window: 2,
            },
        )
    }
}

fn options() -> GenerationOptions {
    GenerationOptions {
        max_output_tokens: 8,
        ..GenerationOptions::default()
    }
}

fn run<F: Future>(future: F) -> F::Output {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        let left = deadline
            .checked_duration_since(Instant::now())
            .expect("text future lost its wakeup or stalled");
        std::thread::park_timeout(left);
    }
}

fn drain(stream: &mut ribn_text::TextStream) -> (String, Vec<u32>, Option<FinishReason>) {
    let mut text = String::new();
    let mut tokens = Vec::new();
    let mut finish = None;
    while let Some(event) = stream.next_blocking() {
        match event.expect("stream event") {
            ribn_text::TextEvent::Delta { token, text: delta } => {
                if let Some(token) = token {
                    tokens.push(token);
                }
                text.push_str(&delta);
            }
            ribn_text::TextEvent::Finished { reason, .. } => finish = Some(reason),
        }
    }
    (text, tokens, finish)
}

#[test]
fn concurrent_callers_share_one_execution_owner() {
    let harness = Harness::default_start();
    harness
        .plan
        .script(b'A', &[u32::from(b'x'), u32::from(b'y'), EOS]);
    harness.plan.script(b'B', &[u32::from(b'p'), EOS]);

    let left = harness.model.clone();
    let right = harness.model.clone();
    let first = std::thread::spawn(move || {
        left.generate_blocking(TextInput::prompt("A"), options())
            .expect("first caller")
    });
    let second = std::thread::spawn(move || {
        right
            .generate_blocking(TextInput::chat(vec![Message::user("B")]), options())
            .expect("second caller")
    });
    let first = first.join().expect("caller thread");
    let second = second.join().expect("caller thread");

    assert_eq!(first.text, "xy");
    assert_eq!(first.reason, FinishReason::Stop);
    assert_eq!(second.text, "p");
    assert_eq!(second.reason, FinishReason::Stop);
}

#[test]
fn async_generation_uses_the_same_owner() {
    let harness = Harness::default_start();
    harness.plan.script(b'A', &[u32::from(b'z'), EOS]);
    let response =
        run(harness.model.generate(TextInput::prompt("A"), options())).expect("async generation");
    assert_eq!(response.text, "z");
    assert_eq!(response.tokens, vec![u32::from(b'z')]);
    // The generated stop token counts: committed output was 'z' then EOS.
    assert_eq!(response.usage.completion_tokens, 2);
}

#[test]
fn decode_failure_settles_only_its_own_request() {
    let harness = Harness::default_start();
    // Token 2000 is outside the fixture vocabulary.
    harness.plan.script(b'A', &[2000, u32::from(b'x'), EOS]);
    harness.plan.script(b'B', &[u32::from(b'k'), EOS]);

    let mut failing = harness
        .model
        .stream_blocking(TextInput::prompt("A"), options())
        .expect("start failing stream");
    let first = failing.next_blocking().expect("one event");
    assert!(matches!(first, Err(TextError::Decode { token: 2000, .. })));
    assert!(first.expect_err("decode error").is_request_local());
    assert!(failing.next_blocking().is_none(), "exactly one error");

    let healthy = harness
        .model
        .generate_blocking(TextInput::prompt("B"), options())
        .expect("peer unaffected");
    assert_eq!(healthy.text, "k");
}

#[test]
fn invalid_utf8_bytes_report_their_token() {
    let harness = Harness::default_start();
    harness.plan.script(b'A', &[0xff, u32::from(b'x'), EOS]);
    let mut stream = harness
        .model
        .stream_blocking(TextInput::prompt("A"), options())
        .expect("start stream");
    let event = stream.next_blocking().expect("one event");
    assert!(matches!(
        event,
        Err(TextError::Encoding { token: 0xff, .. })
    ));
    assert!(stream.next_blocking().is_none());
}

#[test]
fn terminal_flush_replaces_an_incomplete_code_point_once() {
    let harness = Harness::default_start();
    // A Euro sign is 0xe2 0x82 0xac; the fixture stops after two bytes.
    harness.plan.script(b'A', &[0xe2, 0x82, EOS]);
    let response = harness
        .model
        .generate_blocking(TextInput::prompt("A"), options())
        .expect("generation");
    assert_eq!(response.text, "\u{fffd}");
    assert_eq!(response.reason, FinishReason::Stop);
    assert_eq!(response.tokens, vec![0xe2, 0x82]);

    let (text, _, reason) = drain(
        &mut harness
            .model
            .stream_blocking(TextInput::prompt("A"), options())
            .expect("start stream"),
    );
    assert_eq!(reason, Some(FinishReason::Stop));
    assert_eq!(text.matches('\u{fffd}').count(), 1);
}

#[test]
fn cancellation_delivers_its_terminal_and_keeps_buffered_output() {
    let harness = Harness::default_start();
    harness.plan.script(
        b'A',
        &[u32::from(b'a'), u32::from(b'b'), u32::from(b'c'), EOS],
    );

    // Decode progress is held, so the delivered prefill delta is deterministic
    // and the request is still live when it is cancelled.
    harness.plan.holding(true);
    let mut stream = harness
        .model
        .stream_blocking(TextInput::prompt("A"), options())
        .expect("start stream");
    let first = stream.next_blocking().expect("first event").expect("delta");
    assert!(matches!(
        first,
        ribn_text::TextEvent::Delta {
            token: Some(_),
            ref text
        } if text == "a"
    ));
    stream.cancel();
    harness.plan.holding(false);

    // Already-delivered output stays readable; cancellation is a terminal, not a
    // replacement-character flush, and it does not fabricate successful usage.
    let (text, _, reason) = drain(&mut stream);
    assert_eq!(reason, Some(FinishReason::Cancelled));
    assert!(
        text.chars()
            .all(|character| matches!(character, 'a' | 'b' | 'c')),
        "buffered deltas stay readable: {text:?}"
    );

    let healthy = harness
        .model
        .generate_blocking(TextInput::prompt("A"), options())
        .expect("owner still serves after cancellation");
    assert!(!healthy.text.is_empty());
}

#[test]
fn a_stalled_consumer_does_not_block_a_peer() {
    let harness = Harness::default_start();
    harness.plan.script(b'A', &[u32::from(b'a'); 40]);
    harness.plan.script(b'B', &[u32::from(b'k'), EOS]);

    let mut stalled = harness
        .model
        .stream_blocking(
            TextInput::prompt("A"),
            GenerationOptions {
                max_output_tokens: 32,
                ..GenerationOptions::default()
            },
        )
        .expect("start stalled stream");
    let first = stalled.next_blocking().expect("first event");
    assert!(first.is_ok());

    // The stalled request's channel and mailbox fill while its peer runs.
    let peer = harness
        .model
        .generate_blocking(TextInput::prompt("B"), options())
        .expect("peer completes while a consumer stalls");
    assert_eq!(peer.text, "k");

    // The stalled request resumes once its consumer returns.
    let (text, _, reason) = drain(&mut stalled);
    assert!(
        matches!(reason, Some(FinishReason::Length | FinishReason::Stop)),
        "the stalled request still terminates: {reason:?}"
    );
    assert!(text.chars().all(|character| character == 'a'), "{text:?}");
}

#[test]
fn input_bounds_reject_before_admission_and_release_the_permit() {
    let harness = Harness::start(
        ProcessorLimits {
            max_input_bytes: 8,
            max_rendered_bytes: 8,
            max_prompt_tokens: 2,
            max_decoded_token_bytes: 4,
        },
        TextConfig {
            preprocessing_workers: 2,
            batch_window: 1,
        },
    );
    harness.plan.script(b'h', &[u32::from(b'!'), EOS]);

    let oversized = harness
        .model
        .generate_blocking(TextInput::prompt("0123456789"), options())
        .expect_err("over the input bound");
    assert!(matches!(
        oversized,
        TextError::LimitExceeded {
            field: "prompt bytes",
            allowed: 8,
            actual: 10,
        }
    ));

    let chat = harness
        .model
        .generate_blocking(
            TextInput::chat(vec![Message::user("0123456789")]),
            options(),
        )
        .expect_err("over the message bound");
    assert!(matches!(
        chat,
        TextError::LimitExceeded {
            field: "chat message bytes",
            ..
        }
    ));

    // "hi" is two tokens under the byte vocabulary but renders to one byte.
    let tokens = harness
        .model
        .generate_blocking(TextInput::prompt("hi"), options())
        .expect("within bounds");
    assert_eq!(tokens.text, "!");

    // The rejected requests refunded their permits: the same handle still serves.
    let again = harness
        .model
        .generate_blocking(TextInput::prompt("h"), options())
        .expect("still serving");
    assert_eq!(again.text, "!");
}

#[test]
fn prompt_token_bound_is_enforced_after_tokenization() {
    let harness = Harness::start(
        ProcessorLimits {
            max_input_bytes: 4096,
            max_rendered_bytes: 4096,
            max_prompt_tokens: 2,
            max_decoded_token_bytes: 4,
        },
        TextConfig::default(),
    );
    let error = harness
        .model
        .generate_blocking(TextInput::prompt("hello"), options())
        .expect_err("prompt is five tokens");
    assert!(matches!(
        error,
        TextError::LimitExceeded {
            field: "prompt tokens",
            allowed: 2,
            actual: 5,
        }
    ));
}

#[test]
fn batch_yields_ordered_results_and_settles_each_item() {
    let harness = Harness::start(
        limits(),
        TextConfig {
            preprocessing_workers: 2,
            batch_window: 3,
        },
    );
    harness.plan.script(b'A', &[u32::from(b'a'), EOS]);
    harness.plan.script(b'B', &[u32::from(b'b'), EOS]);
    harness.plan.script(b'C', &[u32::from(b'c'), EOS]);

    let results = harness.model.generate_batch(vec![
        ribn_text::TextRequest::new(TextInput::prompt("A"), options()),
        ribn_text::TextRequest::new(TextInput::chat(Vec::new()), options()),
        ribn_text::TextRequest::new(TextInput::prompt("B"), options()),
        ribn_text::TextRequest::new(TextInput::prompt("C"), options()),
    ]);
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].as_ref().expect("first").text, "a");
    assert!(matches!(
        results[1].as_ref().expect_err("empty chat"),
        TextError::InvalidInput(_)
    ));
    assert_eq!(results[2].as_ref().expect("third").text, "b");
    assert_eq!(results[3].as_ref().expect("fourth").text, "c");
}

#[test]
fn batch_window_bounds_lookahead() {
    struct Counting {
        pulled: Arc<AtomicUsize>,
        remaining: usize,
    }
    impl Iterator for Counting {
        type Item = ribn_text::TextRequest;
        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            self.pulled.fetch_add(1, Ordering::SeqCst);
            Some(ribn_text::TextRequest::new(
                TextInput::prompt("A"),
                options(),
            ))
        }
    }

    let harness = Harness::start(
        limits(),
        TextConfig {
            preprocessing_workers: 2,
            batch_window: 2,
        },
    );
    harness.plan.script(b'A', &[u32::from(b'a'), EOS]);
    let pulled = Arc::new(AtomicUsize::new(0));
    let source = Counting {
        pulled: pulled.clone(),
        remaining: 6,
    };

    let mut batch = harness.model.batch(source);
    assert_eq!(harness.model.batch_window(), 2);
    let mut yielded = 0usize;
    while let Some(result) = batch.next_result() {
        assert!(result.is_ok(), "every item settles: {result:?}");
        yielded += 1;
        let pulled = pulled.load(Ordering::SeqCst);
        assert!(
            pulled <= yielded + 2,
            "lookahead exceeded the window: pulled {pulled} for {yielded} results"
        );
        if yielded == 1 {
            assert!(
                pulled <= 3,
                "at most one refill may follow the first result: pulled {pulled}"
            );
        }
    }
    assert_eq!(yielded, 6);
}

#[test]
fn dropping_a_batch_abandons_delivery_without_stranding_the_owner() {
    let harness = Harness::start(
        limits(),
        TextConfig {
            preprocessing_workers: 2,
            batch_window: 2,
        },
    );
    harness.plan.script(b'A', &[u32::from(b'a'), EOS]);

    {
        let mut batch = harness.model.batch(vec![
            ribn_text::TextRequest::new(TextInput::prompt("A"), options()),
            ribn_text::TextRequest::new(TextInput::prompt("A"), options()),
        ]);
        let first = batch.next_result().expect("first result");
        assert!(first.is_ok());
        // The second stream is abandoned by dropping the batch.
    }

    let response = harness
        .model
        .generate_blocking(TextInput::prompt("A"), options())
        .expect("owner still serves after abandonment");
    assert_eq!(response.text, "a");
}

#[test]
fn owner_shutdown_closes_admission_and_reports_success() {
    let mut harness = Harness::default_start();
    harness.plan.script(b'A', &[u32::from(b'a'), EOS]);
    harness.owner.shutdown().expect("clean shutdown");

    let error = harness
        .model
        .generate_blocking(TextInput::prompt("A"), options())
        .expect_err("admission is closed");
    assert!(matches!(error, TextError::Admission(_) | TextError::Closed));
}

#[test]
fn invalid_assembly_settings_are_rejected() {
    let plan = Arc::new(Plan::default());
    let engine = Engine::new(
        Fixture::new(plan),
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 256,
            max_buffered_events: 8,
            max_events_per_request: 4,
        },
        SchedulePolicy {
            max_batch_tokens: 8,
            ..SchedulePolicy::default()
        },
    )
    .expect("fixture engine");
    let driver = DriverConfig::for_engine(&engine);
    let (shutdown, handle) = Driver::spawn(engine, driver).expect("driver spawn");
    let Err(error) = TextOwner::new(
        processor(limits()),
        shutdown,
        handle,
        TextConfig {
            preprocessing_workers: 0,
            batch_window: 2,
        },
    ) else {
        panic!("zero preprocessing workers must be rejected");
    };
    assert!(matches!(error, TextError::InvalidInput(_)));
}

#[test]
fn a_decode_failure_inside_a_batch_settles_only_that_item() {
    let harness = Harness::default_start();
    harness.plan.script(b'A', &[2000, u32::from(b'x'), EOS]);
    harness.plan.script(b'B', &[u32::from(b'k'), EOS]);

    let results = harness.model.generate_batch(vec![
        ribn_text::TextRequest::new(TextInput::prompt("A"), options()),
        ribn_text::TextRequest::new(TextInput::prompt("B"), options()),
    ]);
    assert_eq!(results.len(), 2);
    assert!(matches!(
        results[0].as_ref().expect_err("out-of-vocabulary token"),
        TextError::Decode { token: 2000, .. }
    ));
    assert_eq!(
        results[1]
            .as_ref()
            .expect("the peer item is unaffected")
            .text,
        "k"
    );
}

#[test]
fn shared_handle_overload_is_reported_per_item() {
    // Eight global events over a four-event mailbox leaves two request permits.
    let harness = Harness::with_events(
        limits(),
        TextConfig {
            preprocessing_workers: 2,
            batch_window: 2,
        },
        8,
    );
    harness.plan.script(b'A', &[u32::from(b'a'); 40]);
    harness.plan.script(b'B', &[u32::from(b'k'), EOS]);

    // Two stalled consumers retain both permits: their output is never read, so
    // neither request can reach a terminal.
    let held = [
        harness
            .model
            .stream_blocking(
                TextInput::prompt("A"),
                GenerationOptions {
                    max_output_tokens: 32,
                    ..GenerationOptions::default()
                },
            )
            .expect("first held stream"),
        harness
            .model
            .stream_blocking(
                TextInput::prompt("A"),
                GenerationOptions {
                    max_output_tokens: 32,
                    ..GenerationOptions::default()
                },
            )
            .expect("second held stream"),
    ];

    // A third request reports overload for itself, not as an owner failure.
    let error = harness
        .model
        .generate_blocking(TextInput::prompt("B"), options())
        .expect_err("all permits are retained");
    assert!(matches!(
        error,
        TextError::Admission(DriverError::Overloaded)
    ));
    assert!(
        !error.is_request_local(),
        "overload is a capacity condition, not invalid input"
    );

    // The same holds per batch item rather than failing a whole batch.
    let results = harness.model.generate_batch(vec![
        ribn_text::TextRequest::new(TextInput::prompt("B"), options()),
        ribn_text::TextRequest::new(TextInput::prompt("B"), options()),
    ]);
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Err(TextError::Admission(DriverError::Overloaded))))
    );

    // Releasing the stalled consumers returns the permits.
    drop(held);
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match harness
            .model
            .generate_blocking(TextInput::prompt("B"), options())
        {
            Ok(response) => {
                assert_eq!(response.text, "k");
                break;
            }
            Err(TextError::Admission(DriverError::Overloaded)) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(error) => panic!("model stopped serving after overload: {error}"),
        }
    }
}
