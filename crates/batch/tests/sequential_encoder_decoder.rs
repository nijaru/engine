use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, Engine, EngineConfig, Event, ExecutionError, ExecutorInfo, FinishReason,
    GenerationExecutor, GenerationLimits, GenerationOptions, RequestId, SchedulePolicy, SequenceId,
    StepCompletion, SubmissionId, TokenRequest,
};
use ribn_batch::{BatchConfig, BatchExecutor, BatchRuntime, Job, JobOutput};
use ribn_foundation::ParameterVersion;

type EncoderState = Arc<[u32]>;

struct Encoder;

impl BatchExecutor for Encoder {
    type Input = Vec<u32>;
    type Output = EncoderState;
    type Error = Infallible;

    fn parameter_version(&self) -> ParameterVersion {
        ParameterVersion::new(3)
    }

    fn max_batch_items(&self) -> usize {
        4
    }

    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
        Ok(batch
            .into_iter()
            .map(|job| {
                let request = job.request();
                JobOutput::new(request, Arc::<[u32]>::from(job.into_input()))
            })
            .collect())
    }
}

#[derive(Default)]
struct Shared {
    prepared: HashMap<RequestId, EncoderState>,
    admitted: HashMap<RequestId, usize>,
}

struct Decoder {
    info: ExecutorInfo,
    shared: Arc<Mutex<Shared>>,
    states: HashMap<SequenceId, EncoderState>,
    pending: Option<Vec<StepCompletion>>,
    next_submission: u64,
}

impl Decoder {
    fn new(shared: Arc<Mutex<Shared>>) -> Self {
        Self {
            info: ExecutorInfo {
                name: "conditioned-decoder-fixture".to_owned(),
                limits: GenerationLimits {
                    context_tokens: 32,
                    max_sequences: 4,
                    max_batch_tokens: 16,
                    max_decode_tokens: 1,
                },
            },
            shared,
            states: HashMap::new(),
            pending: None,
            next_submission: 0,
        }
    }
}

impl GenerationExecutor for Decoder {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        request_id: RequestId,
        sequence: SequenceId,
        _: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let state = self
            .shared
            .lock()
            .expect("shared conditioning")
            .prepared
            .remove(&request_id)
            .ok_or_else(|| ExecutionError::new("missing prepared encoder state"))?;
        let pointer = state.as_ptr() as usize;
        self.shared
            .lock()
            .expect("shared conditioning")
            .admitted
            .insert(request_id, pointer);
        self.states.insert(sequence, state);
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        if self.pending.is_some() {
            return Err(ExecutionError::new(
                "decoder fixture already has pending work",
            ));
        }
        self.next_submission += 1;
        let rows = batch
            .iter()
            .map(|item| {
                let state = self
                    .states
                    .get(&item.sequence)
                    .ok_or_else(|| ExecutionError::new("decoder state is missing"))?;
                let token = 100_u32
                    .checked_add(state.first().copied().unwrap_or_default())
                    .ok_or_else(|| ExecutionError::new("fixture token overflow"))?;
                Ok(StepCompletion {
                    sequence: item.sequence,
                    prefix: item.prefix + item.token_budget,
                    tokens: vec![token; item.output_budget as usize],
                })
            })
            .collect::<Result<Vec<_>, ExecutionError>>()?;
        self.pending = Some(rows);
        Ok(SubmissionId::new(self.next_submission))
    }

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        Ok(self.pending.take())
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.states.remove(&sequence);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.pending = None;
        Ok(())
    }
}

fn engine(shared: Arc<Mutex<Shared>>) -> Engine {
    Engine::new(
        Decoder::new(shared),
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 64,
            max_buffered_events: 16,
            max_events_per_request: 8,
        },
        SchedulePolicy {
            max_batch_tokens: 8,
            prefill_chunk_tokens: 8,
            decode_tokens: 1,
            max_decode_only_steps: 1,
        },
    )
    .expect("decoder engine")
}

fn run_until_finished(engine: &mut Engine, request: RequestId) -> (u32, FinishReason) {
    let mut token = None;
    for _ in 0..8 {
        engine.step().expect("decoder step");
        while let Some(event) = engine.pop_event_for(request) {
            match event {
                Event::Token { token: value, .. } => token = Some(value),
                Event::Finished { reason, .. } => {
                    return (token.expect("generated token"), reason);
                }
            }
        }
    }
    panic!("decoder request did not finish")
}

#[test]
fn sequential_encoder_state_is_correlated_by_request_identity_without_serialization() {
    let mut encoder = BatchRuntime::new(
        Encoder,
        BatchConfig {
            max_queued_requests: 4,
        },
    )
    .expect("encoder runtime");
    encoder.submit(vec![3_u32, 30]).expect("first encode");
    encoder.submit(vec![7_u32, 70]).expect("second encode");
    assert!(encoder.step().expect("encoder step"));
    let first_state = Arc::clone(encoder.pop_completed().expect("first state").output());
    let second_state = Arc::clone(encoder.pop_completed().expect("second state").output());

    let first_pointer = first_state.as_ptr() as usize;
    let second_pointer = second_state.as_ptr() as usize;
    let shared = Arc::new(Mutex::new(Shared::default()));
    let mut decoder = engine(Arc::clone(&shared));
    let options = GenerationOptions {
        max_output_tokens: 1,
        ..GenerationOptions::default()
    };
    let first_request = decoder
        .enqueue(TokenRequest::new([1_u32], options.clone()))
        .expect("first decode request");
    let second_request = decoder
        .enqueue(TokenRequest::new([1_u32], options))
        .expect("second decode request");

    {
        let mut state = shared.lock().expect("shared conditioning");
        state
            .prepared
            .insert(second_request, Arc::clone(&second_state));
        state
            .prepared
            .insert(first_request, Arc::clone(&first_state));
    }

    let (first_token, first_reason) = run_until_finished(&mut decoder, first_request);
    let (second_token, second_reason) = run_until_finished(&mut decoder, second_request);
    assert_eq!(first_token, 103);
    assert_eq!(second_token, 107);
    assert_eq!(first_reason, FinishReason::Length);
    assert_eq!(second_reason, FinishReason::Length);

    let state = shared.lock().expect("shared conditioning");
    assert_eq!(state.admitted[&first_request], first_pointer);
    assert_eq!(state.admitted[&second_request], second_pointer);
    assert!(state.prepared.is_empty());
}

#[test]
fn decoder_rejects_missing_prepared_encoder_state() {
    let shared = Arc::new(Mutex::new(Shared::default()));
    let mut decoder = engine(shared);
    let request = decoder
        .enqueue(TokenRequest::new(
            [1_u32],
            GenerationOptions {
                max_output_tokens: 1,
                ..GenerationOptions::default()
            },
        ))
        .expect("decode request");
    decoder.step().expect("admission failure is request-local");
    let event = decoder.pop_event_for(request).expect("terminal event");
    assert!(matches!(
        event,
        Event::Finished {
            reason: FinishReason::Failed(error),
            ..
        } if error.to_string() == "missing prepared encoder state"
    ));
}
