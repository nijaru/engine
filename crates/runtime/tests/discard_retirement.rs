//! Discarded delivery must not erase the physical retirement owner.
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, Engine, Event, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, GenerationOptions, RequestId, SequenceId, StepCompletion, StepOutcome,
    SubmissionId, TokenRequest,
};

#[derive(Default)]
struct Control {
    fail_poll: bool,
    fail_release_once: bool,
    release_attempts: usize,
    synchronized: bool,
}

struct Model {
    info: ExecutorInfo,
    control: Arc<Mutex<Control>>,
    pending: Option<Vec<BatchItem>>,
}

impl GenerationExecutor for Model {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _: RequestId,
        _: SequenceId,
        _: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        Ok(Admission::Ready)
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        self.pending = Some(batch.to_vec());
        Ok(SubmissionId::new(1))
    }

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepOutcome>>, ExecutionError> {
        if self.control.lock().unwrap().fail_poll {
            return Err(ExecutionError::new("completion uncertain"));
        }
        Ok(self.pending.take().map(|batch| {
            batch
                .into_iter()
                .map(|item| {
                    StepOutcome::Progress(StepCompletion {
                        sequence: item.sequence,
                        prefix: item.prefix + item.token_budget,
                        tokens: vec![7; item.output_budget as usize],
                    })
                })
                .collect()
        }))
    }

    fn release(&mut self, _: SequenceId) -> Result<(), ExecutionError> {
        let mut control = self.control.lock().unwrap();
        control.release_attempts += 1;
        if control.fail_release_once {
            control.fail_release_once = false;
            return Err(ExecutionError::new("release must retry"));
        }
        assert!(!control.fail_poll || control.synchronized);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.pending = None;
        self.control.lock().unwrap().synchronized = true;
        Ok(())
    }
}

fn engine() -> (Engine, Arc<Mutex<Control>>) {
    let control = Arc::new(Mutex::new(Control::default()));
    let model = Model {
        info: ExecutorInfo {
            name: "discard retirement".into(),
            limits: GenerationLimits {
                context_tokens: 16,
                max_sequences: 2,
                max_batch_tokens: 8,
                max_decode_tokens: 1,
            },
        },
        control: Arc::clone(&control),
        pending: None,
    };
    (Engine::with_defaults(model).unwrap(), control)
}

fn input() -> TokenRequest {
    TokenRequest::new(
        vec![1],
        GenerationOptions {
            max_output_tokens: 1,
            ..GenerationOptions::default()
        },
    )
}

#[test]
fn discarded_mailbox_reuse_does_not_corrupt_a_failed_release_retry() {
    let (mut engine, control) = engine();
    let abandoned = engine.enqueue(input()).unwrap();
    engine.step().unwrap();
    engine.discard(abandoned);
    control.lock().unwrap().fail_release_once = true;
    assert!(engine.step().is_err());
    assert_eq!(engine.status().active_sequences, 1);
    assert_eq!(engine.status().buffered_events, 0);

    // The mailbox is reusable even though the old sequence still owns resources.
    let peer = engine.enqueue(input()).unwrap();
    engine.discard(abandoned);
    engine.step().unwrap();
    engine.step().unwrap();
    assert!(matches!(
        engine.pop_event_for(peer),
        Some(Event::Token { token: 7, .. })
    ));
    assert!(matches!(
        engine.pop_event_for(peer),
        Some(Event::Finished { .. })
    ));
    assert!(engine.pop_event().is_none());
    assert_eq!(engine.status().active_sequences, 0);
    assert_eq!(control.lock().unwrap().release_attempts, 3);
}

#[test]
fn discarded_faulted_submission_retires_only_after_shutdown_completion() {
    let (mut engine, control) = engine();
    let request = engine.enqueue(input()).unwrap();
    engine.step().unwrap();
    engine.discard(request);
    control.lock().unwrap().fail_poll = true;
    assert!(engine.step().is_err());
    engine.discard(request);
    assert_eq!(engine.status().buffered_events, 0);
    assert_eq!(engine.status().active_sequences, 1);
    assert_eq!(control.lock().unwrap().release_attempts, 0);
    engine.shutdown().unwrap();
    engine.discard(request);
    assert_eq!(engine.status().active_sequences, 0);
    assert!(engine.pop_event().is_none());
    assert_eq!(control.lock().unwrap().release_attempts, 1);
}
