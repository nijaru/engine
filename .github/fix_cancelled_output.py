from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


p = Path("crates/core/src/serving_runtime.rs")
s = p.read_text()
old = '''                    for completed in event.events() {
                        let request = completed.request();
                        if let Some(token) = completed.output_token() {
                            self.next_tokens.insert(request, token);
                            self.generated_tokens
                                .push_back(GeneratedToken::new(request, token));
                        }
                        if completed.phase() == crate::execution::ExecutionPhase::Prefill {
'''
new = '''                    for completed in event.events() {
                        let request = completed.request();
                        let lifecycle = self
                            .scheduler
                            .slot_for_request(request)
                            .and_then(|slot| self.scheduler.slots().get(slot))
                            .ok_or(ServingRuntimeError::RequestMissing(request))?
                            .lifecycle();
                        if let Some(token) = completed.output_token() {
                            match lifecycle {
                                crate::serving::RequestLifecycle::Runnable => {
                                    self.next_tokens.insert(request, token);
                                    self.generated_tokens
                                        .push_back(GeneratedToken::new(request, token));
                                }
                                crate::serving::RequestLifecycle::Completed => {
                                    self.generated_tokens
                                        .push_back(GeneratedToken::new(request, token));
                                }
                                crate::serving::RequestLifecycle::Cancelled
                                | crate::serving::RequestLifecycle::Failed => {}
                                crate::serving::RequestLifecycle::Waiting
                                | crate::serving::RequestLifecycle::Submitting
                                | crate::serving::RequestLifecycle::InFlight(_)
                                | crate::serving::RequestLifecycle::Cancelling(_) => {
                                    return Err(ServingRuntimeError::CompletionLifecycle(request));
                                }
                            }
                        }
                        if completed.phase() == crate::execution::ExecutionPhase::Prefill {
'''
s = replace_once(s, old, new, "completion output lifecycle filtering")
old = '''    PromptMissing(RequestId),
    DecodeTokenMissing(RequestId),
    SubmissionCollision(BackendSubmissionId),
'''
new = '''    PromptMissing(RequestId),
    DecodeTokenMissing(RequestId),
    RequestMissing(RequestId),
    CompletionLifecycle(RequestId),
    SubmissionCollision(BackendSubmissionId),
'''
s = replace_once(s, old, new, "serving error variants")
old = '''            Self::DecodeTokenMissing(request) => {
                write!(
                    f,
                    "next decode token for request {} is unavailable",
                    request.get()
                )
            }
            Self::SubmissionCollision(id) => {
'''
new = '''            Self::DecodeTokenMissing(request) => {
                write!(
                    f,
                    "next decode token for request {} is unavailable",
                    request.get()
                )
            }
            Self::RequestMissing(request) => {
                write!(f, "request {} disappeared during completion", request.get())
            }
            Self::CompletionLifecycle(request) => write!(
                f,
                "request {} remained in a transient lifecycle after completion",
                request.get()
            ),
            Self::SubmissionCollision(id) => {
'''
s = replace_once(s, old, new, "serving error display")
old = '''        serving.poll_completions().expect("pending poll");
        serving.poll_completions().expect("completion poll");
        assert_eq!(serving.scheduler().counts().terminal(), 1);
        assert_eq!(serving.scheduler().counts().runnable(), 1);
    }
'''
new = '''        serving.poll_completions().expect("pending poll");
        serving.poll_completions().expect("completion poll");
        assert_eq!(serving.scheduler().counts().terminal(), 1);
        assert_eq!(serving.scheduler().counts().runnable(), 1);
        assert_eq!(serving.generated_token_count(), 1);
        assert_eq!(
            serving.pop_generated_token(),
            Some(GeneratedToken::new(
                RequestId::new(2).expect("request ID"),
                7
            ))
        );
        assert_eq!(serving.generated_token_count(), 0);
    }
'''
s = replace_once(s, old, new, "cancelled output suppression test")
p.write_text(s)
