//! Text preprocessing above the token runtime.
//!
//! Formatting, tokenization and per-token decoding are model-specific; the
//! runtime only sees encoded tokens. Preparation therefore runs on a bounded
//! worker pool, after the caller has reserved a request permit, so tokenization
//! cannot outgrow the permit pool or occupy an async executor thread.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use engine_gguf::{ChatMessage, ChatTemplateOptions, GgufTokenizer};
use flume::{Receiver, Sender};
use ribn::GenerationOptions;
use ribn::TokenRequest;
use ribn::driver::RequestPermit;

use crate::error::TextError;
use crate::input::TextInput;

/// Separate byte bounds for text input, rendering and retained decode output.
///
/// These are text-layer limits. The token driver keeps its own encoded-input
/// envelope, and a request must satisfy both.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessorLimits {
    /// Raw prompt text, or summed message role and content bytes, per request.
    pub max_input_bytes: usize,
    /// Rendered chat-template bytes per request.
    pub max_rendered_bytes: usize,
    /// Encoded prompt tokens per request.
    pub max_prompt_tokens: usize,
    /// Decoded bytes a single token may contribute.
    pub max_decoded_token_bytes: usize,
}

impl Default for ProcessorLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024,
            max_rendered_bytes: 512 * 1024,
            max_prompt_tokens: 65_536,
            max_decoded_token_bytes: 4 * 1024,
        }
    }
}

impl ProcessorLimits {
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.max_input_bytes > 0
            && self.max_rendered_bytes > 0
            && self.max_prompt_tokens > 0
            && self.max_decoded_token_bytes > 0
    }
}

/// Model-specific text preparation and incremental decoding.
///
/// Implementations are shared across preprocessing workers and consuming
/// streams, so they must be stateless per call or internally synchronized.
pub trait TextProcessor: Send + Sync + 'static {
    #[must_use]
    fn limits(&self) -> ProcessorLimits;

    /// Format and tokenize one input. Called only after admission.
    ///
    /// # Errors
    /// Returns a request-local error for an invalid, oversized, unrenderable or
    /// untokenizable input.
    fn encode(&self, input: TextInput) -> Result<Vec<u32>, TextError>;

    /// Replace `out` with one token's decoded bytes.
    ///
    /// # Errors
    /// Returns a request-local error for an unknown token, an unsupported
    /// vocabulary symbol, or a token over the decoded byte bound.
    fn decode_token_into(&self, token: u32, out: &mut Vec<u8>) -> Result<(), TextError>;

    #[must_use]
    fn eos_token_id(&self) -> u32;
}

/// GGUF tokenizer and chat template with enforced text bounds.
pub struct GgufProcessor {
    tokenizer: GgufTokenizer,
    limits: ProcessorLimits,
}

impl GgufProcessor {
    #[must_use]
    pub const fn new(tokenizer: GgufTokenizer, limits: ProcessorLimits) -> Self {
        Self { tokenizer, limits }
    }

    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String, TextError> {
        let mut rendered = LimitedText::new(self.limits.max_rendered_bytes);
        // Preserve the existing Qwen CLI behavior: generation prompt on,
        // thinking off until reasoning controls have an explicit public API.
        let options = ChatTemplateOptions::new(true, false);
        match self
            .tokenizer
            .render_chat_to(messages, options, &mut rendered)
        {
            Ok(()) => Ok(rendered.text),
            Err(_) if rendered.exceeded.is_some() => Err(rendered.exceeded_error()),
            Err(error) => Err(TextError::Processor(error)),
        }
    }
}

impl TextProcessor for GgufProcessor {
    fn limits(&self) -> ProcessorLimits {
        self.limits
    }

    fn encode(&self, input: TextInput) -> Result<Vec<u32>, TextError> {
        let tokens = match input {
            TextInput::Tokens(tokens) => tokens,
            TextInput::Prompt(text) => {
                if text.len() > self.limits.max_input_bytes {
                    return Err(TextError::limit(
                        "prompt bytes",
                        self.limits.max_input_bytes,
                        text.len(),
                    ));
                }
                self.tokenizer.encode(&text).map_err(TextError::Processor)?
            }
            TextInput::Chat(messages) => {
                if messages.is_empty() {
                    return Err(TextError::InvalidInput(
                        "chat input must contain at least one message",
                    ));
                }
                let mut bytes = 0usize;
                for message in &messages {
                    bytes = bytes
                        .saturating_add(message.role.len())
                        .saturating_add(message.content.len());
                }
                if bytes > self.limits.max_input_bytes {
                    return Err(TextError::limit(
                        "chat message bytes",
                        self.limits.max_input_bytes,
                        bytes,
                    ));
                }
                let messages = messages
                    .into_iter()
                    .map(|message| ChatMessage::new(message.role, message.content))
                    .collect::<Vec<_>>();
                let rendered = self.render_chat(&messages)?;
                self.tokenizer
                    .encode_rendered(&rendered)
                    .map_err(TextError::Processor)?
            }
        };
        if tokens.len() > self.limits.max_prompt_tokens {
            return Err(TextError::limit(
                "prompt tokens",
                self.limits.max_prompt_tokens,
                tokens.len(),
            ));
        }
        Ok(tokens)
    }

    fn decode_token_into(&self, token: u32, out: &mut Vec<u8>) -> Result<(), TextError> {
        self.tokenizer
            .decode_token_into(token, out, self.limits.max_decoded_token_bytes)
            .map_err(|source| TextError::Decode { token, source })
    }

    fn eos_token_id(&self) -> u32 {
        self.tokenizer.eos_token_id()
    }
}

/// A writer that fails instead of retaining more than its byte bound.
struct LimitedText {
    text: String,
    limit: usize,
    /// Attempted size at the point the bound was crossed, for diagnostics.
    exceeded: Option<usize>,
}

impl LimitedText {
    const fn new(limit: usize) -> Self {
        Self {
            text: String::new(),
            limit,
            exceeded: None,
        }
    }

    fn exceeded_error(&self) -> TextError {
        TextError::limit(
            "rendered prompt bytes",
            self.limit,
            self.exceeded.unwrap_or(self.limit),
        )
    }
}

impl io::Write for LimitedText {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.text.len()) {
            self.exceeded = Some(self.text.len().saturating_add(bytes.len()));
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "rendered prompt exceeds its byte bound",
            ));
        }
        // Templates render text; a non-UTF-8 write is a template defect and must
        // not be silently lossy.
        let text = std::str::from_utf8(bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "template emitted invalid UTF-8")
        })?;
        self.text.push_str(text);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Encoded input that still owns its admission permit.
pub(crate) struct Prepared {
    pub(crate) request: TokenRequest,
    pub(crate) permit: RequestPermit,
}

struct Job {
    input: TextInput,
    options: GenerationOptions,
    permit: RequestPermit,
    reply: Sender<Result<Prepared, TextError>>,
}

struct PoolState {
    jobs: Sender<Job>,
    closed: AtomicBool,
}

/// Owner of the preprocessing workers. Workers exit once every client and this
/// owner have released the queue.
pub(crate) struct Pool {
    state: Arc<PoolState>,
}

/// Cloneable preprocessing submission handle.
#[derive(Clone)]
pub(crate) struct PoolClient {
    state: Arc<PoolState>,
}

impl Pool {
    /// Start `workers` preprocessing threads with a queue bounded by `queue`.
    pub(crate) fn start(processor: &Arc<dyn TextProcessor>, workers: usize, queue: usize) -> Self {
        let (jobs, receiver) = flume::bounded(queue);
        for index in 0..workers {
            let processor = Arc::clone(processor);
            let receiver = receiver.clone();
            // A failed thread simply reduces preprocessing capacity; the queue
            // still bounds outstanding work, so losing one worker is not fatal.
            let _ = thread::Builder::new()
                .name(format!("ribn-preprocess-{index}"))
                .spawn(move || run_worker(&*processor, &receiver));
        }
        drop(receiver);
        Self {
            state: Arc::new(PoolState {
                jobs,
                closed: AtomicBool::new(false),
            }),
        }
    }

    pub(crate) fn client(&self) -> PoolClient {
        PoolClient {
            state: self.state.clone(),
        }
    }

    /// Reject new preprocessing; in-flight jobs finish and return their permits.
    pub(crate) fn close(&self) {
        self.state.closed.store(true, Ordering::SeqCst);
    }
}

impl PoolClient {
    pub(crate) fn is_closed(&self) -> bool {
        self.state.closed.load(Ordering::SeqCst)
    }

    /// Queue preparation under an existing permit.
    ///
    /// The queue length cannot exceed the permit pool, because every queued job
    /// owns one permit; a full queue therefore means the layer is shutting down.
    pub(crate) fn submit(
        &self,
        input: TextInput,
        options: GenerationOptions,
        permit: RequestPermit,
    ) -> Result<Receiver<Result<Prepared, TextError>>, TextError> {
        if self.is_closed() {
            return Err(TextError::Closed);
        }
        let (reply, receiver) = flume::bounded(1);
        self.state
            .jobs
            .try_send(Job {
                input,
                options,
                permit,
                reply,
            })
            .map_err(|_| TextError::Closed)?;
        Ok(receiver)
    }
}

fn run_worker(processor: &dyn TextProcessor, jobs: &Receiver<Job>) {
    while let Ok(job) = jobs.recv() {
        let result = prepare(processor, job.input, job.options, job.permit);
        // A dropped caller means the result, and its permit, are released here.
        let _ = job.reply.try_send(result);
    }
}

fn prepare(
    processor: &dyn TextProcessor,
    input: TextInput,
    mut options: GenerationOptions,
    permit: RequestPermit,
) -> Result<Prepared, TextError> {
    let tokens = processor.encode(input)?;
    let eos = processor.eos_token_id();
    if !options.stop_tokens.contains(&eos) {
        options.stop_tokens.push(eos);
    }
    Ok(Prepared {
        request: TokenRequest::new(tokens, options),
        permit,
    })
}
