use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use engine_gguf::{ChatMessage, ChatTemplateOptions, GgufFile, GgufTokenizer, MetadataValue};
use engine_qwen::{QwenCuda, QwenLoadOptions};
use ribn::{Engine, EngineError, Event, GenerationOptions, RequestId, TokenRequest, Usage};

use crate::{FinishReason, Message, TextInput};

/// Model loading and execution limits for the currently supported text path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadOptions {
    pub device: u16,
    pub context_tokens: u32,
    pub max_sequences: usize,
    /// Same-sequence prefill chunk size, or `None` for the serial prefill
    /// path.
    pub prefill_chunk_members: Option<usize>,
    pub weight_budget_bytes: Option<u64>,
    pub headroom_bytes: u64,
}

impl Default for LoadOptions {
    fn default() -> Self {
        let qwen = QwenLoadOptions::default();
        Self {
            device: qwen.device,
            context_tokens: qwen.context_tokens,
            max_sequences: qwen.max_sequences,
            prefill_chunk_members: qwen.prefill_chunk_members,
            weight_budget_bytes: qwen.weight_budget_bytes,
            headroom_bytes: qwen.headroom_bytes,
        }
    }
}

impl From<LoadOptions> for QwenLoadOptions {
    fn from(value: LoadOptions) -> Self {
        Self {
            device: value.device,
            context_tokens: value.context_tokens,
            max_sequences: value.max_sequences,
            prefill_chunk_members: value.prefill_chunk_members,
            weight_budget_bytes: value.weight_budget_bytes,
            headroom_bytes: value.headroom_bytes,
        }
    }
}

pub use engine_qwen::MemoryReport;

#[derive(Clone, Debug, PartialEq)]
pub struct TextRequest {
    pub input: TextInput,
    pub options: GenerationOptions,
}

impl TextRequest {
    #[must_use]
    pub fn new(input: TextInput, options: GenerationOptions) -> Self {
        Self { input, options }
    }
}

/// Loaded text model with reusable tokenizer and generation runtime.
pub struct TextModel {
    tokenizer: GgufTokenizer,
    engine: Engine,
    memory: MemoryReport,
}

impl TextModel {
    /// Load a supported local GGUF model and prepare its CUDA executor.
    ///
    /// # Errors
    /// Returns an explicit unsupported-architecture error instead of inferring
    /// execution support from the existence of GGUF metadata.
    pub fn load(path: impl Into<PathBuf>, options: LoadOptions) -> Result<Self, TextError> {
        let path = path.into();
        let file = GgufFile::open(&path).map_err(TextError::from_display)?;
        let architecture = file
            .metadata("general.architecture")
            .and_then(MetadataValue::as_str)
            .ok_or_else(|| TextError::new("model artifact is missing general.architecture"))?;
        if architecture != "qwen35" {
            return Err(TextError::new(format!(
                "unsupported model architecture {architecture:?}; this build currently executes qwen35 GGUF only"
            )));
        }
        let tokenizer = file.tokenizer().map_err(TextError::from_display)?;
        drop(file);
        let executor =
            QwenCuda::load_gguf(path, options.into()).map_err(TextError::from_display)?;
        let memory = executor.memory_report();
        let engine = Engine::with_defaults(executor).map_err(TextError::from_display)?;
        Ok(Self {
            tokenizer,
            engine,
            memory,
        })
    }

    #[must_use]
    pub const fn memory_report(&self) -> MemoryReport {
        self.memory
    }

    /// Apply Ribn's shared text-input preparation without executing the model.
    ///
    /// # Errors
    /// Returns chat-template or tokenization errors.
    pub fn tokenize(&self, input: TextInput) -> Result<Vec<u32>, TextError> {
        encode_input(&self.tokenizer, input)
    }

    /// Decode token IDs with the loaded model's tokenizer.
    ///
    /// # Errors
    /// Returns an error for invalid token IDs or invalid decoded UTF-8.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, TextError> {
        self.tokenizer
            .decode(tokens)
            .map_err(TextError::from_display)
    }

    /// Start a streaming generation request.
    ///
    /// # Errors
    /// Returns formatting, tokenization, admission, or execution errors.
    pub fn stream(
        &mut self,
        input: TextInput,
        mut options: GenerationOptions,
    ) -> Result<TextStream<'_>, TextError> {
        let tokens = encode_input(&self.tokenizer, input)?;
        if !options.stop_tokens.contains(&self.tokenizer.eos_token_id()) {
            options.stop_tokens.push(self.tokenizer.eos_token_id());
        }
        let request = self
            .engine
            .enqueue(TokenRequest::new(tokens, options))
            .map_err(TextError::from_display)?;
        Ok(TextStream {
            model: self,
            request,
            decoder: Utf8Decoder::default(),
            pending_finish: None,
            finished: false,
        })
    }

    /// Generate one raw-text completion to completion.
    ///
    /// # Errors
    /// Returns formatting, tokenization, admission, decoding, or execution errors.
    pub fn generate(
        &mut self,
        prompt: impl Into<String>,
        options: GenerationOptions,
    ) -> Result<TextResponse, TextError> {
        self.complete(TextInput::Prompt(prompt.into()), options)
    }

    /// Generate from structured chat messages using the artifact's chat template.
    ///
    /// # Errors
    /// Returns formatting, tokenization, admission, decoding, or execution errors.
    pub fn chat(
        &mut self,
        messages: impl Into<Vec<Message>>,
        options: GenerationOptions,
    ) -> Result<TextResponse, TextError> {
        self.complete(TextInput::Chat(messages.into()), options)
    }

    /// Generate from already-tokenized input without text preprocessing.
    ///
    /// # Errors
    /// Returns admission, decoding, or execution errors.
    pub fn generate_tokens(
        &mut self,
        tokens: impl Into<Vec<u32>>,
        options: GenerationOptions,
    ) -> Result<TextResponse, TextError> {
        self.complete(TextInput::Tokens(tokens.into()), options)
    }

    /// Submit several requests before driving execution so the runtime can
    /// batch compatible work. Results preserve input order.
    ///
    /// # Errors
    /// Returns formatting, tokenization, admission, decoding, or execution errors.
    pub fn generate_batch(
        &mut self,
        requests: impl IntoIterator<Item = TextRequest>,
    ) -> Result<Vec<TextResponse>, TextError> {
        let requests = requests.into_iter().collect::<Vec<_>>();
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let mut responses = Vec::with_capacity(requests.len());
        let mut positions = HashMap::with_capacity(requests.len());
        for request in requests {
            let tokens = encode_input(&self.tokenizer, request.input)?;
            let mut options = request.options;
            if !options.stop_tokens.contains(&self.tokenizer.eos_token_id()) {
                options.stop_tokens.push(self.tokenizer.eos_token_id());
            }
            let id = match self.engine.enqueue(TokenRequest::new(tokens, options)) {
                Ok(id) => id,
                Err(error) => {
                    self.cancel_batch(positions.keys().copied());
                    return Err(TextError::from_display(error));
                }
            };
            positions.insert(id, responses.len());
            responses.push(BatchResponse::default());
        }

        let mut remaining = responses.len();
        while remaining > 0 {
            let status = self.engine.step().map_err(TextError::from_display)?;
            while let Some(event) = self.engine.pop_event() {
                let index = positions[&event.request()];
                let response = &mut responses[index];
                match event {
                    Event::Token { token, .. } => {
                        let bytes = self
                            .tokenizer
                            .decode_bytes(&[token])
                            .map_err(TextError::from_display)?;
                        response.text.push_str(&response.decoder.push(&bytes)?);
                        response.tokens.push(token);
                    }
                    Event::Finished { reason, usage, .. } => {
                        response.text.push_str(&response.decoder.finish());
                        response.finish = Some((reason, usage));
                        remaining -= 1;
                    }
                }
            }
            if !status.submitted && !status.completed && remaining > 0 {
                std::thread::yield_now();
            }
        }

        responses
            .into_iter()
            .map(BatchResponse::finish)
            .collect::<Result<Vec<_>, _>>()
    }

    fn cancel_batch(&mut self, requests: impl IntoIterator<Item = RequestId>) {
        let requests = requests.into_iter().collect::<Vec<_>>();
        for request in &requests {
            let _ = self.engine.cancel(*request);
        }
        // This path runs only before batch execution starts, so cancellation is
        // immediate. Drive terminal bookkeeping and discard cancellation events.
        let _ = self.engine.step();
        for request in requests {
            while self.engine.pop_event_for(request).is_some() {}
        }
    }

    fn complete(
        &mut self,
        input: TextInput,
        options: GenerationOptions,
    ) -> Result<TextResponse, TextError> {
        let mut text = String::new();
        let mut tokens = Vec::new();
        let mut finish = None;
        let stream = self.stream(input, options)?;
        for event in stream {
            match event? {
                TextEvent::Delta { token, text: delta } => {
                    if let Some(token) = token {
                        tokens.push(token);
                    }
                    text.push_str(&delta);
                }
                TextEvent::Finished { reason, usage } => finish = Some((reason, usage)),
            }
        }
        let (reason, usage) =
            finish.ok_or_else(|| TextError::new("generation ended without a terminal event"))?;
        Ok(TextResponse {
            text,
            tokens,
            reason,
            usage,
        })
    }
}

impl TextModel {
    /// Establish device completion and release all active sequence resources.
    ///
    /// # Errors
    /// Reports synchronization or resource-release failure.
    pub fn shutdown(&mut self) -> Result<(), TextError> {
        self.engine.shutdown().map_err(TextError::from_display)
    }
}

/// Incremental text event. Token IDs remain available even when one token does
/// not independently form valid UTF-8 and therefore has an empty text delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextEvent {
    /// Incremental decoded text. `token` is absent only for a final replacement
    /// character when generation ends inside a UTF-8 code point.
    Delta {
        token: Option<u32>,
        text: String,
    },
    Finished {
        reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextResponse {
    pub text: String,
    pub tokens: Vec<u32>,
    pub reason: FinishReason,
    pub usage: Usage,
}

/// Synchronous streaming request over one loaded model.
/// Dropping an unfinished stream records cancellation; in-flight device work
/// remains owned until the runtime observes completion.
pub struct TextStream<'a> {
    model: &'a mut TextModel,
    request: RequestId,
    decoder: Utf8Decoder,
    pending_finish: Option<(FinishReason, Usage)>,
    finished: bool,
}

impl Iterator for TextStream<'_> {
    type Item = Result<TextEvent, TextError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if let Some((reason, usage)) = self.pending_finish.take() {
            self.finished = true;
            return Some(Ok(TextEvent::Finished { reason, usage }));
        }
        loop {
            if let Some(event) = self.model.engine.pop_event_for(self.request) {
                match event {
                    Event::Token { token, .. } => {
                        let bytes = match self.model.tokenizer.decode_bytes(&[token]) {
                            Ok(bytes) => bytes,
                            Err(error) => return Some(Err(TextError::from_display(error))),
                        };
                        let text = match self.decoder.push(&bytes) {
                            Ok(text) => text,
                            Err(error) => return Some(Err(error)),
                        };
                        return Some(Ok(TextEvent::Delta {
                            token: Some(token),
                            text,
                        }));
                    }
                    Event::Finished { reason, usage, .. } => {
                        let trailing = self.decoder.finish();
                        if trailing.is_empty() {
                            self.finished = true;
                            return Some(Ok(TextEvent::Finished { reason, usage }));
                        }
                        self.pending_finish = Some((reason, usage));
                        return Some(Ok(TextEvent::Delta {
                            token: None,
                            text: trailing,
                        }));
                    }
                }
            }
            match self.model.engine.step() {
                Ok(status) => {
                    if !status.submitted && !status.completed {
                        std::thread::yield_now();
                    }
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(TextError::from_display(error)));
                }
            }
        }
    }
}

impl Drop for TextStream<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.model.engine.cancel(self.request);
        }
    }
}

fn encode_input(tokenizer: &GgufTokenizer, input: TextInput) -> Result<Vec<u32>, TextError> {
    match input {
        TextInput::Prompt(text) => tokenizer.encode(&text).map_err(TextError::from_display),
        TextInput::Tokens(tokens) => Ok(tokens),
        TextInput::Chat(messages) => {
            if messages.is_empty() {
                return Err(TextError::new(
                    "chat input must contain at least one message",
                ));
            }
            let messages = messages
                .into_iter()
                .map(|message| ChatMessage::new(message.role, message.content))
                .collect::<Vec<_>>();
            // Preserve the existing Qwen CLI behavior: generation prompt on,
            // thinking off until reasoning controls have an explicit public API.
            tokenizer
                .encode_chat(&messages, ChatTemplateOptions::new(true, false))
                .map_err(TextError::from_display)
        }
    }
}

#[derive(Default)]
struct BatchResponse {
    decoder: Utf8Decoder,
    text: String,
    tokens: Vec<u32>,
    finish: Option<(FinishReason, Usage)>,
}

impl BatchResponse {
    fn finish(self) -> Result<TextResponse, TextError> {
        let (reason, usage) = self
            .finish
            .ok_or_else(|| TextError::new("generation ended without a terminal event"))?;
        Ok(TextResponse {
            text: self.text,
            tokens: self.tokens,
            reason,
            usage,
        })
    }
}

#[derive(Default)]
struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    fn push(&mut self, bytes: &[u8]) -> Result<String, TextError> {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let text = text.to_owned();
                self.pending.clear();
                Ok(text)
            }
            Err(error) if error.error_len().is_none() => {
                let valid = error.valid_up_to();
                let prefix = String::from_utf8(self.pending[..valid].to_vec())
                    .expect("validated UTF-8 prefix");
                self.pending.drain(..valid);
                Ok(prefix)
            }
            Err(error) => Err(TextError::new(format!(
                "generated bytes are not valid UTF-8: {error}"
            ))),
        }
    }

    fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            String::new()
        } else {
            let trailing = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            trailing
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextError(String);

impl TextError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn from_display(error: impl fmt::Display) -> Self {
        Self::new(error.to_string())
    }
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for TextError {}

impl From<EngineError> for TextError {
    fn from(error: EngineError) -> Self {
        Self::from_display(error)
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8Decoder;

    #[test]
    fn incremental_decoder_retains_split_codepoints() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.push(&[0xe2]).unwrap(), "");
        assert_eq!(decoder.push(&[0x82]).unwrap(), "");
        assert_eq!(decoder.push(&[0xac]).unwrap(), "€");
    }

    #[test]
    fn terminal_incomplete_codepoint_is_replaced_once() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.push(&[0xe2]).unwrap(), "");
        assert_eq!(decoder.finish(), "�");
    }
}
