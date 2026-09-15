//! Owned text delivery: incremental UTF-8 decoding over one token stream.

use std::sync::Arc;

use ribn::driver::GenerationStream;
use ribn::{Event, FinishReason, RequestId, Usage};

use crate::error::TextError;
use crate::process::TextProcessor;

/// Incremental text event. Token IDs remain available even when one token does
/// not independently form valid UTF-8 and therefore has an empty text delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextEvent {
    /// Incremental decoded text. `token` is absent only for a final replacement
    /// character when a normal or cancelled generation ends inside a code point.
    Delta {
        token: Option<u32>,
        text: String,
    },
    Finished {
        reason: FinishReason,
        usage: Usage,
    },
}

/// One owned text request. Dropping an unfinished stream abandons delivery
/// without waiting for device completion; the execution owner retains retirement.
pub struct TextStream {
    processor: Arc<dyn TextProcessor>,
    request: RequestId,
    stream: Option<GenerationStream>,
    decoder: Utf8Decoder,
    bytes: Vec<u8>,
    pending: Option<(FinishReason, Usage)>,
    done: bool,
}

impl TextStream {
    pub(crate) fn new(processor: Arc<dyn TextProcessor>, stream: GenerationStream) -> Self {
        let request = stream.request_id();
        Self {
            processor,
            request,
            stream: Some(stream),
            decoder: Utf8Decoder::default(),
            bytes: Vec::new(),
            pending: None,
            done: false,
        }
    }

    /// Runtime identity allocated before this stream was returned.
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request
    }

    /// Record cancellation intent. Buffered output is preserved and an ordinary
    /// `Cancelled` terminal still arrives; this does not wait for the device.
    pub fn cancel(&self) {
        if let Some(stream) = &self.stream {
            stream.cancel();
        }
    }

    /// Await one event. Cancelling this future loses no event.
    pub async fn next_async(&mut self) -> Option<Result<TextEvent, TextError>> {
        if self.done {
            return None;
        }
        if let Some((reason, usage)) = self.pending.take() {
            self.done = true;
            return Some(Ok(TextEvent::Finished { reason, usage }));
        }
        let event = self.stream.as_mut()?.next().await;
        Some(self.accept(event?))
    }

    /// Blocking counterpart of [`Self::next_async`]. Not for async executor tasks.
    pub fn next_blocking(&mut self) -> Option<Result<TextEvent, TextError>> {
        if self.done {
            return None;
        }
        if let Some((reason, usage)) = self.pending.take() {
            self.done = true;
            return Some(Ok(TextEvent::Finished { reason, usage }));
        }
        let event = self.stream.as_mut()?.next_blocking();
        Some(self.accept(event?))
    }

    fn accept(
        &mut self,
        event: Result<Event, ribn::driver::DriverError>,
    ) -> Result<TextEvent, TextError> {
        match event {
            Ok(Event::Token { token, .. }) => {
                self.processor
                    .decode_token_into(token, &mut self.bytes)
                    .map_err(|error| self.abandon(error))?;
                let text = self
                    .decoder
                    .push(token, &self.bytes)
                    .map_err(|error| self.abandon(error))?;
                Ok(TextEvent::Delta {
                    token: Some(token),
                    text,
                })
            }
            Ok(Event::Finished { reason, usage, .. }) => {
                let trailing = self.decoder.finish();
                // Release decode scratch: a terminated stream keeps only its
                // terminal outcome until its consumer reads it.
                self.bytes = Vec::new();
                if trailing.is_empty() {
                    self.done = true;
                    return Ok(TextEvent::Finished { reason, usage });
                }
                self.pending = Some((reason, usage));
                Ok(TextEvent::Delta {
                    token: None,
                    text: trailing,
                })
            }
            // Owner failure preserves already-delivered events and does not
            // invent a successful terminal flush or usage.
            Err(error) => {
                self.done = true;
                Err(TextError::Owner(error))
            }
        }
    }

    /// A request-local decode failure abandons only this request.
    fn abandon(&mut self, error: TextError) -> TextError {
        self.done = true;
        self.bytes = Vec::new();
        // Dropping the driver stream records discard intent without joining.
        self.stream = None;
        error
    }
}

impl Iterator for TextStream {
    type Item = Result<TextEvent, TextError>;

    /// Blocking consumption. Async clients call [`TextStream::next_async`].
    fn next(&mut self) -> Option<Self::Item> {
        self.next_blocking()
    }
}

#[derive(Default)]
pub(crate) struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    /// Decode one token's bytes, retaining an incomplete final code point. The
    /// retained prefix is shorter than one code point, so it stays bounded.
    pub(crate) fn push(&mut self, token: u32, bytes: &[u8]) -> Result<String, TextError> {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let text = text.to_owned();
                self.pending.clear();
                Ok(text)
            }
            Err(error) if error.error_len().is_none() => {
                let valid = error.valid_up_to();
                let prefix = std::str::from_utf8(&self.pending[..valid])
                    .expect("prefix validated by from_utf8")
                    .to_owned();
                self.pending.drain(..valid);
                Ok(prefix)
            }
            Err(source) => Err(TextError::Encoding { token, source }),
        }
    }

    /// Flush an incomplete final code point exactly once. Terminal bookkeeping,
    /// not recovery: the replacement character is the only honest output.
    pub(crate) fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            String::new()
        } else {
            let trailing = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            trailing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8Decoder;

    #[test]
    fn incremental_decoder_retains_split_codepoints() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.push(1, &[0xe2]).unwrap(), "");
        assert_eq!(decoder.push(2, &[0x82]).unwrap(), "");
        assert_eq!(decoder.push(3, &[0xac]).unwrap(), "€");
    }

    #[test]
    fn terminal_incomplete_codepoint_is_replaced_once() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.push(1, &[0xe2]).unwrap(), "");
        assert_eq!(decoder.finish(), "\u{fffd}");
        assert!(decoder.finish().is_empty());
    }

    #[test]
    fn invalid_bytes_report_the_offending_token() {
        let mut decoder = Utf8Decoder::default();
        let error = decoder.push(7, &[0xff]).expect_err("invalid UTF-8");
        assert!(matches!(error, super::TextError::Encoding { token: 7, .. }));
    }
}
