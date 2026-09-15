//! Cloneable text generation handle over one owned token execution worker.

use std::collections::VecDeque;
use std::sync::Arc;

use ribn::driver::{DriverOwner, GenerationHandle};
use ribn::{FinishReason, GenerationOptions, Usage};

use crate::error::TextError;
use crate::input::TextInput;
use crate::process::{Pool, PoolClient, Prepared, ProcessorLimits, TextProcessor};
use crate::stream::{TextEvent, TextStream};

/// One text-generation request: input plus generation options.
#[derive(Clone, Debug, PartialEq)]
pub struct TextRequest {
    pub input: TextInput,
    pub options: GenerationOptions,
}

impl TextRequest {
    #[must_use]
    pub const fn new(input: TextInput, options: GenerationOptions) -> Self {
        Self { input, options }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextResponse {
    pub text: String,
    pub tokens: Vec<u32>,
    pub reason: FinishReason,
    pub usage: Usage,
}

/// Assembly settings for the text layer. Execution bounds come from the token
/// driver the handle already owns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextConfig {
    /// Preprocessing worker threads. Bounds concurrent tokenization.
    pub preprocessing_workers: usize,
    /// Offline batch admission window, clamped to the driver's request permits.
    pub batch_window: usize,
}

impl Default for TextConfig {
    fn default() -> Self {
        let workers = std::thread::available_parallelism().map_or(1, |count| count.get().min(4));
        Self {
            preprocessing_workers: workers,
            batch_window: 8,
        }
    }
}

/// Cloneable text-generation handle.
///
/// Every clone shares one execution worker and one preprocessing pool. The
/// non-cloneable [`TextOwner`] controls shutdown.
#[derive(Clone)]
pub struct TextModel {
    processor: Arc<dyn TextProcessor>,
    pool: PoolClient,
    handle: GenerationHandle,
    window: usize,
}

impl TextModel {
    #[must_use]
    pub fn limits(&self) -> ProcessorLimits {
        self.processor.limits()
    }

    /// Offline batch admission window, bounded by the driver's request permits.
    #[must_use]
    pub const fn batch_window(&self) -> usize {
        self.window
    }

    /// Reserve, preprocess on the bounded pool, then enqueue.
    ///
    /// # Errors
    /// Overload, a request-local preparation failure, or a closed owner.
    async fn prepare(
        &self,
        input: TextInput,
        options: GenerationOptions,
    ) -> Result<Prepared, TextError> {
        let permit = self.handle.try_reserve().map_err(TextError::Admission)?;
        let reply = self.pool.submit(input, options, permit)?;
        reply.recv_async().await.map_err(|_| TextError::Closed)?
    }

    /// Blocking counterpart of [`Self::prepare`]. Not for async executor tasks.
    fn prepare_blocking(
        &self,
        input: TextInput,
        options: GenerationOptions,
    ) -> Result<Prepared, TextError> {
        let permit = self.handle.try_reserve().map_err(TextError::Admission)?;
        let reply = self.pool.submit(input, options, permit)?;
        reply.recv().map_err(|_| TextError::Closed)?
    }

    /// Start a streaming request. Returns after runtime enqueue, not after model
    /// admission or device completion.
    ///
    /// # Errors
    /// Overload, a request-local preparation failure, an enqueue rejection, or a
    /// failed execution owner.
    pub async fn stream(
        &self,
        input: TextInput,
        options: GenerationOptions,
    ) -> Result<TextStream, TextError> {
        let prepared = self.prepare(input, options).await?;
        let stream = prepared
            .permit
            .stream(prepared.request)
            .await
            .map_err(TextError::Admission)?;
        Ok(TextStream::new(self.processor.clone(), stream))
    }

    /// Blocking counterpart of [`Self::stream`]. Not for async executor tasks.
    ///
    /// # Errors
    /// Overload, a request-local preparation failure, an enqueue rejection, or a
    /// failed execution owner.
    pub fn stream_blocking(
        &self,
        input: TextInput,
        options: GenerationOptions,
    ) -> Result<TextStream, TextError> {
        let prepared = self.prepare_blocking(input, options)?;
        let stream = prepared
            .permit
            .stream_blocking(prepared.request)
            .map_err(TextError::Admission)?;
        Ok(TextStream::new(self.processor.clone(), stream))
    }

    /// Generate one response to completion.
    ///
    /// # Errors
    /// Stream failures and a missing terminal event.
    pub async fn generate(
        &self,
        input: TextInput,
        options: GenerationOptions,
    ) -> Result<TextResponse, TextError> {
        let mut stream = self.stream(input, options).await?;
        let mut collector = Collector::default();
        while let Some(event) = stream.next_async().await {
            collector.apply(event?);
        }
        collector.finish()
    }

    /// Blocking counterpart of [`Self::generate`]. Not for async executor tasks.
    ///
    /// # Errors
    /// Stream failures and a missing terminal event.
    pub fn generate_blocking(
        &self,
        input: TextInput,
        options: GenerationOptions,
    ) -> Result<TextResponse, TextError> {
        let mut stream = self.stream_blocking(input, options)?;
        let mut collector = Collector::default();
        while let Some(event) = stream.next_blocking() {
            collector.apply(event?);
        }
        collector.finish()
    }

    /// Ordered incremental results over a bounded admission window. Blocking;
    /// async callers interleave their own [`Self::stream`] futures instead.
    pub fn batch<'a>(
        &'a self,
        requests: impl IntoIterator<Item = TextRequest> + 'a,
    ) -> TextBatch<'a> {
        TextBatch {
            model: self,
            source: Box::new(requests.into_iter()),
            window: VecDeque::with_capacity(self.window),
            drained: false,
        }
    }

    /// Collect ordered per-item outcomes.
    ///
    /// Each item settles independently, so the returned collection has no
    /// batch-wide error. This intentionally allocates every requested result;
    /// use [`Self::batch`] for a bounded workload.
    pub fn generate_batch(
        &self,
        requests: impl IntoIterator<Item = TextRequest>,
    ) -> Vec<Result<TextResponse, TextError>> {
        self.batch(requests).collect()
    }
}

/// Ordered, incremental offline batching over a bounded admission window.
///
/// The caller's iterator is pulled lazily as window capacity frees, so lookahead
/// is bounded and an unencodable input settles only itself. Dropping the batch
/// abandons its live streams without waiting for device completion.
pub struct TextBatch<'a> {
    model: &'a TextModel,
    source: Box<dyn Iterator<Item = TextRequest> + 'a>,
    window: VecDeque<Result<TextStream, TextError>>,
    drained: bool,
}

impl TextBatch<'_> {
    fn fill(&mut self) {
        while !self.drained && self.window.len() < self.model.window {
            let Some(request) = self.source.next() else {
                self.drained = true;
                break;
            };
            let stream = self.model.stream_blocking(request.input, request.options);
            self.window.push_back(stream);
        }
    }

    /// Next input's ordered outcome, or `None` once every input settled.
    pub fn next_result(&mut self) -> Option<Result<TextResponse, TextError>> {
        self.fill();
        let front = self.window.pop_front()?;
        let result = match front {
            Err(error) => Err(error),
            Ok(mut stream) => {
                let mut collector = Collector::default();
                let mut failure = None;
                while let Some(event) = stream.next_blocking() {
                    match event {
                        Ok(event) => collector.apply(event),
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    }
                }
                match failure {
                    Some(error) => Err(error),
                    None => collector.finish(),
                }
            }
        };
        self.fill();
        Some(result)
    }
}

impl Iterator for TextBatch<'_> {
    type Item = Result<TextResponse, TextError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_result()
    }
}

#[derive(Default)]
struct Collector {
    text: String,
    tokens: Vec<u32>,
    finish: Option<(FinishReason, Usage)>,
}

impl Collector {
    fn apply(&mut self, event: TextEvent) {
        match event {
            TextEvent::Delta { token, text } => {
                if let Some(token) = token {
                    self.tokens.push(token);
                }
                self.text.push_str(&text);
            }
            TextEvent::Finished { reason, usage } => self.finish = Some((reason, usage)),
        }
    }

    fn finish(self) -> Result<TextResponse, TextError> {
        let (reason, usage) = self.finish.ok_or(TextError::MissingTerminal)?;
        Ok(TextResponse {
            text: self.text,
            tokens: self.tokens,
            reason,
            usage,
        })
    }
}

/// Non-cloneable lifecycle owner for one loaded text model.
///
/// Dropping it requests final cleanup; call an explicit shutdown method to
/// observe synchronization and release failures.
pub struct TextOwner {
    model: TextModel,
    driver: DriverOwner,
    pool: Pool,
}

impl TextOwner {
    /// Assemble the text layer over an already spawned token driver.
    ///
    /// # Errors
    /// Rejects invalid processor bounds or zero workers/window.
    pub fn new(
        processor: Arc<dyn TextProcessor>,
        driver: DriverOwner,
        handle: GenerationHandle,
        config: TextConfig,
    ) -> Result<Self, TextError> {
        if !processor.limits().is_valid() {
            return Err(TextError::InvalidInput(
                "text processor bounds must all be nonzero",
            ));
        }
        if config.preprocessing_workers == 0 {
            return Err(TextError::InvalidInput(
                "text preprocessing needs at least one worker",
            ));
        }
        if config.batch_window == 0 {
            return Err(TextError::InvalidInput("batch window must be nonzero"));
        }
        let window = config.batch_window.min(handle.capacity()).max(1);
        let pool = Pool::start(&processor, config.preprocessing_workers, handle.capacity());
        Ok(Self {
            model: TextModel {
                processor,
                pool: pool.client(),
                handle,
                window,
            },
            driver,
            pool,
        })
    }

    #[must_use]
    pub const fn model(&self) -> &TextModel {
        &self.model
    }

    /// Close preprocessing and establish device completion.
    ///
    /// # Errors
    /// Reports cleanup failure; the same worker remains available for retry.
    pub fn shutdown(&mut self) -> Result<(), TextError> {
        self.pool.close();
        self.driver.shutdown().map_err(TextError::Owner)
    }

    /// Async counterpart of [`Self::shutdown`], independent of the caller's
    /// runtime. Cancelling it keeps the driver's retry ownership.
    ///
    /// # Errors
    /// Reports cleanup failure; the same worker remains available for retry.
    pub async fn shutdown_async(&mut self) -> Result<(), TextError> {
        self.pool.close();
        self.driver.shutdown_async().await.map_err(TextError::Owner)
    }
}

impl Drop for TextOwner {
    fn drop(&mut self) {
        // Reject new preprocessing even while cloned handles still exist.
        self.pool.close();
    }
}
