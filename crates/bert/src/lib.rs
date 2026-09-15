//! BERT-style bidirectional encoder execution.
//!
//! This crate owns one model family's semantics: which parameters exist, how a
//! package's tensors map onto them, and in what order the encoder applies them. It
//! does not own artifacts ([`ribn_hf`] resolves a package), kernels or physical
//! device state (`engine-nvidia` owns those), or scheduling and resource accounting.
//!
//! The encoder exists in this repository as a second, materially different model
//! path: it is bidirectional, pooling-capable and non-autoregressive, so it
//! pressure-tests the boundaries that a decoder-only path cannot.

mod config;
mod weights;

pub use config::{BertConfig, ConfigError};
pub use weights::{HostWeights, WeightError};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{CudaBertEncoder, EncoderError, EncoderOutput, EncoderRequest};
