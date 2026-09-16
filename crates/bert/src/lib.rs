//! BERT-style bidirectional encoder execution.
//!
//! This crate owns one model family's semantics: which parameters exist, how a
//! package's tensors map onto them, and in what order the encoder applies them. It
//! does not own artifacts ([`ribn_hf`] resolves a package), kernels or physical
//! device state (`engine-nvidia` owns those), or the batching runtime
//! ([`ribn_batch`] owns reservation, readiness and result retention).
//!
//! What it does own on the scheduling side is its own executor seam: which request
//! shapes it accepts, how many device bytes each accepted request holds, and the
//! device-resident result a consumer must await before reading. [`request`] makes
//! those decisions on the host, [`executor`] binds them to the batching runtime, and
//! device execution consumes them rather than re-deriving them.
//!
//! The encoder exists in this repository as a second, materially different model
//! path: it is bidirectional, pooling-capable and non-autoregressive, so it
//! pressure-tests the boundaries that a decoder-only path cannot.

mod config;
mod request;
mod weights;

pub use config::{BertConfig, ConfigError};
pub use request::{
    EncoderConstraint, EncoderRequest, EncoderShape, envelope, select, shape, shape_of, validate,
};
pub use weights::{HostWeights, WeightError};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
mod executor;

#[cfg(feature = "cuda")]
pub use cuda::{CudaBertEncoder, EncoderError, EncoderOutput, EncoderSubmission};

#[cfg(feature = "cuda")]
pub use executor::{EncoderCompletion, EncoderResult};
