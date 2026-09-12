use std::error::Error;
use std::fmt;

use ribn_batch::{BatchConfig, BatchExecutor, BatchRuntime, Job, JobOutput};
use ribn_foundation::{ParameterVersion, ScalarType};
use ribn_safetensors::SafeTensorArtifact;

#[derive(Clone, Debug, Eq, PartialEq)]
enum EncoderError {
    Artifact(String),
    UnsupportedWeights,
    ShapeOverflow,
    InvalidPayload,
    EmptyInput,
    InputTooLong,
    TokenOutOfRange(u32),
    BatchTooLarge { tokens: usize, limit: usize },
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(message) => write!(f, "encoder artifact error: {message}"),
            Self::UnsupportedWeights => {
                f.write_str("reference encoder requires one rank-2 F32 embedding tensor")
            }
            Self::ShapeOverflow => f.write_str("reference encoder shape overflowed"),
            Self::InvalidPayload => f.write_str("reference encoder tensor payload is invalid"),
            Self::EmptyInput => f.write_str("encoder input must not be empty"),
            Self::InputTooLong => f.write_str("encoder input is too long for the reference path"),
            Self::TokenOutOfRange(token) => write!(f, "encoder token {token} is out of range"),
            Self::BatchTooLarge { tokens, limit } => {
                write!(
                    f,
                    "encoder batch has {tokens} tokens but its limit is {limit}"
                )
            }
        }
    }
}

impl Error for EncoderError {}

/// Artifact-backed reference encoder used only to pressure-test model loading and
/// non-AR batching. SafeTensors remains an artifact representation; this model
/// integration decides that `embeddings.weight` is a logical embedding parameter.
struct ArtifactEncoder {
    version: ParameterVersion,
    vocab: usize,
    width: usize,
    embeddings: Vec<f32>,
    max_items: usize,
    max_batch_tokens: usize,
}

impl ArtifactEncoder {
    fn load(
        artifact: &SafeTensorArtifact,
        version: ParameterVersion,
        max_batch_tokens: usize,
    ) -> Result<Self, EncoderError> {
        let tensor = artifact
            .tensor("embeddings.weight")
            .map_err(|error| EncoderError::Artifact(error.to_string()))?;
        if tensor.scalar_type() != &ScalarType::F32 || tensor.shape().len() != 2 {
            return Err(EncoderError::UnsupportedWeights);
        }
        let vocab = tensor.shape()[0];
        let width = tensor.shape()[1];
        let elements = vocab
            .checked_mul(width)
            .ok_or(EncoderError::ShapeOverflow)?;
        let expected_bytes = elements.checked_mul(4).ok_or(EncoderError::ShapeOverflow)?;
        if tensor.data().len() != expected_bytes {
            return Err(EncoderError::InvalidPayload);
        }
        let mut embeddings = Vec::with_capacity(elements);
        for encoded in tensor.data().chunks_exact(4) {
            let bytes: [u8; 4] = encoded
                .try_into()
                .map_err(|_| EncoderError::InvalidPayload)?;
            embeddings.push(f32::from_le_bytes(bytes));
        }
        Ok(Self {
            version,
            vocab,
            width,
            embeddings,
            max_items: 4,
            max_batch_tokens,
        })
    }

    fn encode(&self, tokens: &[u32]) -> Result<Vec<f32>, EncoderError> {
        if tokens.is_empty() {
            return Err(EncoderError::EmptyInput);
        }
        let mut output = vec![0.0_f32; self.width];
        for &token_id in tokens {
            let token =
                usize::try_from(token_id).map_err(|_| EncoderError::TokenOutOfRange(token_id))?;
            if token >= self.vocab {
                return Err(EncoderError::TokenOutOfRange(token_id));
            }
            let start = token
                .checked_mul(self.width)
                .ok_or(EncoderError::ShapeOverflow)?;
            let end = start
                .checked_add(self.width)
                .ok_or(EncoderError::ShapeOverflow)?;
            for (destination, source) in output.iter_mut().zip(&self.embeddings[start..end]) {
                *destination += source;
            }
        }
        let length = u16::try_from(tokens.len()).map_err(|_| EncoderError::InputTooLong)?;
        let scale = 1.0 / f32::from(length);
        for value in &mut output {
            *value *= scale;
        }
        Ok(output)
    }
}

impl BatchExecutor for ArtifactEncoder {
    type Input = Vec<u32>;
    type Output = Vec<f32>;
    type Error = EncoderError;

    fn parameter_version(&self) -> ParameterVersion {
        self.version
    }

    fn max_batch_items(&self) -> usize {
        self.max_items
    }

    fn select_batch(&self, candidates: &[&Self::Input]) -> usize {
        let mut selected = 0;
        let mut tokens = 0_usize;
        for candidate in candidates {
            let next = tokens.saturating_add(candidate.len());
            if selected > 0 && next > self.max_batch_tokens {
                break;
            }
            tokens = next;
            selected += 1;
        }
        selected
    }

    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
        let batch_tokens = batch.iter().map(|job| job.input().len()).sum::<usize>();
        if batch_tokens > self.max_batch_tokens {
            return Err(EncoderError::BatchTooLarge {
                tokens: batch_tokens,
                limit: self.max_batch_tokens,
            });
        }
        batch
            .into_iter()
            .map(|job| {
                let request = job.request();
                let output = self.encode(job.input())?;
                Ok(JobOutput::new(request, output))
            })
            .collect()
    }
}

fn artifact_fixture() -> Vec<u8> {
    let rows = [
        [1.0_f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let mut data = Vec::new();
    for row in rows {
        for value in row {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut header = format!(
        r#"{{"embeddings.weight":{{"dtype":"F32","shape":[4,3],"data_offsets":[0,{}]}}}}"#,
        data.len()
    )
    .into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut artifact = Vec::new();
    artifact.extend_from_slice(
        &u64::try_from(header.len())
            .expect("fixture header length")
            .to_le_bytes(),
    );
    artifact.extend_from_slice(&header);
    artifact.extend_from_slice(&data);
    artifact
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((*actual - *expected).abs() <= 1.0e-6);
    }
}

#[test]
fn safetensors_weights_flow_through_model_mapping_and_non_ar_runtime() {
    let artifact = SafeTensorArtifact::from_bytes(artifact_fixture()).expect("artifact");
    let encoder = ArtifactEncoder::load(&artifact, ParameterVersion::new(23), 5)
        .expect("artifact-backed encoder");
    let mut runtime = BatchRuntime::new(
        encoder,
        BatchConfig {
            max_queued_requests: 8,
        },
    )
    .expect("runtime");

    let first = runtime.submit(vec![0, 1]).expect("first request");
    let second = runtime.submit(vec![2, 3, 0]).expect("second request");
    assert!(runtime.step().expect("batch"));

    let first_output = runtime.pop_completed().expect("first output");
    let second_output = runtime.pop_completed().expect("second output");
    assert_eq!(first_output.request(), first);
    assert_eq!(second_output.request(), second);
    assert_eq!(first_output.parameter_version(), ParameterVersion::new(23));
    assert_close(first_output.output(), &[0.5, 0.5, 0.0]);
    assert_close(second_output.output(), &[2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]);
}
