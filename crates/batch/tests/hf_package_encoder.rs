use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, Job, JobOutput, StepOutcome,
};
use ribn_foundation::{ParameterVersion, ScalarType};
use ribn_hf::LocalModelPackage;

static NEXT_DIR: AtomicU64 = AtomicU64::new(1);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("ribn-hf-encoder-{}-{id}", process::id()));
        fs::create_dir(&path).expect("test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug)]
enum ModelError {
    UnsupportedArchitecture,
    Package(String),
    UnsupportedWeights,
    InvalidPayload,
    TokenOutOfRange(u32),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArchitecture => f.write_str("unsupported fixture architecture"),
            Self::Package(message) => write!(f, "model package error: {message}"),
            Self::UnsupportedWeights => f.write_str("unsupported fixture embedding weights"),
            Self::InvalidPayload => f.write_str("invalid fixture embedding payload"),
            Self::TokenOutOfRange(token) => write!(f, "token {token} is out of range"),
        }
    }
}

impl Error for ModelError {}

struct PackageEncoder {
    version: ParameterVersion,
    vocab: usize,
    width: usize,
    embeddings: Vec<f32>,
}

impl PackageEncoder {
    fn load(package: &LocalModelPackage, version: ParameterVersion) -> Result<Self, ModelError> {
        if package.config()["architectures"][0].as_str() != Some("FixtureEncoder") {
            return Err(ModelError::UnsupportedArchitecture);
        }
        let artifact = package
            .open_weights_for("embeddings.weight")
            .map_err(|error| ModelError::Package(error.to_string()))?;
        let tensor = artifact
            .tensor("embeddings.weight")
            .map_err(|error| ModelError::Package(error.to_string()))?;
        if tensor.scalar_type() != &ScalarType::F32 || tensor.shape().len() != 2 {
            return Err(ModelError::UnsupportedWeights);
        }
        let vocab = tensor.shape()[0];
        let width = tensor.shape()[1];
        let (values, remainder) = tensor.data().as_chunks::<4>();
        if !remainder.is_empty() || values.len() != vocab.saturating_mul(width) {
            return Err(ModelError::InvalidPayload);
        }
        let embeddings = values
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        Ok(Self {
            version,
            vocab,
            width,
            embeddings,
        })
    }

    fn encode(&self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let mut output = vec![0.0_f32; self.width];
        for &token_id in tokens {
            let token =
                usize::try_from(token_id).map_err(|_| ModelError::TokenOutOfRange(token_id))?;
            if token >= self.vocab {
                return Err(ModelError::TokenOutOfRange(token_id));
            }
            let start = token.saturating_mul(self.width);
            for (destination, source) in output
                .iter_mut()
                .zip(&self.embeddings[start..start + self.width])
            {
                *destination += source;
            }
        }
        let length = u16::try_from(tokens.len()).map_err(|_| ModelError::InvalidPayload)?;
        if length == 0 {
            return Err(ModelError::InvalidPayload);
        }
        let scale = 1.0 / f32::from(length);
        for value in &mut output {
            *value *= scale;
        }
        Ok(output)
    }
}

impl BatchExecutor for PackageEncoder {
    type Input = Vec<u32>;
    type Output = Vec<f32>;
    type Error = ModelError;
    type Constraint = Infallible;

    fn parameter_version(&self) -> ParameterVersion {
        self.version
    }

    fn max_batch_items(&self) -> usize {
        8
    }

    fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
        BatchSelection::Ready {
            items: candidates.len(),
        }
    }

    fn retained_bytes(&self, _input: &Self::Input) -> u64 {
        u64::try_from(self.width)
            .unwrap_or(u64::MAX)
            .saturating_mul(4)
    }

    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, Self::Error> {
        batch
            .into_iter()
            .map(|job| {
                let request = job.request();
                Ok(JobOutput::new(request, self.encode(job.input())?))
            })
            .collect()
    }
}

fn safetensors_fixture() -> Vec<u8> {
    let values = [
        1.0_f32, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, // token 1
        0.0, 0.0, 1.0, // token 2
    ];
    let mut data = Vec::new();
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    let mut header = format!(
        r#"{{"embeddings.weight":{{"dtype":"F32","shape":[3,3],"data_offsets":[0,{}]}}}}"#,
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
fn hf_package_metadata_and_weights_remain_separate_from_model_semantics() {
    let dir = TestDir::new();
    fs::write(
        dir.path().join("config.json"),
        br#"{"architectures":["FixtureEncoder"],"hidden_size":3}"#,
    )
    .expect("config");
    fs::write(dir.path().join("model.safetensors"), safetensors_fixture()).expect("weights");
    fs::write(dir.path().join("tokenizer.json"), b"{}\n").expect("tokenizer metadata");

    let package = LocalModelPackage::open(dir.path()).expect("package");
    let encoder = PackageEncoder::load(&package, ParameterVersion::new(31)).expect("encoder");
    let mut runtime = BatchRuntime::new(
        encoder,
        BatchConfig {
            max_waiting_requests: 8,
            ..BatchConfig::default()
        },
    )
    .expect("runtime");
    let request = runtime.submit(vec![0, 2]).expect("request");
    assert!(matches!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 1 }
    ));
    let result = runtime.pop_completed().expect("result");
    assert_eq!(result.request(), request);
    assert_eq!(result.parameter_version(), ParameterVersion::new(31));
    assert_close(result.output().expect("result"), &[0.5, 0.0, 0.5]);
    assert!(package.package_file("tokenizer.json").is_ok());
}
