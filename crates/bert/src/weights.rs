//! Host-side parameter loading and shape validation.
//!
//! The encoder declares which parameters a BERT checkpoint must carry and what
//! shape each one has. Resolving the package and reading tensor bytes belongs to
//! [`ribn_hf`]; interpreting those bytes as this model's parameters belongs here.

use std::collections::BTreeMap;
use std::fmt;

use ribn_hf::{LocalModelPackage, LocalWeightSet, PackageError};

use crate::config::BertConfig;

/// One dense projection: `weight` is `[output, input]` in row-major order, matching
/// the checkpoint's `torch.nn.Linear` layout, and `bias` has one value per output.
#[derive(Clone, Debug, PartialEq)]
pub struct Dense {
    pub input: usize,
    pub output: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One normalization layer's affine parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Every parameter of one encoder layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub query: Dense,
    pub key: Dense,
    pub value: Dense,
    pub attention_output: Dense,
    pub attention_norm: LayerNorm,
    pub intermediate: Dense,
    pub output: Dense,
    pub output_norm: LayerNorm,
}

/// A complete encoder's parameters on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct HostWeights {
    pub word_embeddings: Vec<f32>,
    pub position_embeddings: Vec<f32>,
    pub token_type_embeddings: Vec<f32>,
    pub embedding_norm: LayerNorm,
    pub layers: Vec<Layer>,
    pub pooler: Dense,
}

impl HostWeights {
    /// Read every parameter this encoder needs from a resolved local package.
    ///
    /// # Errors
    /// Returns [`WeightError`] for a missing parameter, a wrong shape or dtype, or
    /// a package resolution failure.
    pub fn load(package: &LocalModelPackage, config: &BertConfig) -> Result<Self, WeightError> {
        let hidden = config.hidden_size;
        let mut reader = Reader::new(package);
        let weights = Self {
            word_embeddings: reader.tensor(
                "embeddings.word_embeddings.weight",
                &[config.vocab_size, hidden],
            )?,
            position_embeddings: reader.tensor(
                "embeddings.position_embeddings.weight",
                &[config.max_position_embeddings, hidden],
            )?,
            token_type_embeddings: reader.tensor(
                "embeddings.token_type_embeddings.weight",
                &[config.type_vocab_size, hidden],
            )?,
            embedding_norm: LayerNorm {
                weight: reader.tensor("embeddings.LayerNorm.weight", &[hidden])?,
                bias: reader.tensor("embeddings.LayerNorm.bias", &[hidden])?,
            },
            layers: (0..config.num_hidden_layers)
                .map(|layer| {
                    let prefix = format!("encoder.layer.{layer}");
                    Ok(Layer {
                        query: reader.dense(
                            &format!("{prefix}.attention.self.query"),
                            hidden,
                            hidden,
                        )?,
                        key: reader.dense(
                            &format!("{prefix}.attention.self.key"),
                            hidden,
                            hidden,
                        )?,
                        value: reader.dense(
                            &format!("{prefix}.attention.self.value"),
                            hidden,
                            hidden,
                        )?,
                        attention_output: reader.dense(
                            &format!("{prefix}.attention.output.dense"),
                            hidden,
                            hidden,
                        )?,
                        attention_norm: LayerNorm {
                            weight: reader.tensor(
                                &format!("{prefix}.attention.output.LayerNorm.weight"),
                                &[hidden],
                            )?,
                            bias: reader.tensor(
                                &format!("{prefix}.attention.output.LayerNorm.bias"),
                                &[hidden],
                            )?,
                        },
                        intermediate: reader.dense(
                            &format!("{prefix}.intermediate.dense"),
                            hidden,
                            config.intermediate_size,
                        )?,
                        output: reader.dense(
                            &format!("{prefix}.output.dense"),
                            config.intermediate_size,
                            hidden,
                        )?,
                        output_norm: LayerNorm {
                            weight: reader
                                .tensor(&format!("{prefix}.output.LayerNorm.weight"), &[hidden])?,
                            bias: reader
                                .tensor(&format!("{prefix}.output.LayerNorm.bias"), &[hidden])?,
                        },
                    })
                })
                .collect::<Result<Vec<_>, WeightError>>()?,
            pooler: reader.dense("pooler.dense", hidden, hidden)?,
        };
        Ok(weights)
    }
}

impl Dense {
    /// Flattened row-major offset of `weight[row][column]`.
    #[must_use]
    pub const fn offset(&self, row: usize, column: usize) -> usize {
        row * self.input + column
    }
}

/// Why a checkpoint cannot supply this encoder's parameters.
#[derive(Debug)]
pub enum WeightError {
    Package(PackageError),
    Missing(String),
    Shape {
        parameter: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    Dtype {
        parameter: String,
    },
    Length {
        parameter: String,
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Package(error) => write!(f, "package resolution failed: {error}"),
            Self::Missing(parameter) => {
                write!(f, "checkpoint is missing required parameter {parameter}")
            }
            Self::Shape {
                parameter,
                expected,
                found,
            } => write!(
                f,
                "parameter {parameter} has shape {found:?}, expected {expected:?}"
            ),
            Self::Dtype { parameter } => write!(
                f,
                "parameter {parameter} is not F32; this encoder executes F32 checkpoints"
            ),
            Self::Length {
                parameter,
                expected,
                found,
            } => write!(
                f,
                "parameter {parameter} carries {found} values, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for WeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Package(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PackageError> for WeightError {
    fn from(error: PackageError) -> Self {
        Self::Package(error)
    }
}

struct Reader<'a> {
    weights: LocalWeightSet<'a>,
    /// Tensors already decoded, keyed by parameter name. A BERT parameter is
    /// consumed once each, so this only avoids a second decode within one load.
    decoded: BTreeMap<String, Vec<f32>>,
}

impl<'a> Reader<'a> {
    fn new(package: &'a LocalModelPackage) -> Self {
        Self {
            weights: package.weight_set(),
            decoded: BTreeMap::new(),
        }
    }

    fn dense(&mut self, prefix: &str, input: usize, output: usize) -> Result<Dense, WeightError> {
        Ok(Dense {
            input,
            output,
            weight: self.tensor(&format!("{prefix}.weight"), &[output, input])?,
            bias: self.tensor(&format!("{prefix}.bias"), &[output])?,
        })
    }

    fn tensor(&mut self, parameter: &str, expected: &[usize]) -> Result<Vec<f32>, WeightError> {
        if let Some(values) = self.decoded.get(parameter) {
            return Ok(values.clone());
        }
        let tensor = self
            .weights
            .tensor(parameter)
            .map_err(|error| match error {
                PackageError::UnknownParameter(_) => WeightError::Missing(parameter.to_owned()),
                other => WeightError::Package(other),
            })?;
        if tensor.scalar_type() != &ribn_foundation::ScalarType::F32 {
            return Err(WeightError::Dtype {
                parameter: parameter.to_owned(),
            });
        }
        if tensor.shape() != expected {
            return Err(WeightError::Shape {
                parameter: parameter.to_owned(),
                expected: expected.to_vec(),
                found: tensor.shape().to_vec(),
            });
        }
        let (chunks, remainder) = tensor.data().as_chunks::<4>();
        debug_assert!(
            remainder.is_empty(),
            "validated F32 payload is 4-byte aligned"
        );
        let values = chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<f32>>();
        if values.len() != expected.iter().product::<usize>() {
            return Err(WeightError::Length {
                parameter: parameter.to_owned(),
                expected: expected.iter().product(),
                found: values.len(),
            });
        }
        self.decoded.insert(parameter.to_owned(), values.clone());
        Ok(values)
    }
}
