//! Architecture configuration independent of the artifact format that carried it.

use std::fmt;

use serde_json::Value;

/// Geometry and numerical policy of one BERT encoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl BertConfig {
    /// Read a `config.json` value produced by a Hugging Face checkpoint.
    ///
    /// Only the fields this execution path uses are required; a checkpoint may
    /// carry many more without affecting them. `hidden_act` must be `gelu` because
    /// the device path implements that activation and silently substituting
    /// another would change the model's arithmetic.
    ///
    /// # Errors
    /// Returns [`ConfigError`] for a missing, wrongly typed or unusable field.
    pub fn from_json(config: &Value) -> Result<Self, ConfigError> {
        let model_type = config.get("model_type").and_then(Value::as_str);
        if model_type != Some("bert") {
            return Err(ConfigError::UnsupportedModelType(
                model_type.unwrap_or("<absent>").to_owned(),
            ));
        }
        let activation = config
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("gelu");
        if activation != "gelu" {
            return Err(ConfigError::UnsupportedActivation(activation.to_owned()));
        }
        let config = Self {
            vocab_size: required_usize(config, "vocab_size")?,
            hidden_size: required_usize(config, "hidden_size")?,
            num_hidden_layers: required_usize(config, "num_hidden_layers")?,
            num_attention_heads: required_usize(config, "num_attention_heads")?,
            intermediate_size: required_usize(config, "intermediate_size")?,
            max_position_embeddings: required_usize(config, "max_position_embeddings")?,
            type_vocab_size: required_usize(config, "type_vocab_size")?,
            layer_norm_eps: required_f32(config, "layer_norm_eps")?,
        };
        config.validate()?;
        Ok(config)
    }

    /// # Errors
    /// Returns [`ConfigError`] for geometry that cannot describe a BERT encoder.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.intermediate_size == 0
            || self.max_position_embeddings == 0
            || self.type_vocab_size == 0
        {
            return Err(ConfigError::NonPositive);
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(ConfigError::IndivisibleHeads {
                hidden_size: self.hidden_size,
                heads: self.num_attention_heads,
            });
        }
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            return Err(ConfigError::InvalidEpsilon(self.layer_norm_eps));
        }
        Ok(())
    }

    /// Width of one attention head.
    #[must_use]
    pub const fn head_width(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Why a configuration cannot drive this encoder.
#[derive(Debug)]
pub enum ConfigError {
    MissingField(&'static str),
    InvalidField { field: &'static str, value: String },
    UnsupportedModelType(String),
    UnsupportedActivation(String),
    NonPositive,
    IndivisibleHeads { hidden_size: usize, heads: usize },
    InvalidEpsilon(f32),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "config.json is missing {field}"),
            Self::InvalidField { field, value } => {
                write!(f, "config.json field {field} is not usable: {value}")
            }
            Self::UnsupportedModelType(found) => write!(
                f,
                "this encoder executes model_type=bert checkpoints, found {found:?}"
            ),
            Self::UnsupportedActivation(found) => write!(
                f,
                "this encoder executes hidden_act=gelu, found {found:?}; substituting \
                 another activation would change the model's arithmetic"
            ),
            Self::NonPositive => f.write_str("every BERT geometry field must be positive"),
            Self::IndivisibleHeads { hidden_size, heads } => write!(
                f,
                "hidden_size {hidden_size} is not divisible by {heads} attention heads"
            ),
            Self::InvalidEpsilon(epsilon) => {
                write!(
                    f,
                    "layer_norm_eps must be finite and positive, found {epsilon}"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn required_usize(config: &Value, field: &'static str) -> Result<usize, ConfigError> {
    let value = config.get(field).ok_or(ConfigError::MissingField(field))?;
    let number = value.as_u64().ok_or_else(|| ConfigError::InvalidField {
        field,
        value: value.to_string(),
    })?;
    usize::try_from(number).map_err(|_| ConfigError::InvalidField {
        field,
        value: number.to_string(),
    })
}

fn required_f32(config: &Value, field: &'static str) -> Result<f32, ConfigError> {
    let value = config.get(field).ok_or(ConfigError::MissingField(field))?;
    let number = value.as_f64().ok_or_else(|| ConfigError::InvalidField {
        field,
        value: value.to_string(),
    })?;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "checkpoint epsilon is a small positive scalar, not model data"
    )]
    let narrowed = number as f32;
    if !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(ConfigError::InvalidField {
            field,
            value: number.to_string(),
        });
    }
    Ok(narrowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(overrides: &[(&str, Value)]) -> Value {
        let mut value = serde_json::json!({
            "model_type": "bert",
            "hidden_act": "gelu",
            "vocab_size": 13,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "intermediate_size": 20,
            "max_position_embeddings": 16,
            "type_vocab_size": 3,
            "layer_norm_eps": 1.0e-12,
        });
        for (key, replacement) in overrides {
            value[key] = replacement.clone();
        }
        value
    }

    #[test]
    fn reads_the_fixture_geometry() {
        let parsed = BertConfig::from_json(&config(&[])).unwrap();
        assert_eq!(parsed.hidden_size, 12);
        assert_eq!(parsed.num_hidden_layers, 2);
        assert_eq!(parsed.head_width(), 4);
        assert!((parsed.layer_norm_eps - 1.0e-12).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_geometry_that_cannot_execute() {
        assert!(matches!(
            BertConfig::from_json(&config(&[("model_type", serde_json::json!("roberta"))]))
                .unwrap_err(),
            ConfigError::UnsupportedModelType(_)
        ));
        assert!(matches!(
            BertConfig::from_json(&config(&[("hidden_act", serde_json::json!("relu"))]))
                .unwrap_err(),
            ConfigError::UnsupportedActivation(_)
        ));
        assert!(matches!(
            BertConfig::from_json(&config(&[("num_attention_heads", serde_json::json!(5))]))
                .unwrap_err(),
            ConfigError::IndivisibleHeads { .. }
        ));
        assert!(matches!(
            BertConfig::from_json(&config(&[("hidden_size", serde_json::json!(0))])).unwrap_err(),
            ConfigError::NonPositive
        ));
        assert!(matches!(
            BertConfig::from_json(&config(&[("layer_norm_eps", serde_json::json!(0.0))]))
                .unwrap_err(),
            ConfigError::InvalidField { .. }
        ));
        let missing = config(&[]);
        let mut missing = missing.as_object().unwrap().clone();
        missing.remove("vocab_size");
        assert!(matches!(
            BertConfig::from_json(&Value::Object(missing)).unwrap_err(),
            ConfigError::MissingField("vocab_size")
        ));
        assert!(matches!(
            BertConfig::from_json(&config(&[("hidden_size", serde_json::json!("twelve"))]))
                .unwrap_err(),
            ConfigError::InvalidField { .. }
        ));
    }
}
