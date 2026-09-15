//! Host-side checks that the fixture package maps onto this encoder's parameters.

use std::path::PathBuf;

use engine_bert::{BertConfig, ConfigError, HostWeights};
use ribn_hf::LocalModelPackage;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../batch/tests/fixtures/bert-tiny")
}

#[test]
fn fixture_package_supplies_every_parameter() {
    let package = LocalModelPackage::open(fixture()).expect("fixture package");
    let config = BertConfig::from_json(package.config()).expect("fixture config");
    let weights = HostWeights::load(&package, &config).expect("fixture parameters");

    assert_eq!(weights.word_embeddings.len(), 13 * 12);
    assert_eq!(weights.position_embeddings.len(), 16 * 12);
    assert_eq!(weights.token_type_embeddings.len(), 3 * 12);
    assert_eq!(weights.layers.len(), 2);
    assert_eq!(weights.embedding_norm.weight.len(), 12);
    assert_eq!(weights.pooler.weight.len(), 12 * 12);
    for layer in &weights.layers {
        assert_eq!(layer.query.output, 12);
        assert_eq!(layer.query.offset(11, 11), 143);
        assert_eq!(layer.intermediate.input, 12);
        assert_eq!(layer.intermediate.output, 20);
        assert_eq!(layer.output.input, 20);
        assert_eq!(layer.output.output, 12);
    }
    assert!(weights.word_embeddings.iter().any(|value| *value != 0.0));
}

#[test]
fn unsupported_architecture_is_named_rather_than_inferred() {
    let package = LocalModelPackage::open(fixture()).expect("fixture package");
    let mut config = package.config().clone();
    config["model_type"] = serde_json::json!("roberta");
    assert!(matches!(
        BertConfig::from_json(&config).unwrap_err(),
        ConfigError::UnsupportedModelType(_)
    ));
}
