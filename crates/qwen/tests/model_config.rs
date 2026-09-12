//! This model definition is usable with no GGUF or CUDA feature enabled.
use engine_qwen::{QwenConfig, QwenLayerKind};

fn config() -> QwenConfig {
    QwenConfig {
        context_length: 262_144,
        embedding_length: 5120,
        feed_forward_length: 17_408,
        block_count: 65,
        attention_heads: 24,
        kv_heads: 4,
        key_length: 256,
        value_length: 256,
        full_attention_interval: 4,
        ssm_group_count: 16,
        ssm_inner_size: 6144,
        ssm_state_size: 128,
        ssm_time_step_rank: 48,
        ssm_conv_kernel: 4,
        nextn_predict_layers: 1,
    }
}

#[test]
fn model_geometry_and_layers_do_not_require_a_checkpoint_reader() {
    let config = config();
    config.validate().unwrap();
    assert_eq!(config.language_layer_count(), Some(64));
    for layer in 0..64 {
        let expected = if (layer + 1) % 4 == 0 {
            QwenLayerKind::FullAttention
        } else {
            QwenLayerKind::Recurrent
        };
        assert_eq!(config.layer_kind(layer).unwrap(), expected);
    }
    assert!(config.layer_kind(64).is_err());
    assert!(config.layer_kind(u32::MAX).is_err());
}

#[test]
fn malformed_geometry_is_rejected_without_division_or_index_panics() {
    let mut config = config();
    config.full_attention_interval = 0;
    assert!(config.validate().is_err());
    assert!(config.layer_kind(0).is_err());
    config.full_attention_interval = 4;
    config.nextn_predict_layers = 66;
    assert!(config.validate().is_err());
}
