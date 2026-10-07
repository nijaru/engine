use engine_bert::{BertConfig, HostWeights};
use ribn_hf::LocalModelPackage;

fn add(values: &[f32], count: &mut usize, hash: &mut u64) {
    *count += values.len();
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: bert_loading_bench PACKAGE")?;
    let package = LocalModelPackage::open(path)?;
    let config = BertConfig::from_json(package.config())?;
    let started = std::time::Instant::now();
    let weights = HostWeights::load(&package, &config)?;
    let elapsed = started.elapsed();
    let mut count = 0;
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for values in [
        &weights.word_embeddings,
        &weights.position_embeddings,
        &weights.token_type_embeddings,
        &weights.embedding_norm.weight,
        &weights.embedding_norm.bias,
    ] {
        add(values, &mut count, &mut hash);
    }
    for layer in &weights.layers {
        for value in [
            &layer.query,
            &layer.key,
            &layer.value,
            &layer.attention_output,
            &layer.intermediate,
            &layer.output,
        ] {
            add(&value.weight, &mut count, &mut hash);
            add(&value.bias, &mut count, &mut hash);
        }
        for values in [
            &layer.attention_norm.weight,
            &layer.attention_norm.bias,
            &layer.output_norm.weight,
            &layer.output_norm.bias,
        ] {
            add(values, &mut count, &mut hash);
        }
    }
    add(&weights.pooler.weight, &mut count, &mut hash);
    add(&weights.pooler.bias, &mut count, &mut hash);
    println!(
        "parameters={count} hash={hash:016x} load_s={:.6}",
        elapsed.as_secs_f64()
    );
    std::hint::black_box(weights);
    Ok(())
}
