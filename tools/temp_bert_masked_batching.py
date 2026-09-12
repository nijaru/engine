from pathlib import Path


path = Path("crates/batch/tests/bert_architecture.rs")
text = path.read_text()

old = '''    fn forward(&self, hidden_states: &[f32], config: &BertConfig) -> Vec<f32> {
        let sequence = hidden_states.len() / config.hidden_size;
        let query = self.query.forward_rows(hidden_states);
        let key = self.key.forward_rows(hidden_states);
        let value = self.value.forward_rows(hidden_states);
        let context = attention_context(
            &query,
            &key,
            &value,
            sequence,
            config.hidden_size,
            config.num_attention_heads,
        );
'''
new = '''    fn forward(
        &self,
        hidden_states: &[f32],
        attention_mask: Option<&[bool]>,
        config: &BertConfig,
    ) -> Vec<f32> {
        let sequence = hidden_states.len() / config.hidden_size;
        let query = self.query.forward_rows(hidden_states);
        let key = self.key.forward_rows(hidden_states);
        let value = self.value.forward_rows(hidden_states);
        let context = attention_context(
            &query,
            &key,
            &value,
            attention_mask,
            sequence,
            config.hidden_size,
            config.num_attention_heads,
        );
'''
if old not in text:
    raise SystemExit("BERT layer forward anchor not found")
text = text.replace(old, new, 1)

old = '''fn attention_context(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence: usize,
'''
new = '''fn attention_context(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    attention_mask: Option<&[bool]>,
    sequence: usize,
'''
if old not in text:
    raise SystemExit("attention_context signature anchor not found")
text = text.replace(old, new, 1)

old = '''            for key_position in 0..sequence {
                let mut score = 0.0_f32;
'''
new = '''            for key_position in 0..sequence {
                if attention_mask.is_some_and(|mask| !mask[key_position]) {
                    scores[key_position] = f32::NEG_INFINITY;
                    continue;
                }
                let mut score = 0.0_f32;
'''
if old not in text:
    raise SystemExit("attention score loop anchor not found")
text = text.replace(old, new, 1)

old = '''struct BertInput {
    token_ids: Vec<u32>,
    token_type_ids: Option<Vec<u32>>,
}

impl BertInput {
    fn new(token_ids: impl Into<Vec<u32>>) -> Self {
        Self {
            token_ids: token_ids.into(),
            token_type_ids: None,
        }
    }
}
'''
new = '''struct BertInput {
    token_ids: Vec<u32>,
    token_type_ids: Option<Vec<u32>>,
    attention_mask: Option<Vec<bool>>,
}

impl BertInput {
    fn new(token_ids: impl Into<Vec<u32>>) -> Self {
        Self {
            token_ids: token_ids.into(),
            token_type_ids: None,
            attention_mask: None,
        }
    }

    fn with_attention_mask(mut self, attention_mask: impl Into<Vec<bool>>) -> Self {
        self.attention_mask = Some(attention_mask.into());
        self
    }
}
'''
if old not in text:
    raise SystemExit("BertInput anchor not found")
text = text.replace(old, new, 1)

old = '''struct BertReference {
    config: BertConfig,
'''
new = '''#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BertBatchLayout {
    Ragged,
    Padded,
}

struct BertReference {
    config: BertConfig,
'''
if old not in text:
    raise SystemExit("BertReference anchor not found")
text = text.replace(old, new, 1)

old = '''    max_batch_items: usize,
    max_batch_tokens: usize,
    loaded_artifacts: usize,
'''
new = '''    max_batch_items: usize,
    max_batch_tokens: usize,
    batch_layout: BertBatchLayout,
    loaded_artifacts: usize,
'''
if old not in text:
    raise SystemExit("BertReference fields anchor not found")
text = text.replace(old, new, 1)

old = '''            max_batch_items,
            max_batch_tokens,
            loaded_artifacts,
        })
    }

    fn encode(&self, input: &BertInput) -> Result<BertOutput, ModelError> {
        let mut hidden_states = self.embeddings.forward(input, &self.config)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.config);
        }
'''
new = '''            max_batch_items,
            max_batch_tokens,
            batch_layout: BertBatchLayout::Ragged,
            loaded_artifacts,
        })
    }

    fn with_batch_layout(mut self, batch_layout: BertBatchLayout) -> Self {
        self.batch_layout = batch_layout;
        self
    }

    fn encode(&self, input: &BertInput) -> Result<BertOutput, ModelError> {
        let attention_mask = input.attention_mask.as_deref();
        if let Some(mask) = attention_mask {
            if mask.len() != input.token_ids.len() {
                return Err(ModelError::InvalidInput(
                    "attention_mask length must match token_ids".to_owned(),
                ));
            }
            if !mask.iter().copied().any(|value| value) {
                return Err(ModelError::InvalidInput(
                    "attention_mask must retain at least one token".to_owned(),
                ));
            }
        }
        let mut hidden_states = self.embeddings.forward(input, &self.config)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask, &self.config);
        }
'''
if old not in text:
    raise SystemExit("BertReference load/encode anchor not found")
text = text.replace(old, new, 1)

old = '''    fn select_batch(&self, candidates: &[&Self::Input]) -> usize {
        let mut selected = 0_usize;
        let mut tokens = 0_usize;
        for input in candidates {
            let Some(next) = tokens.checked_add(input.token_ids.len()) else {
                break;
            };
            if selected > 0 && next > self.max_batch_tokens {
                break;
            }
            selected += 1;
            tokens = next;
            if tokens >= self.max_batch_tokens {
                break;
            }
        }
        selected
    }
'''
new = '''    fn select_batch(&self, candidates: &[&Self::Input]) -> usize {
        let mut selected = 0_usize;
        let mut summed_tokens = 0_usize;
        let mut max_sequence = 0_usize;
        for input in candidates {
            let Some(next_sum) = summed_tokens.checked_add(input.token_ids.len()) else {
                break;
            };
            let next_selected = selected + 1;
            let next_max = max_sequence.max(input.token_ids.len());
            let projected_tokens = match self.batch_layout {
                BertBatchLayout::Ragged => next_sum,
                BertBatchLayout::Padded => next_max
                    .checked_mul(next_selected)
                    .unwrap_or(usize::MAX),
            };
            if selected > 0 && projected_tokens > self.max_batch_tokens {
                break;
            }
            selected = next_selected;
            summed_tokens = next_sum;
            max_sequence = next_max;
            if projected_tokens >= self.max_batch_tokens {
                break;
            }
        }
        selected
    }
'''
if old not in text:
    raise SystemExit("select_batch anchor not found")
text = text.replace(old, new, 1)

append = '''

#[test]
fn bert_attention_mask_makes_padding_semantically_inert_for_real_tokens() {
    let dir = TestDir::new();
    write_bert_package(dir.path());
    let package = LocalModelPackage::open(dir.path()).expect("package");
    let model = BertReference::load(&package, ParameterVersion::new(43), 4, 8).expect("BERT");

    let base = model
        .encode(&BertInput::new(vec![0, 1]))
        .expect("unmasked reference");
    let padded = model
        .encode(&BertInput::new(vec![0, 1, 2]).with_attention_mask(vec![true, true, false]))
        .expect("masked padded input");

    let real_values = base.sequence * base.hidden;
    assert_close(
        &padded.last_hidden_state[..real_values],
        &base.last_hidden_state,
    );
    assert_close(
        padded
            .pooled_output
            .as_deref()
            .expect("padded pooler output"),
        base.pooled_output.as_deref().expect("base pooler output"),
    );
}

#[test]
fn padded_layout_cost_can_shorten_a_batch_that_fits_ragged_execution() {
    let dir = TestDir::new();
    write_bert_package(dir.path());
    let package = LocalModelPackage::open(dir.path()).expect("package");

    let ragged = BertReference::load(&package, ParameterVersion::new(44), 4, 6)
        .expect("ragged BERT")
        .with_batch_layout(BertBatchLayout::Ragged);
    let mut ragged_runtime = BatchRuntime::new(
        ragged,
        BatchConfig {
            max_queued_requests: 4,
        },
    )
    .expect("ragged runtime");
    ragged_runtime
        .submit(BertInput::new(vec![0, 1, 2, 0]))
        .expect("ragged first");
    ragged_runtime
        .submit(BertInput::new(vec![1, 2]))
        .expect("ragged second");
    assert!(ragged_runtime.step().expect("ragged batch"));
    assert_eq!(ragged_runtime.queued(), 0);

    let padded = BertReference::load(&package, ParameterVersion::new(45), 4, 6)
        .expect("padded BERT")
        .with_batch_layout(BertBatchLayout::Padded);
    let mut padded_runtime = BatchRuntime::new(
        padded,
        BatchConfig {
            max_queued_requests: 4,
        },
    )
    .expect("padded runtime");
    padded_runtime
        .submit(BertInput::new(vec![0, 1, 2, 0]))
        .expect("padded first");
    padded_runtime
        .submit(BertInput::new(vec![1, 2]))
        .expect("padded second");
    assert!(padded_runtime.step().expect("padded batch"));
    assert_eq!(padded_runtime.queued(), 1);
    assert!(padded_runtime.step().expect("padded tail"));
    assert_eq!(padded_runtime.queued(), 0);
}
'''
if "fn bert_attention_mask_makes_padding_semantically_inert_for_real_tokens" in text:
    raise SystemExit("masked BERT tests already present")
path.write_text(text + append)
