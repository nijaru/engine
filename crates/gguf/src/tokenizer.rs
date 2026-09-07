//! Byte-BPE tokenization and embedded text chat templates.
use crate::{
    GgufError, MetadataValue, optional_u32, required_i32_array, required_string,
    required_string_array, required_u32,
};
use engine_core::{PromptFormat, PromptPolicy, SpecialTokenPolicy};
use regex::Regex;
use std::collections::BTreeMap;

/// A text role/content pair passed to the artifact's embedded chat template.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    #[must_use]
    pub fn role(&self) -> &str {
        &self.role
    }

    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// Options controlling the generation suffix emitted by Qwen's chat template.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChatTemplateOptions {
    add_generation_prompt: bool,
    enable_thinking: bool,
}

impl ChatTemplateOptions {
    #[must_use]
    pub const fn new(add_generation_prompt: bool, enable_thinking: bool) -> Self {
        Self {
            add_generation_prompt,
            enable_thinking,
        }
    }

    #[must_use]
    pub const fn add_generation_prompt(self) -> bool {
        self.add_generation_prompt
    }

    #[must_use]
    pub const fn enable_thinking(self) -> bool {
        self.enable_thinking
    }
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self::new(true, true)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTokenizer {
    model: String,
    pretokenizer: String,
    chat_template: Option<String>,
    tokens: Vec<String>,
    merges: Vec<String>,
    token_types: Vec<i32>,
    bos_token_id: u32,
    eos_token_id: u32,
    padding_token_id: Option<u32>,
    token_ids: BTreeMap<String, u32>,
    merge_ranks: BTreeMap<(String, String), u32>,
}

const QWEN35_PRETOKENIZER: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s+";

fn byte_is_direct(byte: u8) -> bool {
    (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
}

fn byte_to_unicode(byte: u8) -> char {
    if byte_is_direct(byte) {
        char::from(byte)
    } else {
        let rank = (0..byte)
            .filter(|candidate| !byte_is_direct(*candidate))
            .count();
        char::from_u32(256 + u32::try_from(rank).unwrap_or(0)).unwrap_or('\u{fffd}')
    }
}

fn unicode_to_byte(character: char) -> Option<u8> {
    let codepoint = u32::from(character);
    if let Ok(byte) = u8::try_from(codepoint)
        && byte_is_direct(byte)
    {
        return Some(byte);
    }
    let rank = codepoint.checked_sub(256)?;
    let mut current_rank = 0;
    for byte in 0..=u8::MAX {
        if !byte_is_direct(byte) {
            if current_rank == rank {
                return Some(byte);
            }
            current_rank += 1;
        }
    }
    None
}

fn build_token_ids(tokens: &[String]) -> Result<BTreeMap<String, u32>, GgufError> {
    let mut token_ids = BTreeMap::new();
    for (index, token) in tokens.iter().enumerate() {
        let index = u32::try_from(index).map_err(|_| {
            GgufError::InvalidTokenizer("vocabulary is too large for a u32 token ID")
        })?;
        if token_ids.insert(token.clone(), index).is_some() {
            return Err(GgufError::InvalidTokenizer(
                "vocabulary contains duplicate tokens",
            ));
        }
    }
    Ok(token_ids)
}

fn build_merge_ranks(merges: &[String]) -> Result<BTreeMap<(String, String), u32>, GgufError> {
    let mut ranks = BTreeMap::new();
    for (rank, merge) in merges.iter().enumerate() {
        let (left, right) = merge
            .split_once(' ')
            .ok_or_else(|| GgufError::TokenizerEncoding {
                detail: format!("merge {merge:?} does not contain two symbols"),
            })?;
        if left.is_empty() || right.is_empty() || right.contains(' ') {
            return Err(GgufError::TokenizerEncoding {
                detail: format!("merge {merge:?} does not contain exactly two symbols"),
            });
        }
        let rank = u32::try_from(rank)
            .map_err(|_| GgufError::InvalidTokenizer("merge table is too large for a u32 rank"))?;
        if ranks
            .insert((left.to_owned(), right.to_owned()), rank)
            .is_some()
        {
            return Err(GgufError::InvalidTokenizer(
                "merge table contains duplicates",
            ));
        }
    }
    Ok(ranks)
}

impl GgufTokenizer {
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    #[must_use]
    pub fn pretokenizer(&self) -> &str {
        &self.pretokenizer
    }

    #[must_use]
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    #[must_use]
    pub fn merges(&self) -> &[String] {
        &self.merges
    }

    #[must_use]
    pub fn token_types(&self) -> &[i32] {
        &self.token_types
    }

    #[must_use]
    pub const fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    #[must_use]
    pub const fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    #[must_use]
    pub const fn padding_token_id(&self) -> Option<u32> {
        self.padding_token_id
    }

    /// Encode ordinary text with the embedded Qwen3.5 GPT-2/BPE vocabulary.
    /// Special-token parsing is intentionally not implicit; callers must handle
    /// chat-template or special-token policy before invoking this method.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::InvalidTokenizer`] for an unsupported pretokenizer,
    /// or [`GgufError::TokenizerEncoding`] when the pretokenizer leaves a gap,
    /// a merge is malformed, or a final BPE symbol is absent from the
    /// vocabulary.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, GgufError> {
        if self.model != "gpt2" || self.pretokenizer != "qwen35" {
            return Err(GgufError::InvalidTokenizer(
                "only the embedded Qwen3.5 GPT-2 tokenizer is supported",
            ));
        }
        let pretokenizer = Regex::new(QWEN35_PRETOKENIZER)
            .map_err(|_| GgufError::InvalidTokenizer("invalid Qwen3.5 pretokenizer pattern"))?;
        let mut encoded = Vec::new();
        let mut end = 0;
        for part in pretokenizer.find_iter(text) {
            if part.start() != end {
                return Err(GgufError::TokenizerEncoding {
                    detail: "pretokenizer did not cover the complete input".to_owned(),
                });
            }
            end = part.end();
            let mapped: Vec<String> = part
                .as_str()
                .as_bytes()
                .iter()
                .map(|byte| byte_to_unicode(*byte).to_string())
                .collect();
            let symbols = self.bpe(mapped);
            for symbol in symbols {
                let token_id = self.token_ids.get(&symbol).copied().ok_or_else(|| {
                    GgufError::TokenizerEncoding {
                        detail: format!("BPE symbol {symbol:?} is absent from the vocabulary"),
                    }
                })?;
                encoded.push(token_id);
            }
        }
        if end != text.len() {
            return Err(GgufError::TokenizerEncoding {
                detail: "pretokenizer did not cover the complete input".to_owned(),
            });
        }
        Ok(encoded)
    }

    /// Decode GPT-2/BPE token IDs back to UTF-8 text.
    ///
    /// Special-marker tokens are returned literally, which lets callers keep
    /// stop-marker handling explicit while still supporting chat prompt and
    /// generated-text round trips.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::TokenizerEncoding`] when a token ID is outside the
    /// vocabulary, a token contains an unsupported symbol, or the decoded byte
    /// sequence is not UTF-8.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, GgufError> {
        String::from_utf8(self.decode_bytes(token_ids)?).map_err(|error| {
            GgufError::TokenizerEncoding {
                detail: format!("decoded tokens are not UTF-8: {error}"),
            }
        })
    }

    /// Decode token IDs to bytes, preserving incomplete UTF-8 at token boundaries.
    /// Text frontends choose their own finalization policy for truncated output.
    ///
    /// # Errors
    /// Returns an error for invalid token IDs or unsupported vocabulary symbols.
    pub fn decode_bytes(&self, token_ids: &[u32]) -> Result<Vec<u8>, GgufError> {
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            let token = self
                .tokens
                .get(
                    usize::try_from(token_id).map_err(|_| GgufError::TokenizerEncoding {
                        detail: format!("token ID {token_id} does not fit this platform"),
                    })?,
                )
                .ok_or_else(|| GgufError::TokenizerEncoding {
                    detail: format!("token ID {token_id} is outside the vocabulary"),
                })?;
            for character in token.chars() {
                let byte =
                    unicode_to_byte(character).ok_or_else(|| GgufError::TokenizerEncoding {
                        detail: format!(
                            "token {token_id} contains unsupported symbol {character:?}"
                        ),
                    })?;
                bytes.push(byte);
            }
        }
        Ok(bytes)
    }

    /// Render the artifact's embedded template for text-only messages.
    ///
    /// Tool calls and multimodal payloads are not represented by this API.
    /// Unsupported template syntax or operations are errors, never a fallback
    /// to a different template. Rendering is bounded by `MiniJinja` fuel.
    ///
    /// # Errors
    /// Returns an error for missing/unsupported templates or invalid roles.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String, GgufError> {
        let template = self
            .chat_template
            .as_deref()
            .ok_or(GgufError::UnsupportedPromptFormat(
                "the GGUF has no embedded chat template",
            ))?;
        if messages.iter().any(|message| {
            !matches!(
                message.role(),
                "system" | "user" | "assistant" | "developer"
            )
        }) {
            return Err(GgufError::UnsupportedPromptFormat(
                "only text chat roles are supported",
            ));
        }
        let mut environment = minijinja::Environment::new();
        environment.set_fuel(Some(1_000_000));
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        environment
            .add_template("chat", template)
            .map_err(|error| template_error(&error))?;
        environment
            .get_template("chat")
            .map_err(|error| template_error(&error))?
            .render(minijinja::context! {
                messages => messages,
                add_generation_prompt => options.add_generation_prompt(),
                enable_thinking => options.enable_thinking(),
            })
            .map_err(|error| template_error(&error))
    }

    /// Render and encode the embedded Qwen text chat template, preserving its
    /// special markers as their vocabulary IDs instead of passing them through
    /// ordinary GPT-2/BPE encoding.
    ///
    /// # Errors
    ///
    /// Returns chat-template, special-token, or ordinary tokenizer errors.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<Vec<u32>, GgufError> {
        let rendered = self.render_chat(messages, options)?;
        self.encode_rendered_chat(&rendered)
    }

    fn encode_rendered_chat(&self, rendered: &str) -> Result<Vec<u32>, GgufError> {
        const SPECIAL_MARKERS: [&str; 4] = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"];
        let mut encoded = Vec::new();
        let mut cursor = 0;
        while cursor < rendered.len() {
            let next = SPECIAL_MARKERS
                .iter()
                .filter_map(|marker| {
                    rendered[cursor..]
                        .find(marker)
                        .map(|index| (index, *marker))
                })
                .min_by_key(|(index, _)| *index);
            let Some((relative_index, marker)) = next else {
                encoded.extend(self.encode(&rendered[cursor..])?);
                break;
            };
            let marker_start = cursor + relative_index;
            if marker_start > cursor {
                encoded.extend(self.encode(&rendered[cursor..marker_start])?);
            }
            let token_id = self.token_ids.get(marker).copied().ok_or_else(|| {
                GgufError::TokenizerEncoding {
                    detail: format!("chat special token {marker:?} is absent from the vocabulary"),
                }
            })?;
            encoded.push(token_id);
            cursor = marker_start + marker.len();
        }
        Ok(encoded)
    }

    /// Encode text according to an explicit request prompt policy. Plain text
    /// remains separate from [`Self::encode_chat`], because a chat request must
    /// carry message structure rather than being silently flattened.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::UnsupportedPromptFormat`] for an embedded chat
    /// template and forwards ordinary tokenizer errors from [`Self::encode`].
    pub fn encode_with_policy(
        &self,
        text: &str,
        policy: PromptPolicy,
    ) -> Result<Vec<u32>, GgufError> {
        if policy.format() == PromptFormat::EmbeddedChatTemplate {
            return Err(GgufError::UnsupportedPromptFormat(
                "embedded chat templates require structured messages; use encode_chat",
            ));
        }
        let mut encoded = self.encode(text)?;
        match policy.special_tokens() {
            SpecialTokenPolicy::None => {}
            SpecialTokenPolicy::AddBos => encoded.insert(0, self.bos_token_id()),
            SpecialTokenPolicy::AddEos => encoded.push(self.eos_token_id()),
            SpecialTokenPolicy::AddBosAndEos => {
                encoded.insert(0, self.bos_token_id());
                encoded.push(self.eos_token_id());
            }
        }
        Ok(encoded)
    }

    fn bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for index in 0..symbols.len() - 1 {
                let pair = (symbols[index].clone(), symbols[index + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair)
                    && best.is_none_or(|(best_rank, _)| rank < best_rank)
                {
                    best = Some((rank, index));
                }
            }
            let Some((_, index)) = best else {
                break;
            };
            let merged = format!("{}{}", symbols[index], symbols[index + 1]);
            symbols.splice(index..=index + 1, [merged]);
        }
        symbols
    }

    pub(crate) fn from_metadata(
        metadata: &BTreeMap<String, MetadataValue>,
    ) -> Result<Self, GgufError> {
        let tokens = required_string_array(metadata, "tokenizer.ggml.tokens")?;
        let merges = required_string_array(metadata, "tokenizer.ggml.merges")?;
        let token_types = required_i32_array(metadata, "tokenizer.ggml.token_type")?;
        if tokens.is_empty() || tokens.len() != token_types.len() {
            return Err(GgufError::InvalidTokenizer(
                "token and token-type arrays must have the same nonzero length",
            ));
        }
        let token_ids = build_token_ids(&tokens)?;
        let merge_ranks = build_merge_ranks(&merges)?;
        let chat_template = metadata
            .get("tokenizer.chat_template")
            .map(|value| {
                value.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                    GgufError::MetadataTypeMismatch {
                        key: "tokenizer.chat_template".to_owned(),
                    }
                })
            })
            .transpose()?;
        Ok(Self {
            model: required_string(metadata, "tokenizer.ggml.model")?,
            pretokenizer: required_string(metadata, "tokenizer.ggml.pre")?,
            chat_template,
            tokens,
            merges,
            token_types,
            bos_token_id: required_u32(metadata, "tokenizer.ggml.bos_token_id")?,
            eos_token_id: required_u32(metadata, "tokenizer.ggml.eos_token_id")?,
            padding_token_id: optional_u32(metadata, "tokenizer.ggml.padding_token_id")?,
            token_ids,
            merge_ranks,
        })
    }
}

fn template_error(error: &minijinja::Error) -> GgufError {
    GgufError::TokenizerEncoding {
        detail: format!("embedded chat template: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_embedded_tokenizer_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            MetadataValue::String("gpt2".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.pre".to_owned(),
            MetadataValue::String("qwen35".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            MetadataValue::Array(vec![
                MetadataValue::String("a".to_owned()),
                MetadataValue::String("b".to_owned()),
                MetadataValue::String("ab".to_owned()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_owned(),
            MetadataValue::Array(vec![MetadataValue::String("a b".to_owned())]),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            MetadataValue::Array(vec![
                MetadataValue::I32(1),
                MetadataValue::I32(3),
                MetadataValue::I32(1),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_owned(),
            MetadataValue::U32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_owned(),
            MetadataValue::U32(2),
        );
        metadata.insert(
            "tokenizer.ggml.padding_token_id".to_owned(),
            MetadataValue::U32(0),
        );
        metadata.insert(
            "tokenizer.chat_template".to_owned(),
            MetadataValue::String("{{ messages[0].content }}".to_owned()),
        );

        let tokenizer = GgufTokenizer::from_metadata(&metadata).expect("tokenizer metadata");
        assert_eq!(tokenizer.model(), "gpt2");
        assert_eq!(tokenizer.pretokenizer(), "qwen35");
        assert_eq!(tokenizer.tokens(), &["a", "b", "ab"]);
        assert_eq!(tokenizer.merges(), &["a b"]);
        assert_eq!(tokenizer.token_types(), &[1, 3, 1]);
        assert_eq!(tokenizer.encode("ab").expect("BPE encoding"), vec![2]);
        assert_eq!(tokenizer.decode(&[2]).expect("BPE decoding"), "ab");
        assert!(matches!(
            tokenizer.decode(&[99]),
            Err(GgufError::TokenizerEncoding { .. })
        ));
        assert!(tokenizer.encode("").expect("empty encoding").is_empty());
        assert_eq!(tokenizer.bos_token_id(), 1);
        assert_eq!(tokenizer.eos_token_id(), 2);
        assert_eq!(tokenizer.padding_token_id(), Some(0));
        assert_eq!(tokenizer.chat_template(), Some("{{ messages[0].content }}"));
        let messages = [
            ChatMessage::new("system", " system "),
            ChatMessage::new("user", " question "),
            ChatMessage::new("assistant", " answer "),
        ];
        assert_eq!(
            tokenizer
                .render_chat(&messages, ChatTemplateOptions::default())
                .expect("Qwen chat rendering"),
            " system "
        );
        assert!(matches!(
            tokenizer.render_chat(
                &[ChatMessage::new("tool", "result")],
                ChatTemplateOptions::new(false, false),
            ),
            Err(GgufError::UnsupportedPromptFormat(_))
        ));
        assert!(matches!(
            tokenizer.encode_chat(&messages, ChatTemplateOptions::default()),
            Err(GgufError::TokenizerEncoding { .. })
        ));
        assert_eq!(
            tokenizer
                .encode_with_policy(
                    "ab",
                    PromptPolicy::new(PromptFormat::PlainText, SpecialTokenPolicy::AddBosAndEos,),
                )
                .expect("explicit boundary tokens"),
            vec![1, 2, 2]
        );
        assert!(matches!(
            tokenizer.encode_with_policy(
                "ab",
                PromptPolicy::new(PromptFormat::EmbeddedChatTemplate, SpecialTokenPolicy::None,),
            ),
            Err(GgufError::UnsupportedPromptFormat(_))
        ));
    }

    #[test]
    fn encodes_qwen_chat_special_tokens_without_bpe_flattening() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            MetadataValue::String("gpt2".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.pre".to_owned(),
            MetadataValue::String("qwen35".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            MetadataValue::Array(
                [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "a",
                    "s",
                    "i",
                    "t",
                    "n",
                    "Ċ",
                ]
                .into_iter()
                .map(|token| MetadataValue::String(token.to_owned()))
                .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_owned(),
            MetadataValue::Array(Vec::new()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            MetadataValue::Array(vec![MetadataValue::I32(4); 10]),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_owned(),
            MetadataValue::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_owned(),
            MetadataValue::U32(1),
        );
        metadata.insert(
            "tokenizer.chat_template".to_owned(),
            MetadataValue::String("{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% if not enable_thinking %}\n</think>\n\n{% endif %}{% endif %}".to_owned()),
        );
        let tokenizer = GgufTokenizer::from_metadata(&metadata).expect("chat tokenizer");
        assert_eq!(
            tokenizer
                .encode_chat(&[], ChatTemplateOptions::new(true, true))
                .expect("thinking generation prompt"),
            vec![0, 4, 5, 5, 6, 5, 7, 4, 8, 7, 9, 2, 9]
        );
        assert_eq!(
            tokenizer
                .encode_chat(&[], ChatTemplateOptions::new(true, false))
                .expect("non-thinking generation prompt"),
            vec![0, 4, 5, 5, 6, 5, 7, 4, 8, 7, 9, 2, 9, 9, 3, 9, 9]
        );
    }

    fn tokenizer(template: &str) -> GgufTokenizer {
        let tokens = (0..=255_u8)
            .map(|byte| byte_to_unicode(byte).to_string())
            .collect::<Vec<_>>();
        GgufTokenizer {
            model: "gpt2".into(),
            pretokenizer: "qwen35".into(),
            chat_template: Some(template.into()),
            token_ids: build_token_ids(&tokens).unwrap(),
            tokens,
            merges: Vec::new(),
            merge_ranks: BTreeMap::new(),
            token_types: vec![1; 256],
            bos_token_id: 0,
            eos_token_id: 1,
            padding_token_id: None,
        }
    }

    #[test]
    fn byte_decoding_preserves_partial_unicode_at_a_token_limit() {
        let tokenizer = tokenizer("");
        let tokens = tokenizer.encode("A€").unwrap();
        assert_eq!(tokenizer.decode(&tokens).unwrap(), "A€");
        let partial = tokenizer.decode_bytes(&tokens[..tokens.len() - 1]).unwrap();
        assert_eq!(partial, b"A\xe2\x82");
        assert_eq!(String::from_utf8_lossy(&partial), "A�");
        assert!(tokenizer.decode(&tokens[..tokens.len() - 1]).is_err());
        assert!(tokenizer.decode_bytes(&[256]).is_err());
    }

    #[test]
    fn pinned_template_matches_python_jinja_rendering() {
        let tokenizer = tokenizer(include_str!("../tests/fixtures/qwen38-chat.jinja"));
        assert_eq!(
            tokenizer
                .render_chat(
                    &[ChatMessage::new("user", "Hello")],
                    ChatTemplateOptions::new(true, false)
                )
                .unwrap(),
            include_str!("../tests/fixtures/qwen38-chat-0.txt")
        );
        let messages = [
            ChatMessage::new("system", "Be concise"),
            ChatMessage::new("developer", "Be accurate"),
            ChatMessage::new("user", "Hello"),
            ChatMessage::new("assistant", "Hi"),
            ChatMessage::new("user", "Again"),
        ];
        assert_eq!(
            tokenizer
                .render_chat(&messages, ChatTemplateOptions::new(true, true))
                .unwrap(),
            include_str!("../tests/fixtures/qwen38-chat-1.txt")
        );
        assert!(
            tokenizer
                .render_chat(&[], ChatTemplateOptions::default())
                .is_err()
        );
        assert!(
            tokenizer
                .render_chat(
                    &[
                        ChatMessage::new("user", "Hello"),
                        ChatMessage::new("system", "late")
                    ],
                    ChatTemplateOptions::default()
                )
                .is_err()
        );
    }

    #[test]
    fn template_execution_budget_rejects_excessive_work() {
        let template = "{% for i in range(100000) %}{% for j in range(100000) %}{% set k = i + j %}{% endfor %}{% endfor %}";
        assert!(
            tokenizer(template)
                .render_chat(&[], ChatTemplateOptions::default())
                .is_err()
        );
    }

    #[test]
    fn invalid_or_unsupported_templates_are_not_replaced() {
        for template in [
            "{% invalid %}",
            "{{ raise_exception('invalid chat') }}",
            "{{ unknown_function() }}",
        ] {
            assert!(
                tokenizer(template)
                    .render_chat(&[], ChatTemplateOptions::default())
                    .is_err()
            );
        }
        assert_eq!(
            tokenizer("custom {{ messages[0].content }}")
                .render_chat(
                    &[ChatMessage::new("user", "Hello")],
                    ChatTemplateOptions::default()
                )
                .unwrap(),
            "custom Hello"
        );
    }
}
