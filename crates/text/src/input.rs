/// One structured chat message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Text-generation input before model-specific formatting and tokenization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextInput {
    /// Raw text completion. No chat template is applied.
    Prompt(String),
    /// Structured chat messages. The model artifact's chat template is applied.
    Chat(Vec<Message>),
    /// Already-tokenized input. No text preprocessing is performed.
    Tokens(Vec<u32>),
}

impl TextInput {
    #[must_use]
    pub fn prompt(text: impl Into<String>) -> Self {
        Self::Prompt(text.into())
    }

    #[must_use]
    pub fn chat(messages: impl Into<Vec<Message>>) -> Self {
        Self::Chat(messages.into())
    }

    #[must_use]
    pub fn tokens(tokens: impl Into<Vec<u32>>) -> Self {
        Self::Tokens(tokens.into())
    }
}
