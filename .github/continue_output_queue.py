from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


p = Path("crates/core/src/serving_runtime.rs")
s = p.read_text()
s = replace_once(
    s,
    "use std::collections::HashMap;\n",
    "use std::collections::{HashMap, VecDeque};\n",
    "VecDeque import",
)
s = replace_once(
    s,
    """    prompt_tokens: HashMap<RequestId, Arc<[u32]>>,
    next_tokens: HashMap<RequestId, u32>,
}
""",
    """    prompt_tokens: HashMap<RequestId, Arc<[u32]>>,
    next_tokens: HashMap<RequestId, u32>,
    generated_tokens: VecDeque<GeneratedToken>,
}
""",
    "generated queue field",
)
s = replace_once(
    s,
    """            prompt_tokens: HashMap::new(),
            next_tokens: HashMap::new(),
        })
""",
    """            prompt_tokens: HashMap::new(),
            next_tokens: HashMap::new(),
            generated_tokens: VecDeque::new(),
        })
""",
    "generated queue initialization",
)
s = replace_once(
    s,
    """    #[must_use]
    pub fn submission_count(&self) -> usize {
        self.submissions.len()
    }

    /// # Errors
""",
    """    #[must_use]
    pub fn submission_count(&self) -> usize {
        self.submissions.len()
    }

    /// Number of committed generated-token events waiting for the frontend.
    #[must_use]
    pub fn generated_token_count(&self) -> usize {
        self.generated_tokens.len()
    }

    /// Pop the oldest committed generated token across all requests.
    ///
    /// The queue is intentionally independent of request-slot reclamation so a
    /// frontend cannot lose an already committed token by reclaiming terminal
    /// execution state before it drains output.
    pub fn pop_generated_token(&mut self) -> Option<GeneratedToken> {
        self.generated_tokens.pop_front()
    }

    /// # Errors
""",
    "generated queue accessors",
)
s = replace_once(
    s,
    """                        if let Some(token) = completed.output_token() {
                            self.next_tokens.insert(request, token);
                        }
""",
    """                        if let Some(token) = completed.output_token() {
                            self.next_tokens.insert(request, token);
                            self.generated_tokens
                                .push_back(GeneratedToken::new(request, token));
                        }
""",
    "enqueue generated token",
)
marker = """#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ServingIteration {
"""
generated = """/// One committed sampled token ready for a frontend to consume.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GeneratedToken {
    request: RequestId,
    token: u32,
}

impl GeneratedToken {
    #[must_use]
    pub const fn new(request: RequestId, token: u32) -> Self {
        Self { request, token }
    }

    #[must_use]
    pub const fn request(self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn token(self) -> u32 {
        self.token
    }
}

"""
s = replace_once(s, marker, generated + marker, "GeneratedToken type")
old = """        assert_eq!(serving.poll_completions().expect("pending poll"), 0);
        assert_eq!(serving.poll_completions().expect("prefill completion"), 1);

        serving.submit_ready_batch().expect("decode submit");
"""
new = """        assert_eq!(serving.poll_completions().expect("pending poll"), 0);
        assert_eq!(serving.poll_completions().expect("prefill completion"), 1);
        assert_eq!(serving.generated_token_count(), 1);
        assert_eq!(
            serving.pop_generated_token(),
            Some(GeneratedToken::new(
                RequestId::new(1).expect("request ID"),
                7
            ))
        );
        assert_eq!(serving.generated_token_count(), 0);

        serving.submit_ready_batch().expect("decode submit");
"""
s = replace_once(s, old, new, "generated queue serving test")
p.write_text(s)

p = Path("crates/core/src/lib.rs")
s = p.read_text()
s = replace_once(
    s,
    "pub use serving_runtime::{ServingIteration, ServingRuntime, ServingRuntimeError};\n",
    "pub use serving_runtime::{GeneratedToken, ServingIteration, ServingRuntime, ServingRuntimeError};\n",
    "generated token export",
)
p.write_text(s)
