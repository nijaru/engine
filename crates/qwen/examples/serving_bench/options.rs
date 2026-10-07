use super::trace::{Objectives, Workload};
use super::{DEFAULT_CONCURRENCY, DEFAULT_OUTPUT_TOKENS, PROBE_PROMPTS, PROMPT};
use engine_nvidia::GemvMode;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent benchmark input, output and experimental execution flags"
)]
pub(super) struct Options {
    pub concurrency: usize,
    pub output_tokens: u32,
    pub prompt: Arc<[u32]>,
    pub fixture_prompt: bool,
    pub gemv_mode: GemvMode,
    pub grouped_decode: bool,
    pub prefill_chunk: Option<usize>,
    pub print_tokens: bool,
    pub divergence_probe: bool,
    pub context_tokens: u32,
    pub workload: Option<Workload>,
    pub queue_capacity: usize,
    pub continuation_capacity_bytes: Option<u64>,
    pub objectives: Objectives,
    pub result_json: Option<PathBuf>,
}

impl Options {
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        validate_arguments(arguments)?;
        let workload = arguments
            .iter()
            .find_map(|arg| arg.strip_prefix("--workload="))
            .map(|path| Workload::read(Path::new(path)))
            .transpose()?;
        let queue_capacity = parse_usize(arguments, "--queue-capacity=", 0)?;
        let result_json = arguments
            .iter()
            .find_map(|arg| arg.strip_prefix("--result-json="))
            .map(PathBuf::from);
        let continuation_capacity_bytes =
            parse_optional_u64(arguments, "--continuation-capacity-bytes=")?;
        if continuation_capacity_bytes == Some(0) {
            return Err("continuation capacity must be positive".to_owned());
        }
        let objectives = Objectives {
            ttft_ms: parse_optional_u64(arguments, "--ttft-slo-ms=")?,
            itl_ms: parse_optional_u64(arguments, "--itl-slo-ms=")?,
            e2e_ms: parse_optional_u64(arguments, "--e2e-slo-ms=")?,
        };
        if workload.is_some()
            && arguments.iter().any(|arg| {
                ["--tokens=", "--prompt-fixture=", "--prompt-tokens="]
                    .iter()
                    .any(|prefix| arg.starts_with(prefix))
                    || arg == "--divergence-probe"
            })
        {
            return Err("--workload replaces prompt/token/probe flags".to_owned());
        }
        if workload.is_none()
            && (queue_capacity != 0 || objectives.configured() || result_json.is_some())
        {
            return Err("queue, SLO and result-json options require --workload".to_owned());
        }
        let concurrency = parse_usize(arguments, "--concurrency=", DEFAULT_CONCURRENCY)?;
        let output_tokens = parse_u32(arguments, "--tokens=", DEFAULT_OUTPUT_TOKENS)?;
        if concurrency == 0 || output_tokens == 0 {
            return Err("concurrency and token count must be greater than zero".to_owned());
        }
        let gemv_mode = parse_gemv_mode(arguments)?;
        let prefill_chunk = arguments
            .iter()
            .find_map(|argument| argument.strip_prefix("--prefill-chunk="))
            .map(|value| match value {
                "off" => Ok(None),
                value => value
                    .parse::<usize>()
                    .map(Some)
                    .map_err(|_| format!("--prefill-chunk expects off or a number, got {value}")),
            })
            .transpose()?
            .unwrap_or_else(|| {
                workload
                    .as_ref()
                    .map(|_| engine_qwen::DEFAULT_PREFILL_CHUNK_MEMBERS)
            });
        if prefill_chunk == Some(0) {
            return Err("--prefill-chunk must be greater than zero".to_owned());
        }
        let print_tokens = arguments
            .iter()
            .any(|argument| argument == "--print-tokens");
        let input = prompt_input(arguments, print_tokens, output_tokens)?;
        let context_tokens = workload
            .as_ref()
            .map_or(input.context_tokens, |workload| workload.context_tokens);
        Ok(Self {
            concurrency,
            output_tokens,
            prompt: input.tokens,
            fixture_prompt: input.fixture,
            gemv_mode,
            grouped_decode: arguments
                .iter()
                .any(|argument| argument == "--grouped-decode"),
            prefill_chunk,
            print_tokens,
            divergence_probe: input.divergence,
            context_tokens,
            workload,
            queue_capacity,
            continuation_capacity_bytes,
            objectives,
            result_json,
        })
    }

    pub fn prompt_for(&self, index: usize) -> Arc<[u32]> {
        if self.divergence_probe {
            Arc::from(PROBE_PROMPTS[index % PROBE_PROMPTS.len()])
        } else {
            Arc::clone(&self.prompt)
        }
    }
}

fn validate_arguments(arguments: &[String]) -> Result<(), String> {
    let allowed = [
        "--concurrency=",
        "--tokens=",
        "--prompt-fixture=",
        "--prompt-tokens=",
        "--gemv=",
        "--prefill-chunk=",
        "--model=",
        "--workload=",
        "--queue-capacity=",
        "--continuation-capacity-bytes=",
        "--ttft-slo-ms=",
        "--itl-slo-ms=",
        "--e2e-slo-ms=",
        "--result-json=",
    ];
    let mut seen = std::collections::HashSet::new();
    for argument in arguments {
        let key = allowed
            .iter()
            .find(|prefix| argument.starts_with(**prefix))
            .copied()
            .or_else(|| {
                ["--print-tokens", "--divergence-probe", "--grouped-decode"]
                    .into_iter()
                    .find(|flag| argument == flag)
            })
            .ok_or_else(|| format!("unknown option {argument}"))?;
        if !seen.insert(key) {
            return Err(format!("duplicate option {key}"));
        }
    }
    Ok(())
}

struct PromptInput {
    tokens: Arc<[u32]>,
    fixture: bool,
    divergence: bool,
    context_tokens: u32,
}

fn prompt_input(
    arguments: &[String],
    print_tokens: bool,
    output_tokens: u32,
) -> Result<PromptInput, String> {
    if arguments.iter().any(|arg| arg == "--divergence-probe")
        && arguments
            .iter()
            .any(|arg| arg.starts_with("--prompt-fixture=") || arg.starts_with("--prompt-tokens="))
    {
        return Err("--divergence-probe replaces fixture/prompt-length flags".to_owned());
    }
    let fixture_tokens = arguments
        .iter()
        .find_map(|argument| argument.strip_prefix("--prompt-fixture="))
        .map(|path| read_fixture_prompt(Path::new(path)))
        .transpose()?;
    let prompt_tokens = parse_usize(
        arguments,
        "--prompt-tokens=",
        fixture_tokens.as_ref().map_or(PROMPT.len(), Vec::len),
    )?;
    if prompt_tokens == 0 {
        return Err("prompt token count must be greater than zero".to_owned());
    }
    let divergence_probe = arguments
        .iter()
        .any(|argument| argument == "--divergence-probe");
    if divergence_probe && !print_tokens {
        return Err(
            "--divergence-probe requires --print-tokens to emit comparable token streams"
                .to_owned(),
        );
    }
    let longest_prompt = if divergence_probe {
        PROBE_PROMPTS
            .iter()
            .map(|prompt| prompt.len())
            .max()
            .expect("six prompts")
    } else {
        prompt_tokens
    };
    let context_tokens = u32::try_from(longest_prompt)
        .map_err(|_| "prompt length does not fit the runtime".to_owned())?
        .checked_add(output_tokens)
        .ok_or_else(|| "prompt plus output budget overflowed".to_owned())?;
    // Without a fixture, repeat the fixed prompt. This is a timing fixture,
    // not real prompt content. Fixture lengths always select a prefix.
    let prompt = match &fixture_tokens {
        Some(tokens) => {
            if tokens.len() < prompt_tokens {
                return Err(format!(
                    "prompt fixture holds {} tokens, fewer than the requested {prompt_tokens}",
                    tokens.len()
                ));
            }
            Arc::from(&tokens[..prompt_tokens])
        }
        None => Arc::from(
            PROMPT
                .iter()
                .copied()
                .cycle()
                .take(prompt_tokens)
                .collect::<Vec<_>>(),
        ),
    };
    Ok(PromptInput {
        tokens: prompt,
        fixture: fixture_tokens.is_some(),
        divergence: divergence_probe,
        context_tokens,
    })
}

/// Read the fixture format used in `crates/qwen/tests/fixtures`: artifact
/// identity, prompt token IDs, then reference continuation (unused for timing).
pub(super) fn read_fixture_prompt(path: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read prompt fixture {}: {error}", path.display()))?;
    let mut lines = text.lines();
    lines
        .next()
        .ok_or_else(|| format!("prompt fixture {} has no identity line", path.display()))?;
    lines
        .next()
        .ok_or_else(|| format!("prompt fixture {} has no prompt line", path.display()))?
        .split_whitespace()
        .map(|token| {
            token
                .parse::<u32>()
                .map_err(|error| format!("prompt fixture {} token: {error}", path.display()))
        })
        .collect()
}

fn parse_usize(arguments: &[String], prefix: &str, default: usize) -> Result<usize, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix(prefix))
        .map_or(Ok(default), |value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{prefix} expects an integer"))
        })
}

fn parse_u32(arguments: &[String], prefix: &str, default: u32) -> Result<u32, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix(prefix))
        .map_or(Ok(default), |value| {
            value
                .parse::<u32>()
                .map_err(|_| format!("{prefix} expects an integer"))
        })
}

fn parse_gemv_mode(arguments: &[String]) -> Result<GemvMode, String> {
    arguments
        .iter()
        .find_map(|argument| argument.strip_prefix("--gemv="))
        .map_or(Ok(GemvMode::default()), |value| match value {
            "scalar" => Ok(GemvMode::Scalar),
            "warp" => Ok(GemvMode::Warp),
            "int-dot" => Ok(GemvMode::IntegerDot),
            other => Err(format!(
                "--gemv expects scalar, warp, or int-dot, got {other}"
            )),
        })
}

fn parse_optional_u64(arguments: &[String], prefix: &str) -> Result<Option<u64>, String> {
    arguments
        .iter()
        .find_map(|arg| arg.strip_prefix(prefix))
        .map(|value| {
            value
                .parse()
                .map_err(|_| format!("{prefix} expects an integer"))
        })
        .transpose()
}
