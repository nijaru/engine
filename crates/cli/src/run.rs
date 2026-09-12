use std::io::{self, Write};

use engine_gguf::{ChatMessage, ChatTemplateOptions, GgufFile, GgufTokenizer};
use engine_qwen::{QwenCuda, QwenLoadOptions};
use ribn::{
    Engine, EngineConfig, Event, FinishReason, GenerationOptions, SchedulePolicy, TokenRequest,
};

const USAGE: &str =
    "ribn run --model <model.gguf> --prompt <text> [--max-tokens <n>] [--device <ordinal>]";

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    if matches!(arguments, [help] if matches!(help.as_str(), "-h" | "--help")) {
        println!(
            "{USAGE}\nExperimental runtime; greedy text generation on CUDA. GPU qualification is pending."
        );
        return Ok(());
    }
    let options = crate::cli::parse(arguments, USAGE)?;
    let file = GgufFile::open(options.model.clone()).map_err(display)?;
    let tokenizer = file.tokenizer().map_err(display)?;
    let tokens = tokenizer
        .encode_chat(
            &[ChatMessage::new("user", options.prompt)],
            ChatTemplateOptions::new(true, false),
        )
        .map_err(display)?;
    drop(file);
    let prompt_tokens = u32::try_from(tokens.len()).map_err(display)?;
    let context_tokens = prompt_tokens
        .checked_add(options.max_tokens)
        .ok_or("prompt plus output budget overflowed")?;
    eprintln!(
        "preparing Qwen on CUDA device {} (experimental Ribn runtime)",
        options.device
    );
    let prepared = QwenCuda::load_gguf(
        options.model,
        QwenLoadOptions {
            device: options.device,
            context_tokens,
            ..QwenLoadOptions::default()
        },
    )
    .map_err(display)?;
    let memory = prepared.memory_report();
    eprintln!(
        "ready: {} bytes reserved for sequence state; {} device bytes free after preparation",
        memory.reserved_sequence_bytes, memory.free_after_preparation_bytes
    );
    let mut engine = Engine::new(
        prepared,
        EngineConfig {
            max_active_requests: 1,
            max_queued_requests: 0,
            max_queued_input_tokens: u64::from(prompt_tokens),
            max_buffered_events: 16,
            max_events_per_request: 64,
        },
        SchedulePolicy::default(),
    )
    .map_err(display)?;
    engine
        .enqueue(TokenRequest::new(
            tokens,
            GenerationOptions {
                max_output_tokens: options.max_tokens,
                stop_tokens: vec![tokenizer.eos_token_id()],
                ..GenerationOptions::default()
            },
        ))
        .map_err(display)?;
    let result = generate(&mut engine, &tokenizer, &mut io::stdout().lock());
    // Broken pipes, output errors, and model errors still drain device work.
    let shutdown = engine.shutdown().map_err(display);
    match (result, shutdown) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => Err(format!("{error}; shutdown also failed: {cleanup}")),
    }
}

fn generate(
    engine: &mut Engine,
    tokenizer: &GgufTokenizer,
    output: &mut impl Write,
) -> Result<(), String> {
    loop {
        let status = engine.step().map_err(display)?;
        while let Some(event) = engine.pop_event() {
            match event {
                Event::Token { token, .. } => {
                    // A token may contain only part of a UTF-8 code point.
                    // Write its bytes unchanged; do not replace each fragment.
                    output
                        .write_all(&tokenizer.decode_bytes(&[token]).map_err(display)?)
                        .map_err(display)?;
                    output.flush().map_err(display)?;
                }
                Event::Finished { reason, .. } => {
                    return match reason {
                        FinishReason::Length | FinishReason::Stop => Ok(()),
                        FinishReason::Cancelled => Err("generation was cancelled".to_owned()),
                        FinishReason::Failed(error) => Err(display(error)),
                    };
                }
            }
        }
        if !status.submitted && !status.completed {
            std::thread::yield_now();
        }
    }
}

fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}
