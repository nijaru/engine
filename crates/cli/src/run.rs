use std::fs;
use std::io::{self, IsTerminal, Read, Write};

use ribn_text::{
    FinishReason, GenerationOptions, LoadOptions, Message, TextEvent, TextInput, TextModel,
};

const USAGE: &str = "ribn run <model.gguf> [--prompt <text> | --file <path>] [--raw] [--max-tokens <n>] [--context-length <n>] [--device <ordinal>]";

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    if matches!(arguments, [help] if matches!(help.as_str(), "-h" | "--help")) {
        println!(
            "{USAGE}\n\nWithout --raw, text input is sent as one user chat message through the model's embedded chat template. If neither --prompt nor --file is given, piped stdin is used. Interactive terminal chat is not implemented yet.\n\nExperimental Qwen GGUF/CUDA path; GPU qualification is pending."
        );
        return Ok(());
    }
    let options = crate::cli::parse(arguments, USAGE)?;
    let input_text = read_input(&options)?;
    let input = if options.raw {
        TextInput::prompt(input_text)
    } else {
        TextInput::chat(vec![Message::user(input_text)])
    };

    eprintln!(
        "preparing model on CUDA device {} with {}-token context capacity (experimental Ribn runtime)",
        options.device, options.context_length
    );
    let mut model = TextModel::load(
        options.model,
        LoadOptions {
            device: options.device,
            context_tokens: options.context_length,
            ..LoadOptions::default()
        },
    )
    .map_err(display)?;
    let memory = model.memory_report();
    eprintln!(
        "ready: {} bytes reserved for sequence state; {} device bytes free after preparation",
        memory.reserved_sequence_bytes, memory.free_after_preparation_bytes
    );

    let generation = GenerationOptions {
        max_output_tokens: options.max_tokens,
        ..GenerationOptions::default()
    };
    let result = stream(&mut model, input, generation, &mut io::stdout().lock());
    let shutdown = model.shutdown().map_err(display);
    match (result, shutdown) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => Err(format!("{error}; shutdown also failed: {cleanup}")),
    }
}

fn read_input(options: &crate::cli::RunOptions) -> Result<String, String> {
    if let Some(prompt) = &options.prompt {
        return Ok(prompt.clone());
    }
    if let Some(path) = &options.file {
        return fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()));
    }
    let mut stdin = io::stdin();
    if stdin.is_terminal() {
        return Err(
            "no input supplied; use --prompt, --file, or pipe text on stdin (interactive mode is not implemented yet)"
                .to_owned(),
        );
    }
    let mut input = String::new();
    stdin.read_to_string(&mut input).map_err(display)?;
    if input.is_empty() {
        return Err("stdin contained no input".to_owned());
    }
    Ok(input)
}

fn stream(
    model: &mut TextModel,
    input: TextInput,
    options: GenerationOptions,
    output: &mut impl Write,
) -> Result<(), String> {
    let events = model.stream(input, options).map_err(display)?;
    for event in events {
        match event.map_err(display)? {
            TextEvent::Delta { text, .. } => {
                output.write_all(text.as_bytes()).map_err(display)?;
                output.flush().map_err(display)?;
            }
            TextEvent::Finished { reason, .. } => {
                return match reason {
                    FinishReason::Length | FinishReason::Stop => Ok(()),
                    FinishReason::Cancelled => Err("generation was cancelled".to_owned()),
                    FinishReason::Failed(error) => Err(display(error)),
                };
            }
        }
    }
    Err("generation ended without a terminal event".to_owned())
}

fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}
