use std::io::{self, Write};

use ribn_text::{
    FinishReason, GenerationOptions, LoadOptions, Message, TextEvent, TextInput, TextModel,
    TextOwner,
};

const USAGE: &str = "ribn run <model.gguf> [--prompt <text> | --file <path>] [--raw] [--max-tokens <n>] [--context-length <n>] [--device <ordinal>]";

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    if matches!(arguments, [help] if matches!(help.as_str(), "-h" | "--help")) {
        println!(
            "{USAGE}\n\nWithout --raw, text input is sent as one user chat message through the model's embedded chat template. If neither --prompt nor --file is given, piped stdin is used. Interactive terminal chat is not implemented yet.\n\nExperimental Qwen GGUF/CUDA path. Input uses the text processor's byte limit (including the user role in chat mode) and is checked before model loading."
        );
        return Ok(());
    }
    let mut options = crate::cli::parse(arguments, USAGE)?;
    let load = LoadOptions {
        device: options.device,
        context_tokens: options.context_length,
        ..LoadOptions::default()
    };
    let mut message = Message::user("");
    let role_bytes = if options.raw { 0 } else { message.role.len() };
    let max_text_bytes = load
        .limits
        .max_input_bytes
        .checked_sub(role_bytes)
        .ok_or_else(|| "input byte limit cannot hold the chat role".to_owned())?;
    let input_text = crate::input::read(&mut options, max_text_bytes)?;
    let input = if options.raw {
        TextInput::prompt(input_text)
    } else {
        message.content = input_text;
        TextInput::chat(vec![message])
    };

    eprintln!(
        "preparing model on CUDA device {} with {}-token context capacity (experimental Ribn runtime)",
        options.device, options.context_length
    );
    let (mut owner, memory) = TextOwner::load(options.model, load).map_err(display)?;
    eprintln!(
        "ready: {} bytes reserved for sequence state; {} device bytes free after preparation",
        memory.reserved_sequence_bytes, memory.free_after_preparation_bytes
    );

    let generation = GenerationOptions {
        max_output_tokens: options.max_tokens,
        ..GenerationOptions::default()
    };
    let result = stream(owner.model(), input, generation, &mut io::stdout().lock());
    let shutdown = owner.shutdown().map_err(display);
    match (result, shutdown) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => Err(format!("{error}; shutdown also failed: {cleanup}")),
    }
}

fn stream(
    model: &TextModel,
    input: TextInput,
    options: GenerationOptions,
    output: &mut impl Write,
) -> Result<(), String> {
    let events = model.stream_blocking(input, options).map_err(display)?;
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
