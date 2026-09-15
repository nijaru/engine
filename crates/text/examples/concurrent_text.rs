//! Concurrent owned text generation: several callers, one execution owner.
//!
//! Demonstrates the shape the server frontends will use — cloneable handles,
//! independent streams, an abandoned request, and one explicit shutdown owner.
//! Requires a CUDA device and the pinned GGUF.
//!
//! ```text
//! ENGINE_QWEN_GGUF=/path/to/model.gguf \
//! cargo run -p ribn-text --features cuda --example concurrent_text -- \
//!   --prompt 'Explain a mutex.' --tokens 64 --callers 4
//! ```

use std::time::Instant;

use ribn::GenerationOptions;
use ribn_text::{LoadOptions, TextEvent, TextInput, TextOwner};

struct Arguments {
    model: String,
    prompt: String,
    tokens: u32,
    callers: usize,
    sequences: Option<usize>,
    abandon: bool,
}

fn usage() -> String {
    "concurrent_text [model.gguf] [--prompt <text>] [--tokens <n>] [--callers <n>] \
     [--sequences <n>] [--no-abandon]"
        .to_owned()
}

fn parse() -> Result<Arguments, String> {
    let mut model = std::env::var("ENGINE_QWEN_GGUF")
        .or_else(|_| std::env::var("RIBN_MODEL"))
        .unwrap_or_default();
    let mut prompt = "Explain a mutex.".to_owned();
    let mut tokens = 64_u32;
    let mut callers = 4_usize;
    let mut sequences = None;
    let mut abandon = true;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--prompt" => {
                prompt = arguments.next().ok_or_else(usage)?;
            }
            "--tokens" => {
                tokens = arguments
                    .next()
                    .ok_or_else(usage)?
                    .parse()
                    .map_err(|_| usage())?;
            }
            "--callers" => {
                callers = arguments
                    .next()
                    .ok_or_else(usage)?
                    .parse()
                    .map_err(|_| usage())?;
            }
            "--sequences" => {
                sequences = Some(
                    arguments
                        .next()
                        .ok_or_else(usage)?
                        .parse()
                        .map_err(|_| usage())?,
                );
            }
            "--no-abandon" => abandon = false,
            _ if argument.starts_with("--") => return Err(usage()),
            other => other.clone_into(&mut model),
        }
    }
    if model.is_empty() {
        return Err(format!(
            "no model given; pass a path or set ENGINE_QWEN_GGUF\n{}",
            usage()
        ));
    }
    if callers == 0 || tokens == 0 {
        return Err(usage());
    }
    Ok(Arguments {
        model,
        prompt,
        tokens,
        callers,
        sequences,
        abandon,
    })
}

/// Distinct, semantically different questions per caller. Identical output for
/// different questions would mean requests share each other's input or output.
const QUESTIONS: [&str; 6] = [
    "Name one primary color.",
    "Name one planet.",
    "Name one fruit.",
    "Name one country.",
    "Name one musical instrument.",
    "Name one number between one and nine.",
];

fn options(tokens: u32) -> GenerationOptions {
    GenerationOptions {
        max_output_tokens: tokens,
        ..GenerationOptions::default()
    }
}

fn main() -> Result<(), String> {
    let arguments = parse()?;
    println!("loading {}", arguments.model);
    let started = Instant::now();
    let load = LoadOptions {
        max_sequences: arguments
            .sequences
            .unwrap_or(LoadOptions::default().max_sequences),
        ..LoadOptions::default()
    };
    let (mut owner, memory) =
        TextOwner::load(&arguments.model, load).map_err(|error| format!("load failed: {error}"))?;
    println!(
        "ready in {:.1?}: {} bytes reserved for sequence state, {} device bytes free",
        started.elapsed(),
        memory.reserved_sequence_bytes,
        memory.free_after_preparation_bytes
    );
    let model = owner.model().clone();

    // One caller abandons a live stream. Dropping it must not disturb its peers,
    // because the execution owner retains retirement ownership.
    if arguments.abandon {
        let mut abandoned = model
            .stream_blocking(
                TextInput::prompt(&arguments.prompt),
                options(arguments.tokens),
            )
            .map_err(|error| format!("abandoned stream failed to start: {error}"))?;
        if let Some(event) = abandoned.next() {
            match event.map_err(|error| error.to_string())? {
                TextEvent::Delta { text, .. } => print!("[abandoned after {text:?}] "),
                TextEvent::Finished { .. } => {}
            }
        }
        drop(abandoned);
    }

    let mut handles = Vec::with_capacity(arguments.callers);
    for caller in 0..arguments.callers {
        let model = model.clone();
        let question = QUESTIONS[caller % QUESTIONS.len()];
        let prompt = format!("{} {}", question, arguments.prompt);
        let options = options(arguments.tokens);
        handles.push(std::thread::spawn(move || {
            let started = Instant::now();
            let response = model
                .generate_blocking(TextInput::prompt(prompt), options)
                .map_err(|error| error.to_string())?;
            Ok::<_, String>((caller, question, response, started.elapsed()))
        }));
    }

    for handle in handles {
        let (caller, question, response, elapsed) = handle
            .join()
            .map_err(|_| "caller thread panicked".to_owned())?
            .map_err(|error| format!("caller failed: {error}"))?;
        println!(
            "caller {caller} ({question}): {} tokens in {:.1?} ({:?})",
            response.tokens.len(),
            elapsed,
            response.reason
        );
        println!("  {}", response.text.replace('\n', " "));
    }

    // An owner failure is reported by the explicit shutdown, not by Drop.
    owner
        .shutdown()
        .map_err(|error| format!("shutdown failed: {error}"))?;
    println!("shutdown clean after {:.1?}", started.elapsed());
    Ok(())
}
