#[cfg(any(feature = "cuda", test))]
mod cli;
#[cfg(feature = "cuda")]
mod local;
#[cfg(feature = "cuda")]
mod run;

const USAGE: &str = "engine-server <command>\n\ncommands:\n  run      stream one Qwen GGUF request through the experimental Ribn runtime\n  local    legacy Qwen CUDA correctness frontend";

fn main() {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next();
    let rest = arguments.collect::<Vec<_>>();
    let result = match command.as_deref() {
        None | Some("-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("local") => run_local(&rest),
        Some("run") => run_prepared(&rest),
        Some(other) => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };
    if let Err(error) = result {
        eprintln!("engine-server: {error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "cuda")]
fn run_local(arguments: &[String]) -> Result<(), String> {
    local::run(arguments)
}

#[cfg(not(feature = "cuda"))]
fn run_local(_arguments: &[String]) -> Result<(), String> {
    Err("local CUDA inference requires building engine-server with --features cuda".to_owned())
}

#[cfg(feature = "cuda")]
fn run_prepared(arguments: &[String]) -> Result<(), String> {
    run::run(arguments)
}

#[cfg(not(feature = "cuda"))]
fn run_prepared(_arguments: &[String]) -> Result<(), String> {
    Err("Qwen inference requires building engine-server with --features cuda".to_owned())
}
