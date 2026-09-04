#[cfg(feature = "cuda")]
mod local;

const USAGE: &str =
    "engine-server <command>\n\ncommands:\n  local    run one local Qwen3.8 GGUF request on CUDA";

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
