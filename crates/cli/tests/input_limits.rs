//! Exercise rejection before model loading through the CUDA-enabled CLI, without a GPU.
#![cfg(feature = "cuda")]

use std::io::Write;
use std::process::{Command, Stdio};

const INPUT_BYTES: usize = 256 * 1024;

fn piped_input(raw: bool, bytes: &[u8]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ribn"));
    command.args(["run", "missing-input-limit-test.gguf"]);
    if raw {
        command.arg("--raw");
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn oversized_stdin_is_rejected_before_model_loading() {
    for (raw, limit) in [(true, INPUT_BYTES), (false, INPUT_BYTES - 4)] {
        let output = piped_input(raw, &vec![b'x'; limit + 1]);
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains(&format!("{limit}-byte text limit")),
            "{stderr}"
        );
        assert!(!stderr.contains("preparing model"), "{stderr}");
    }
}

#[test]
fn exact_limit_reaches_model_loading_in_raw_and_chat_modes() {
    for (raw, limit) in [(true, INPUT_BYTES), (false, INPUT_BYTES - 4)] {
        let output = piped_input(raw, &vec![b'x'; limit]);
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("preparing model"), "{stderr}");
        assert!(!stderr.contains("text limit"), "{stderr}");
    }
}
