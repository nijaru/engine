//! Bounded UTF-8 ingestion before model loading. Argument storage is caller-owned;
//! file/stdin reads consume at most the payload limit plus one overflow-probe byte.

use std::fs::File;
use std::io::{self, IsTerminal, Read};

use crate::cli::RunOptions;

pub(crate) fn read(options: &mut RunOptions, max_bytes: usize) -> Result<String, String> {
    let stdin = io::stdin();
    read_with_stdin(options, max_bytes, stdin.is_terminal(), stdin.lock())
}

fn read_with_stdin(
    options: &mut RunOptions,
    max_bytes: usize,
    terminal: bool,
    stdin: impl Read,
) -> Result<String, String> {
    if let Some(prompt) = options.prompt.take() {
        if prompt.len() > max_bytes {
            return Err(limit_error(max_bytes));
        }
        return Ok(prompt);
    }
    if let Some(path) = &options.file {
        let file = File::open(path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        return read_bounded(file, max_bytes)
            .map_err(|error| format!("failed to read {}: {error}", path.display()));
    }
    if terminal {
        return Err(
            "no input supplied; use --prompt, --file, or pipe text on stdin (interactive mode is not implemented yet)"
                .to_owned(),
        );
    }
    let text =
        read_bounded(stdin, max_bytes).map_err(|error| format!("failed to read stdin: {error}"))?;
    if text.is_empty() {
        return Err("stdin contained no input".to_owned());
    }
    Ok(text)
}

fn read_bounded(reader: impl Read, max_bytes: usize) -> Result<String, String> {
    let read_limit = u64::try_from(max_bytes)
        .ok()
        .and_then(|limit| limit.checked_add(1))
        .ok_or_else(|| "input byte limit is too large".to_owned())?;
    let mut bytes = Vec::new();
    reader
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    // Test size before UTF-8: the overflow probe may split a code point.
    if bytes.len() > max_bytes {
        return Err(limit_error(max_bytes));
    }
    String::from_utf8(bytes).map_err(|error| format!("input is not valid UTF-8: {error}"))
}

fn limit_error(max_bytes: usize) -> String {
    format!("input exceeds the {max_bytes}-byte text limit")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn options(extra: &[&str]) -> RunOptions {
        let args: Vec<_> = std::iter::once("unused.gguf")
            .chain(extra.iter().copied())
            .map(str::to_owned)
            .collect();
        crate::cli::parse(&args, "usage").unwrap()
    }

    #[test]
    fn accepts_exact_byte_limit_and_preserves_utf8() {
        assert_eq!(read_bounded("é!".as_bytes(), 3).unwrap(), "é!");
        assert_eq!(read_bounded(io::empty(), 0).unwrap(), "");
        assert!(
            read_bounded("é".as_bytes(), 1)
                .unwrap_err()
                .contains("limit")
        );
        assert!(read_bounded(&[0xff][..], 1).unwrap_err().contains("UTF-8"));
    }

    #[test]
    fn reads_only_one_probe_byte_past_the_limit() {
        let mut endless = io::repeat(b'x');
        assert!(
            read_bounded(&mut endless, 8)
                .unwrap_err()
                .contains("8-byte")
        );
        let mut source = Cursor::new(vec![b'x'; 100]);
        assert!(read_bounded(&mut source, 8).is_err());
        assert_eq!(source.position(), 9);
    }

    #[test]
    fn propagates_read_errors() {
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("read failed"))
            }
        }
        assert!(read_bounded(Broken, 8).unwrap_err().contains("read failed"));
    }

    #[test]
    fn prompt_is_moved_and_checked_without_reading_stdin() {
        let mut opts = options(&["--prompt", "é!"]);
        assert_eq!(
            read_with_stdin(&mut opts, 3, true, io::empty()).unwrap(),
            "é!"
        );
        assert!(opts.prompt.is_none());
        let error =
            read_with_stdin(&mut options(&["--prompt", "é!"]), 2, true, io::empty()).unwrap_err();
        assert!(error.contains("2-byte"));
    }

    #[test]
    fn file_input_uses_the_same_bound() {
        let path = std::env::temp_dir().join(format!("ribn-input-{}.txt", std::process::id()));
        std::fs::write(&path, b"text!").unwrap();
        let mut opts = options(&[]);
        opts.file = Some(path.clone());
        let accepted = read(&mut opts, 5);
        let rejected = read(&mut opts, 4);
        std::fs::remove_file(path).unwrap();
        assert_eq!(accepted.unwrap(), "text!");
        assert!(rejected.unwrap_err().contains("4-byte"));
    }

    #[test]
    fn stdin_rejects_empty_terminal_and_oversized_input() {
        assert!(
            read_with_stdin(&mut options(&[]), 4, false, io::empty())
                .unwrap_err()
                .contains("no input")
        );
        assert!(
            read_with_stdin(&mut options(&[]), 4, true, io::empty())
                .unwrap_err()
                .contains("interactive")
        );
        assert!(
            read_with_stdin(&mut options(&[]), 4, false, io::repeat(b'x'))
                .unwrap_err()
                .contains("4-byte")
        );
        assert_eq!(
            read_with_stdin(&mut options(&[]), 4, false, &b"text"[..]).unwrap(),
            "text"
        );
    }
}
