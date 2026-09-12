use std::path::PathBuf;

const DEFAULT_MAX_TOKENS: u32 = 32;
const DEFAULT_CONTEXT_LENGTH: u32 = 4096;

#[derive(Debug)]
pub(crate) struct RunOptions {
    pub(crate) model: PathBuf,
    pub(crate) prompt: Option<String>,
    pub(crate) file: Option<PathBuf>,
    pub(crate) raw: bool,
    pub(crate) max_tokens: u32,
    pub(crate) context_length: u32,
    pub(crate) device: u16,
}

pub(crate) fn parse(arguments: &[String], usage: &str) -> Result<RunOptions, String> {
    let mut model = None;
    let mut prompt = None;
    let mut file = None;
    let mut raw = false;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut context_length = DEFAULT_CONTEXT_LENGTH;
    let mut device = 0_u16;
    let mut index = 0;
    while index < arguments.len() {
        let name = &arguments[index];
        if name == "--raw" {
            raw = true;
            index += 1;
            continue;
        }
        if !name.starts_with('-') && model.is_none() {
            model = Some(PathBuf::from(name));
            index += 1;
            continue;
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {name}; usage: {usage}"))?;
        match name.as_str() {
            "--model" => {
                if model.replace(PathBuf::from(value)).is_some() {
                    return Err("model may be specified only once".to_owned());
                }
            }
            "--prompt" => prompt = Some(value.clone()),
            "--file" => file = Some(PathBuf::from(value)),
            "--max-tokens" => {
                max_tokens = positive_u32(value, "--max-tokens")?;
            }
            "--context-length" => {
                context_length = positive_u32(value, "--context-length")?;
            }
            "--device" => {
                device = value
                    .parse::<u16>()
                    .map_err(|_| "--device expects a non-negative device ordinal".to_owned())?;
            }
            other => return Err(format!("unknown local option {other:?}; usage: {usage}")),
        }
        index += 2;
    }
    if prompt.is_some() && file.is_some() {
        return Err("--prompt and --file are mutually exclusive".to_owned());
    }
    Ok(RunOptions {
        model: model.ok_or_else(|| format!("model is required; usage: {usage}"))?,
        prompt,
        file,
        raw,
        max_tokens,
        context_length,
        device,
    })
}

fn positive_u32(value: &str, option: &str) -> Result<u32, String> {
    let value = value
        .parse::<u32>()
        .map_err(|_| format!("{option} expects a positive integer"))?;
    if value == 0 {
        return Err(format!("{option} must be greater than zero"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_positional_model_and_explicit_raw_prompt() {
        let options = parse(
            &args(&["test.gguf", "--prompt", "--help", "--raw", "--device", "2"]),
            "usage",
        )
        .unwrap();
        assert_eq!(options.prompt.as_deref(), Some("--help"));
        assert_eq!(options.model, PathBuf::from("test.gguf"));
        assert!(options.raw);
        assert_eq!(options.max_tokens, 32);
        assert_eq!(options.context_length, 4096);
        assert_eq!(options.device, 2);
    }

    #[test]
    fn keeps_model_flag_compatible_and_accepts_file_input() {
        let options = parse(
            &args(&[
                "--model",
                "test.gguf",
                "--file",
                "prompt.txt",
                "--context-length",
                "8192",
            ]),
            "usage",
        )
        .unwrap();
        assert_eq!(options.file, Some(PathBuf::from("prompt.txt")));
        assert_eq!(options.context_length, 8192);
    }

    #[test]
    fn rejects_conflicts_and_invalid_values() {
        assert!(parse(&[], "usage").is_err());
        assert!(parse(&args(&["--model"]), "usage").is_err());
        assert!(
            parse(
                &args(&["m.gguf", "--prompt", "hi", "--file", "prompt.txt"]),
                "usage"
            )
            .is_err()
        );
        for option in [
            ["--max-tokens", "0"],
            ["--context-length", "0"],
            ["--device", "-1"],
            ["--unknown", "1"],
        ] {
            let mut arguments = args(&["m.gguf", "--prompt", "hi"]);
            arguments.extend(args(&option));
            assert!(parse(&arguments, "usage").is_err());
        }
    }
}
