use std::path::PathBuf;

const DEFAULT_MAX_TOKENS: u32 = 32;

#[derive(Debug)]
pub(crate) struct RunOptions {
    pub(crate) model: PathBuf,
    pub(crate) prompt: String,
    pub(crate) max_tokens: u32,
    pub(crate) device: u16,
}

pub(crate) fn parse(arguments: &[String], usage: &str) -> Result<RunOptions, String> {
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut device = 0_u16;
    let mut index = 0;
    while index < arguments.len() {
        let name = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {name}; usage: {usage}"))?;
        match name.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value.clone()),
            "--max-tokens" => {
                max_tokens = value
                    .parse::<u32>()
                    .map_err(|_| "--max-tokens expects a positive integer".to_owned())?;
                if max_tokens == 0 {
                    return Err("--max-tokens must be greater than zero".to_owned());
                }
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
    Ok(RunOptions {
        model: model.ok_or_else(|| format!("--model is required; usage: {usage}"))?,
        prompt: prompt.ok_or_else(|| format!("--prompt is required; usage: {usage}"))?,
        max_tokens,
        device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_a_literal_prompt_and_shared_options() {
        let options = parse(
            &args(&[
                "--model",
                "test.gguf",
                "--prompt",
                "--help",
                "--device",
                "2",
            ]),
            "usage",
        )
        .unwrap();
        assert_eq!(options.prompt, "--help");
        assert_eq!(options.model, PathBuf::from("test.gguf"));
        assert_eq!(options.max_tokens, 32);
        assert_eq!(options.device, 2);
    }

    #[test]
    fn rejects_missing_and_invalid_values() {
        assert!(parse(&[], "usage").is_err());
        assert!(parse(&args(&["--model"]), "usage").is_err());
        for option in [
            ["--max-tokens", "0"],
            ["--device", "-1"],
            ["--unknown", "1"],
        ] {
            let mut arguments = args(&["--model", "m.gguf", "--prompt", "hi"]);
            arguments.extend(args(&option));
            assert!(parse(&arguments, "usage").is_err());
        }
    }
}
