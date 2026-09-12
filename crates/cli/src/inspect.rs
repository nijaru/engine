//! Artifact inspection uses the format reader, not model preparation or CUDA.
use std::path::Path;

use engine_gguf::{GgufFile, MetadataValue};

const USAGE: &str = "ribn inspect <model.gguf>";

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    match arguments {
        [help] if matches!(help.as_str(), "-h" | "--help") => {
            println!(
                "{USAGE}\nRead artifact metadata without loading weights or initializing a GPU."
            );
            Ok(())
        }
        [path] => {
            print!("{}", describe(Path::new(path))?);
            Ok(())
        }
        _ => Err(format!("expected one artifact path; usage: {USAGE}")),
    }
}

#[allow(
    clippy::unnecessary_debug_formatting,
    reason = "escape untrusted path characters in terminal output"
)]
fn describe(path: &Path) -> Result<String, String> {
    let file = GgufFile::open(path).map_err(|error| error.to_string())?;
    let architecture = file
        .metadata("general.architecture")
        .and_then(MetadataValue::as_str)
        .unwrap_or("unspecified");
    Ok(format!(
        "artifact: {path:?}\nformat: GGUF v{}\narchitecture: {architecture:?}\ntensors: {}\nexecution: not assessed by metadata inspection\n",
        file.version(),
        file.tensor_count(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspection_accepts_unknown_model_architecture_without_a_gpu() {
        let path = std::env::temp_dir().join(format!("ribn-inspect-{}.gguf", std::process::id()));
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.extend(1_u64.to_le_bytes());
        let key = "general.architecture";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(8_u32.to_le_bytes());
        let value = "new_model";
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        std::fs::write(&path, bytes).unwrap();
        let report = describe(&path).unwrap();
        assert!(report.contains("architecture: \"new_model\""));
        assert!(report.contains("execution: not assessed"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn inspection_requires_one_path() {
        assert!(run(&[]).is_err());
        assert!(run(&["a".into(), "b".into()]).is_err());
    }
}
