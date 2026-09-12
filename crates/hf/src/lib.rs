//! Local Hugging Face-style model-package resolver used for architecture pressure tests.
//!
//! A package owns model/package metadata and resolves weight files. It does not
//! decide model semantics, execution backend, placement, scheduling, tokenizer
//! implementation, or processor behavior. Network download/cache/auth are also
//! intentionally outside this first local-directory pressure test.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

use ribn_safetensors::SafeTensorArtifact;
use serde_json::Value;

const CONFIG_FILE: &str = "config.json";
const SINGLE_WEIGHTS: &str = "model.safetensors";
const WEIGHT_INDEX: &str = "model.safetensors.index.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WeightLayout {
    Single {
        path: PathBuf,
    },
    Sharded {
        index_path: PathBuf,
        shards: Vec<PathBuf>,
        parameter_count: usize,
    },
}

/// Resolved local package. The JSON config is preserved for a model integration
/// to interpret; this layer does not turn architecture names into runtime types.
pub struct LocalModelPackage {
    root: PathBuf,
    config_path: PathBuf,
    config: Value,
    weights: Weights,
}

enum Weights {
    Single(PathBuf),
    Sharded {
        index_path: PathBuf,
        weight_map: BTreeMap<String, PathBuf>,
        shards: Vec<PathBuf>,
    },
}

impl LocalModelPackage {
    /// Resolve a local Hugging Face-style model directory.
    ///
    /// `config.json` is required. Weights may be one `model.safetensors` file or
    /// a `model.safetensors.index.json` whose `weight_map` names local shards.
    /// When both weight forms exist, the explicit shard index wins.
    ///
    /// # Errors
    /// Returns a structured package error for missing/invalid config, unsupported
    /// weight layout, malformed shard index, missing shards, or shard paths that
    /// escape the package root.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, PackageError> {
        let root = root.into();
        let canonical_root = fs::canonicalize(&root).map_err(|error| PackageError::Io {
            path: root.clone(),
            message: error.to_string(),
        })?;
        if !canonical_root.is_dir() {
            return Err(PackageError::NotDirectory(canonical_root));
        }

        let config_path = canonical_root.join(CONFIG_FILE);
        let config_bytes = fs::read(&config_path).map_err(|error| PackageError::Io {
            path: config_path.clone(),
            message: error.to_string(),
        })?;
        let config: Value = serde_json::from_slice(&config_bytes).map_err(|error| {
            PackageError::InvalidJson {
                path: config_path.clone(),
                message: error.to_string(),
            }
        })?;
        if !config.is_object() {
            return Err(PackageError::InvalidConfig(
                "config.json must contain a JSON object",
            ));
        }

        let index_path = canonical_root.join(WEIGHT_INDEX);
        let weights = if index_path.is_file() {
            Weights::from_index(&canonical_root, index_path)?
        } else {
            let weights_path = canonical_root.join(SINGLE_WEIGHTS);
            if !weights_path.is_file() {
                return Err(PackageError::MissingWeights(canonical_root));
            }
            Weights::Single(weights_path)
        };

        Ok(Self {
            root: canonical_root,
            config_path,
            config,
            weights,
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    #[must_use]
    pub const fn config(&self) -> &Value {
        &self.config
    }

    #[must_use]
    pub fn weight_layout(&self) -> WeightLayout {
        match &self.weights {
            Weights::Single(path) => WeightLayout::Single { path: path.clone() },
            Weights::Sharded {
                index_path,
                weight_map,
                shards,
            } => WeightLayout::Sharded {
                index_path: index_path.clone(),
                shards: shards.clone(),
                parameter_count: weight_map.len(),
            },
        }
    }

    /// Resolve the shard that may contain one logical checkpoint parameter.
    /// For an unsharded package every name maps to the single weight artifact;
    /// actual tensor-name validation remains a SafeTensors/model concern.
    #[must_use]
    pub fn weight_file(&self, parameter: &str) -> Option<&Path> {
        match &self.weights {
            Weights::Single(path) => Some(path),
            Weights::Sharded { weight_map, .. } => {
                weight_map.get(parameter).map(PathBuf::as_path)
            }
        }
    }

    /// Open the SafeTensors shard associated with `parameter`.
    ///
    /// # Errors
    /// Returns [`PackageError::UnknownParameter`] for a name absent from a
    /// sharded index, or an artifact error when the resolved shard is invalid.
    pub fn open_weights_for(
        &self,
        parameter: &str,
    ) -> Result<SafeTensorArtifact, PackageError> {
        let path = self
            .weight_file(parameter)
            .ok_or_else(|| PackageError::UnknownParameter(parameter.to_owned()))?;
        SafeTensorArtifact::open(path).map_err(|error| PackageError::Artifact {
            path: path.to_owned(),
            message: error.to_string(),
        })
    }

    /// Resolve another file within the package root without assigning semantics
    /// to tokenizer, processor, generation, chat-template, or modality metadata.
    ///
    /// # Errors
    /// Rejects absolute/parent paths and symlink escapes. A missing file returns
    /// [`PackageError::MissingPackageFile`].
    pub fn package_file(&self, relative: impl AsRef<Path>) -> Result<PathBuf, PackageError> {
        resolve_member(&self.root, relative.as_ref(), false)
    }
}

impl Weights {
    fn from_index(root: &Path, index_path: PathBuf) -> Result<Self, PackageError> {
        let bytes = fs::read(&index_path).map_err(|error| PackageError::Io {
            path: index_path.clone(),
            message: error.to_string(),
        })?;
        let index: Value = serde_json::from_slice(&bytes).map_err(|error| {
            PackageError::InvalidJson {
                path: index_path.clone(),
                message: error.to_string(),
            }
        })?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or(PackageError::InvalidWeightIndex(
                "weight index must contain an object-valued weight_map",
            ))?;
        if weight_map.is_empty() {
            return Err(PackageError::InvalidWeightIndex(
                "weight_map must contain at least one parameter",
            ));
        }

        let mut resolved = BTreeMap::new();
        let mut shards = BTreeSet::new();
        for (parameter, value) in weight_map {
            if parameter.trim().is_empty() {
                return Err(PackageError::InvalidWeightIndex(
                    "weight_map contains an empty parameter name",
                ));
            }
            let shard = value.as_str().ok_or(PackageError::InvalidWeightIndex(
                "weight_map shard values must be strings",
            ))?;
            let path = resolve_member(root, Path::new(shard), true)?;
            shards.insert(path.clone());
            resolved.insert(parameter.clone(), path);
        }
        Ok(Self::Sharded {
            index_path,
            weight_map: resolved,
            shards: shards.into_iter().collect(),
        })
    }
}

fn resolve_member(root: &Path, relative: &Path, shard: bool) -> Result<PathBuf, PackageError> {
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(if shard {
            PackageError::UnsafeShardPath(relative.to_owned())
        } else {
            PackageError::UnsafePackagePath(relative.to_owned())
        });
    }
    let joined = root.join(relative);
    let canonical = fs::canonicalize(&joined).map_err(|error| {
        if shard {
            PackageError::MissingShard {
                path: joined.clone(),
                message: error.to_string(),
            }
        } else {
            PackageError::MissingPackageFile {
                path: joined.clone(),
                message: error.to_string(),
            }
        }
    })?;
    if !canonical.starts_with(root) {
        return Err(if shard {
            PackageError::UnsafeShardPath(relative.to_owned())
        } else {
            PackageError::UnsafePackagePath(relative.to_owned())
        });
    }
    if !canonical.is_file() {
        return Err(if shard {
            PackageError::MissingShard {
                path: canonical,
                message: "not a regular file".to_owned(),
            }
        } else {
            PackageError::MissingPackageFile {
                path: canonical,
                message: "not a regular file".to_owned(),
            }
        });
    }
    Ok(canonical)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PackageError {
    Io { path: PathBuf, message: String },
    NotDirectory(PathBuf),
    InvalidJson { path: PathBuf, message: String },
    InvalidConfig(&'static str),
    MissingWeights(PathBuf),
    InvalidWeightIndex(&'static str),
    MissingShard { path: PathBuf, message: String },
    UnsafeShardPath(PathBuf),
    MissingPackageFile { path: PathBuf, message: String },
    UnsafePackagePath(PathBuf),
    UnknownParameter(String),
    Artifact { path: PathBuf, message: String },
}

impl fmt::Display for PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "could not read {}: {message}", path.display()),
            Self::NotDirectory(path) => write!(f, "{} is not a model-package directory", path.display()),
            Self::InvalidJson { path, message } => {
                write!(f, "invalid JSON in {}: {message}", path.display())
            }
            Self::InvalidConfig(message) => write!(f, "invalid model config: {message}"),
            Self::MissingWeights(root) => write!(
                f,
                "model package {} has neither model.safetensors nor model.safetensors.index.json",
                root.display()
            ),
            Self::InvalidWeightIndex(message) => write!(f, "invalid SafeTensors weight index: {message}"),
            Self::MissingShard { path, message } => {
                write!(f, "model weight shard {} is unavailable: {message}", path.display())
            }
            Self::UnsafeShardPath(path) => {
                write!(f, "weight index shard path {:?} escapes the model package", path)
            }
            Self::MissingPackageFile { path, message } => {
                write!(f, "model package file {} is unavailable: {message}", path.display())
            }
            Self::UnsafePackagePath(path) => {
                write!(f, "package path {:?} escapes the model package", path)
            }
            Self::UnknownParameter(name) => {
                write!(f, "weight index has no parameter {name:?}")
            }
            Self::Artifact { path, message } => {
                write!(f, "invalid weight artifact {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for PackageError {}

#[cfg(test)]
mod tests {
    use std::env;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_DIR: AtomicU64 = AtomicU64::new(1);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!("ribn-hf-{}-{id}", process::id()));
            fs::create_dir(&path).expect("test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn safetensors_fixture(name: &str, values: &[f32], shape: &[usize]) -> Vec<u8> {
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[0,{}]}}}}"#,
            data.len()
        )
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut artifact = Vec::new();
        artifact.extend_from_slice(
            &u64::try_from(header.len())
                .expect("fixture header length")
                .to_le_bytes(),
        );
        artifact.extend_from_slice(&header);
        artifact.extend_from_slice(&data);
        artifact
    }

    fn write_config(root: &Path) {
        fs::write(
            root.join(CONFIG_FILE),
            br#"{"architectures":["FixtureEncoder"],"hidden_size":3}"#,
        )
        .expect("config");
    }

    #[test]
    fn resolves_unsharded_package_without_interpreting_model_semantics() {
        let dir = TestDir::new();
        write_config(dir.path());
        fs::write(
            dir.path().join(SINGLE_WEIGHTS),
            safetensors_fixture("embeddings.weight", &[1.0, 2.0, 3.0], &[1, 3]),
        )
        .expect("weights");
        fs::write(dir.path().join("tokenizer.json"), b"{}\n").expect("tokenizer metadata");

        let package = LocalModelPackage::open(dir.path()).expect("package");
        assert_eq!(
            package.config()["architectures"][0],
            Value::String("FixtureEncoder".to_owned())
        );
        assert!(matches!(package.weight_layout(), WeightLayout::Single { .. }));
        let artifact = package
            .open_weights_for("embeddings.weight")
            .expect("artifact");
        assert_eq!(
            artifact.tensor("embeddings.weight").expect("tensor").shape(),
            [1, 3]
        );
        assert!(package.package_file("tokenizer.json").expect("tokenizer").is_file());
    }

    #[test]
    fn resolves_sharded_weight_index_and_parameter_ownership() {
        let dir = TestDir::new();
        write_config(dir.path());
        let first = "model-00001-of-00002.safetensors";
        let second = "model-00002-of-00002.safetensors";
        fs::write(
            dir.path().join(first),
            safetensors_fixture("embeddings.weight", &[1.0, 2.0, 3.0], &[1, 3]),
        )
        .expect("first shard");
        fs::write(
            dir.path().join(second),
            safetensors_fixture("projection.weight", &[4.0, 5.0, 6.0], &[1, 3]),
        )
        .expect("second shard");
        fs::write(
            dir.path().join(WEIGHT_INDEX),
            format!(
                r#"{{"metadata":{{"total_size":24}},"weight_map":{{"embeddings.weight":"{first}","projection.weight":"{second}"}}}}"#
            ),
        )
        .expect("index");

        let package = LocalModelPackage::open(dir.path()).expect("package");
        let WeightLayout::Sharded {
            shards,
            parameter_count,
            ..
        } = package.weight_layout()
        else {
            panic!("expected sharded weights");
        };
        assert_eq!(shards.len(), 2);
        assert_eq!(parameter_count, 2);
        assert_eq!(
            package
                .weight_file("projection.weight")
                .and_then(Path::file_name)
                .and_then(|name| name.to_str()),
            Some(second)
        );
        let artifact = package
            .open_weights_for("projection.weight")
            .expect("projection shard");
        assert!(artifact.tensor("projection.weight").is_ok());
        assert!(matches!(
            package.open_weights_for("missing.weight"),
            Err(PackageError::UnknownParameter(name)) if name == "missing.weight"
        ));
    }

    #[test]
    fn rejects_weight_index_paths_that_escape_package() {
        let dir = TestDir::new();
        write_config(dir.path());
        fs::write(
            dir.path().join(WEIGHT_INDEX),
            br#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
        )
        .expect("index");
        assert!(matches!(
            LocalModelPackage::open(dir.path()),
            Err(PackageError::UnsafeShardPath(path)) if path == PathBuf::from("../outside.safetensors")
        ));
    }
}
