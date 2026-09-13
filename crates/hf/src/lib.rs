//! Local Hugging Face-style model-package resolver used for architecture pressure tests.
//!
//! A package owns model/package metadata and resolves weight files. It does not
//! decide model semantics, execution backend, placement, scheduling, tokenizer
//! implementation, or processor behavior. Network download/cache/auth are also
//! intentionally outside this first local-directory pressure test.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

use ribn_safetensors::{ArtifactTensor, SafeTensorArtifact};
use serde_json::Value;

const CONFIG_FILE: &str = "config.json";
const SINGLE_WEIGHTS: &str = "model.safetensors";
const WEIGHT_INDEX: &str = "model.safetensors.index.json";
const SNAPSHOTS_DIR: &str = "snapshots";
const BLOBS_DIR: &str = "blobs";
const REPOSITORY_PREFIX: &str = "models--";

/// Where a local package's members may be resolved from.
///
/// A plain package directory confines every member to that directory. A Hugging
/// Face cache snapshot directory additionally authorizes that repository's
/// immutable `blobs` directory, because a cache snapshot is a symlink farm whose
/// entries point at content-addressed blobs beside the `snapshots` directory.
/// Nothing else outside the root is ever authorized, and the trusted shape is
/// narrow on purpose: the root must be `<repository>/snapshots/<revision>` under a
/// `models--` repository directory that contains a `blobs` directory.
///
/// Only the directories are trusted here, not the file names. A member still has
/// to be reached through a relative path inside the package root and to name an
/// existing regular file, so a snapshot cannot redirect the loader at arbitrary
/// host files.
#[derive(Clone, Debug, Eq, PartialEq)]
enum MemberPolicy {
    Directory(PathBuf),
    Snapshot { snapshot: PathBuf, blobs: PathBuf },
}

impl MemberPolicy {
    fn classify(root: PathBuf) -> Self {
        let trusted_blobs = root
            .parent()
            .filter(|snapshots| snapshots.file_name() == Some(OsStr::new(SNAPSHOTS_DIR)))
            .and_then(Path::parent)
            .filter(|repository| {
                repository
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.starts_with(REPOSITORY_PREFIX))
            })
            .map(|repository| repository.join(BLOBS_DIR))
            .filter(|blobs| blobs.is_dir());
        match trusted_blobs {
            Some(blobs) => Self::Snapshot {
                snapshot: root,
                blobs,
            },
            None => Self::Directory(root),
        }
    }

    fn root(&self) -> &Path {
        match self {
            Self::Directory(root) => root,
            Self::Snapshot { snapshot, .. } => snapshot,
        }
    }

    fn allows(&self, canonical: &Path) -> bool {
        match self {
            Self::Directory(root) => canonical.starts_with(root),
            Self::Snapshot { snapshot, blobs } => {
                canonical.starts_with(snapshot) || canonical.starts_with(blobs)
            }
        }
    }
}

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
    members: MemberPolicy,
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

/// Lazy reusable view over the `SafeTensors` files resolved by one local package.
///
/// The first tensor requested from a shard opens and validates that artifact; later
/// tensors from the same shard reuse the owned bytes. The set still does not assign
/// model semantics to parameter names or materialize tensors for a backend.
pub struct LocalWeightSet<'a> {
    weights: &'a Weights,
    artifacts: BTreeMap<PathBuf, SafeTensorArtifact>,
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
        let members = MemberPolicy::classify(canonical_root.clone());

        let config_path = resolve_member(&members, Path::new(CONFIG_FILE), false)?;
        let config_bytes = fs::read(&config_path).map_err(|error| PackageError::Io {
            path: config_path.clone(),
            message: error.to_string(),
        })?;
        let config: Value =
            serde_json::from_slice(&config_bytes).map_err(|error| PackageError::InvalidJson {
                path: config_path.clone(),
                message: error.to_string(),
            })?;
        if !config.is_object() {
            return Err(PackageError::InvalidConfig(
                "config.json must contain a JSON object",
            ));
        }

        let weights = match optional_member(&members, Path::new(WEIGHT_INDEX), true)? {
            Some(index_path) => Weights::from_index(&members, index_path)?,
            None => match optional_member(&members, Path::new(SINGLE_WEIGHTS), true)? {
                Some(path) => Weights::Single(path),
                None => return Err(PackageError::MissingWeights(canonical_root)),
            },
        };

        Ok(Self {
            root: canonical_root,
            members,
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
            Weights::Sharded { weight_map, .. } => weight_map.get(parameter).map(PathBuf::as_path),
        }
    }

    /// Create a lazy reusable weight-set view.
    ///
    /// Prefer this for loading multiple parameters. Each unique shard is opened at
    /// most once for the lifetime of the returned set, while unused shards stay
    /// unopened.
    #[must_use]
    pub fn weight_set(&self) -> LocalWeightSet<'_> {
        LocalWeightSet {
            weights: &self.weights,
            artifacts: BTreeMap::new(),
        }
    }

    /// Open the `SafeTensors` shard associated with one `parameter`.
    ///
    /// This one-shot helper is useful for inspection. Model loaders reading many
    /// tensors should use [`Self::weight_set`] so repeated parameters in one shard
    /// do not reread the whole artifact.
    ///
    /// # Errors
    /// Returns [`PackageError::UnknownParameter`] for a name absent from a
    /// sharded index, or an artifact error when the resolved shard is invalid.
    pub fn open_weights_for(&self, parameter: &str) -> Result<SafeTensorArtifact, PackageError> {
        let path = self
            .weight_file(parameter)
            .ok_or_else(|| PackageError::UnknownParameter(parameter.to_owned()))?;
        open_artifact(path)
    }

    /// Resolve another file within the package without assigning semantics
    /// to tokenizer, processor, generation, chat-template, or modality metadata.
    ///
    /// # Errors
    /// Rejects absolute/parent paths and symlink escapes. A missing file returns
    /// [`PackageError::MissingPackageFile`].
    pub fn package_file(&self, relative: impl AsRef<Path>) -> Result<PathBuf, PackageError> {
        resolve_member(&self.members, relative.as_ref(), false)
    }
}

impl LocalWeightSet<'_> {
    /// Number of unique `SafeTensors` shards opened so far.
    #[must_use]
    pub fn opened_shard_count(&self) -> usize {
        self.artifacts.len()
    }

    /// Resolve and borrow the `SafeTensors` artifact containing one parameter.
    ///
    /// This exposes format-level access for model integrations that need to try
    /// architecture-specific parameter aliases while keeping shard ownership and
    /// reuse in the package layer.
    ///
    /// # Errors
    /// Returns [`PackageError::UnknownParameter`] for a name absent from a sharded
    /// index, or an artifact error when the resolved shard is invalid.
    pub fn artifact(&mut self, parameter: &str) -> Result<&SafeTensorArtifact, PackageError> {
        let path = self
            .weight_path(parameter)
            .ok_or_else(|| PackageError::UnknownParameter(parameter.to_owned()))?
            .to_owned();
        match self.artifacts.entry(path.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
            std::collections::btree_map::Entry::Vacant(entry) => {
                Ok(entry.insert(open_artifact(&path)?))
            }
        }
    }

    /// Resolve and borrow one parameter tensor, lazily opening its shard once.
    ///
    /// Model integrations remain responsible for interpreting the parameter name,
    /// validating model-specific shape requirements, and preparing backend storage.
    ///
    /// # Errors
    /// Returns [`PackageError::UnknownParameter`] for a name absent from a sharded
    /// index, or an artifact error when the shard/tensor is invalid.
    pub fn tensor(&mut self, parameter: &str) -> Result<ArtifactTensor<'_>, PackageError> {
        let path = self
            .weight_path(parameter)
            .ok_or_else(|| PackageError::UnknownParameter(parameter.to_owned()))?
            .to_owned();
        self.artifact(parameter)?
            .tensor(parameter)
            .map_err(|error| PackageError::Artifact {
                path,
                message: error.to_string(),
            })
    }

    fn weight_path(&self, parameter: &str) -> Option<&Path> {
        match self.weights {
            Weights::Single(path) => Some(path),
            Weights::Sharded { weight_map, .. } => weight_map.get(parameter).map(PathBuf::as_path),
        }
    }
}

fn open_artifact(path: &Path) -> Result<SafeTensorArtifact, PackageError> {
    SafeTensorArtifact::open(path).map_err(|error| PackageError::Artifact {
        path: path.to_owned(),
        message: error.to_string(),
    })
}

impl Weights {
    fn from_index(members: &MemberPolicy, index_path: PathBuf) -> Result<Self, PackageError> {
        let bytes = fs::read(&index_path).map_err(|error| PackageError::Io {
            path: index_path.clone(),
            message: error.to_string(),
        })?;
        let index: Value =
            serde_json::from_slice(&bytes).map_err(|error| PackageError::InvalidJson {
                path: index_path.clone(),
                message: error.to_string(),
            })?;
        let weight_map = index.get("weight_map").and_then(Value::as_object).ok_or(
            PackageError::InvalidWeightIndex(
                "weight index must contain an object-valued weight_map",
            ),
        )?;
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
            let path = resolve_member(members, Path::new(shard), true)?;
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

/// Resolve a member path, rejecting escapes and non-files.
fn resolve_member(
    members: &MemberPolicy,
    relative: &Path,
    shard: bool,
) -> Result<PathBuf, PackageError> {
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(unsafe_member_path(relative, shard));
    }
    let joined = members.root().join(relative);
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
    if !members.allows(&canonical) {
        return Err(unsafe_member_path(relative, shard));
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

/// Resolve a member that may legitimately be absent.
fn optional_member(
    members: &MemberPolicy,
    relative: &Path,
    shard: bool,
) -> Result<Option<PathBuf>, PackageError> {
    match resolve_member(members, relative, shard) {
        Ok(path) => Ok(Some(path)),
        Err(PackageError::MissingShard { .. } | PackageError::MissingPackageFile { .. }) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn unsafe_member_path(relative: &Path, shard: bool) -> PackageError {
    if shard {
        PackageError::UnsafeShardPath(relative.to_owned())
    } else {
        PackageError::UnsafePackagePath(relative.to_owned())
    }
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
            Self::NotDirectory(path) => {
                write!(f, "{} is not a model-package directory", path.display())
            }
            Self::InvalidJson { path, message } => {
                write!(f, "invalid JSON in {}: {message}", path.display())
            }
            Self::InvalidConfig(message) => write!(f, "invalid model config: {message}"),
            Self::MissingWeights(root) => write!(
                f,
                "model package {} has neither model.safetensors nor model.safetensors.index.json",
                root.display()
            ),
            Self::InvalidWeightIndex(message) => {
                write!(f, "invalid SafeTensors weight index: {message}")
            }
            Self::MissingShard { path, message } => {
                write!(
                    f,
                    "model weight shard {} is unavailable: {message}",
                    path.display()
                )
            }
            Self::UnsafeShardPath(path) => write!(
                f,
                "weight index shard path {} escapes the model package",
                path.display()
            ),
            Self::MissingPackageFile { path, message } => {
                write!(
                    f,
                    "model package file {} is unavailable: {message}",
                    path.display()
                )
            }
            Self::UnsafePackagePath(path) => {
                write!(
                    f,
                    "package path {} escapes the model package",
                    path.display()
                )
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
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    const CONFIG_BYTES: &[u8] = br#"{"architectures":["FixtureEncoder"],"hidden_size":3}"#;

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
        fs::write(root.join(CONFIG_FILE), CONFIG_BYTES).expect("config");
    }

    /// Build a Hugging Face cache repository the way `huggingface_hub` lays it out:
    /// content-addressed blobs beside a snapshot directory of symlinks.
    ///
    /// Each member is `(blob name, snapshot name, bytes)`. Returns the snapshot a
    /// caller would pass to [`LocalModelPackage::open`] and the canonical blobs
    /// directory that its members are expected to resolve into.
    #[cfg(unix)]
    fn cache_repository(
        root: &Path,
        revision: &str,
        members: &[(&str, &str, Vec<u8>)],
    ) -> (PathBuf, PathBuf) {
        let repository = root.join("models--acme--fixture");
        let blobs = repository.join(BLOBS_DIR);
        let snapshot = repository.join(SNAPSHOTS_DIR).join(revision);
        fs::create_dir_all(&blobs).expect("blobs directory");
        fs::create_dir_all(&snapshot).expect("snapshot directory");
        for (blob, name, bytes) in members {
            fs::write(blobs.join(blob), bytes).expect("blob");
            symlink(
                Path::new("..").join("..").join(BLOBS_DIR).join(blob),
                snapshot.join(name),
            )
            .expect("snapshot symlink");
        }
        (snapshot, fs::canonicalize(&blobs).expect("canonical blobs"))
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
        assert!(matches!(
            package.weight_layout(),
            WeightLayout::Single { .. }
        ));
        let artifact = package
            .open_weights_for("embeddings.weight")
            .expect("artifact");
        assert_eq!(
            artifact
                .tensor("embeddings.weight")
                .expect("tensor")
                .shape(),
            [1, 3]
        );
        let mut weights = package.weight_set();
        assert_eq!(weights.opened_shard_count(), 0);
        assert_eq!(
            weights
                .tensor("embeddings.weight")
                .expect("cached tensor")
                .shape(),
            [1, 3]
        );
        assert_eq!(weights.opened_shard_count(), 1);
        assert!(weights.tensor("embeddings.weight").is_ok());
        assert_eq!(weights.opened_shard_count(), 1);
        assert!(
            package
                .package_file("tokenizer.json")
                .expect("tokenizer")
                .is_file()
        );
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

        let mut weights = package.weight_set();
        assert_eq!(weights.opened_shard_count(), 0);
        assert!(weights.tensor("projection.weight").is_ok());
        assert_eq!(weights.opened_shard_count(), 1);
        assert!(weights.tensor("projection.weight").is_ok());
        assert_eq!(weights.opened_shard_count(), 1);
        assert!(weights.tensor("embeddings.weight").is_ok());
        assert_eq!(weights.opened_shard_count(), 2);
        assert!(matches!(
            weights.tensor("missing.weight"),
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
            Err(PackageError::UnsafeShardPath(path)) if path.as_path() == Path::new("../outside.safetensors")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn resolves_symlinked_cache_snapshot_through_repository_blobs() {
        let dir = TestDir::new();
        let (snapshot, blobs) = cache_repository(
            dir.path(),
            "8f9a1c2d",
            &[
                ("blob-config", CONFIG_FILE, CONFIG_BYTES.to_vec()),
                (
                    "blob-weights",
                    SINGLE_WEIGHTS,
                    safetensors_fixture("embeddings.weight", &[1.0, 2.0, 3.0], &[1, 3]),
                ),
                ("blob-tokenizer", "tokenizer.json", b"{}\n".to_vec()),
            ],
        );

        let package = LocalModelPackage::open(&snapshot).expect("snapshot package");
        assert_eq!(package.root(), fs::canonicalize(&snapshot).expect("root"));
        assert_eq!(package.config_path(), blobs.join("blob-config"));
        assert!(matches!(
            package.weight_layout(),
            WeightLayout::Single { path } if path == blobs.join("blob-weights")
        ));
        let artifact = package
            .open_weights_for("embeddings.weight")
            .expect("artifact through the snapshot link");
        assert!(artifact.tensor("embeddings.weight").is_ok());
        assert_eq!(
            package.package_file("tokenizer.json").expect("tokenizer"),
            blobs.join("blob-tokenizer")
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolves_sharded_cache_snapshot_through_repository_blobs() {
        let dir = TestDir::new();
        let first = "model-00001-of-00002.safetensors";
        let second = "model-00002-of-00002.safetensors";
        let index = format!(
            r#"{{"metadata":{{"total_size":24}},"weight_map":{{"embeddings.weight":"{first}","projection.weight":"{second}"}}}}"#
        );
        let (snapshot, blobs) = cache_repository(
            dir.path(),
            "8f9a1c2d",
            &[
                ("blob-config", CONFIG_FILE, CONFIG_BYTES.to_vec()),
                ("blob-index", WEIGHT_INDEX, index.into_bytes()),
                (
                    "blob-first",
                    first,
                    safetensors_fixture("embeddings.weight", &[1.0, 2.0, 3.0], &[1, 3]),
                ),
                (
                    "blob-second",
                    second,
                    safetensors_fixture("projection.weight", &[4.0, 5.0, 6.0], &[1, 3]),
                ),
            ],
        );

        let package = LocalModelPackage::open(&snapshot).expect("sharded snapshot package");
        let WeightLayout::Sharded {
            shards,
            parameter_count,
            ..
        } = package.weight_layout()
        else {
            panic!("expected sharded weights");
        };
        assert_eq!(parameter_count, 2);
        assert_eq!(
            shards,
            vec![blobs.join("blob-first"), blobs.join("blob-second")]
        );
        let mut weights = package.weight_set();
        assert!(weights.tensor("embeddings.weight").is_ok());
        assert!(weights.tensor("projection.weight").is_ok());
        assert_eq!(weights.opened_shard_count(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn cache_snapshot_policy_still_rejects_links_outside_the_repository() {
        let dir = TestDir::new();
        let outside = dir.path().join("outside.safetensors");
        fs::write(
            &outside,
            safetensors_fixture("embeddings.weight", &[1.0, 2.0, 3.0], &[1, 3]),
        )
        .expect("outside artifact");
        let (snapshot, _) = cache_repository(
            dir.path(),
            "8f9a1c2d",
            &[
                ("blob-config", CONFIG_FILE, CONFIG_BYTES.to_vec()),
                (
                    "blob-weights",
                    SINGLE_WEIGHTS,
                    safetensors_fixture("embeddings.weight", &[1.0, 2.0, 3.0], &[1, 3]),
                ),
            ],
        );
        fs::write(dir.path().join("secret.json"), b"{}\n").expect("outside file");
        symlink(
            dir.path().join("secret.json"),
            snapshot.join("tokenizer.json"),
        )
        .expect("escaping package link");

        let package = LocalModelPackage::open(&snapshot).expect("snapshot package");
        assert!(matches!(
            package.package_file("tokenizer.json"),
            Err(PackageError::UnsafePackagePath(path)) if path.as_path() == Path::new("tokenizer.json")
        ));

        // A weight member that escapes the repository is rejected while opening,
        // so a package never exposes an authorized handle to an outside file.
        let escaped = cache_repository(
            dir.path(),
            "0badc0de",
            &[("blob-config", CONFIG_FILE, CONFIG_BYTES.to_vec())],
        )
        .0;
        symlink(&outside, escaped.join(SINGLE_WEIGHTS)).expect("escaping weight link");
        assert!(matches!(
            LocalModelPackage::open(&escaped),
            Err(PackageError::UnsafeShardPath(path)) if path.as_path() == Path::new(SINGLE_WEIGHTS)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_shaped_directories_that_are_not_cache_repositories_stay_confined() {
        let dir = TestDir::new();
        let outside = dir.path().join("outside.safetensors");
        fs::write(
            &outside,
            safetensors_fixture("embeddings.weight", &[1.0, 2.0, 3.0], &[1, 3]),
        )
        .expect("outside artifact");

        // Same directory shape as a cache snapshot, but the repository directory
        // is not a `models--` cache entry, so only the directory itself is trusted.
        let snapshot = dir.path().join("plain/snapshots/8f9a1c2d");
        fs::create_dir_all(dir.path().join("plain/blobs")).expect("sibling blobs");
        fs::create_dir_all(&snapshot).expect("snapshot directory");
        write_config(&snapshot);
        symlink(&outside, snapshot.join(SINGLE_WEIGHTS)).expect("escaping weight link");

        assert!(matches!(
            LocalModelPackage::open(&snapshot),
            Err(PackageError::UnsafeShardPath(path)) if path.as_path() == Path::new(SINGLE_WEIGHTS)
        ));
    }
}
