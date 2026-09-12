from pathlib import Path


hf = Path("crates/hf/src/lib.rs")
text = hf.read_text()
text = text.replace(
    "/// Lazy reusable view over the SafeTensors files resolved by one local package.",
    "/// Lazy reusable view over the `SafeTensors` files resolved by one local package.",
    1,
)
text = text.replace(
    "    /// Number of unique SafeTensors shards opened so far.",
    "    /// Number of unique `SafeTensors` shards opened so far.",
    1,
)
start_marker = "    /// Resolve and borrow one parameter tensor, lazily opening its shard once.\n"
end_marker = "\n}\n\nfn open_artifact"
start = text.index(start_marker)
end = text.index(end_marker, start)
new = '''    /// Resolve and borrow the `SafeTensors` artifact containing one parameter.
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
    }'''
hf.write_text(text[:start] + new + text[end:])

bert = Path("crates/batch/tests/bert_architecture.rs")
text = bert.read_text()
for line in [
    "use std::collections::BTreeMap;\n",
    "use ribn_hf::LocalModelPackage;\n",
    "use ribn_safetensors::{ArtifactError, SafeTensorArtifact};\n",
]:
    if line not in text:
        raise SystemExit(f"missing BERT import line: {line!r}")
text = text.replace("use std::collections::BTreeMap;\n", "", 1)
text = text.replace(
    "use ribn_hf::LocalModelPackage;\n",
    "use ribn_hf::{LocalModelPackage, LocalWeightSet, PackageError};\n",
    1,
)
text = text.replace(
    "use ribn_safetensors::{ArtifactError, SafeTensorArtifact};\n",
    "use ribn_safetensors::ArtifactError;\n",
    1,
)

start = text.index("struct ArtifactCache<'a> {")
end = text.index("\nfn decode_f32_tensor(", start)
replacement = '''struct ModelWeights<'a> {
    weights: LocalWeightSet<'a>,
}

impl<'a> ModelWeights<'a> {
    fn new(package: &'a LocalModelPackage) -> Self {
        Self {
            weights: package.weight_set(),
        }
    }

    fn load_required(
        &mut self,
        suffix: &str,
        expected_shape: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        self.load_optional(suffix, expected_shape)?
            .ok_or_else(|| ModelError::MissingParameter(suffix.to_owned()))
    }

    fn load_optional(
        &mut self,
        suffix: &str,
        expected_shape: &[usize],
    ) -> Result<Option<Vec<f32>>, ModelError> {
        for name in [format!("bert.{suffix}"), suffix.to_owned()] {
            let artifact = match self.weights.artifact(&name) {
                Ok(artifact) => artifact,
                Err(PackageError::UnknownParameter(_)) => continue,
                Err(error) => return Err(ModelError::Artifact(error.to_string())),
            };
            let tensor = match artifact.tensor(&name) {
                Ok(tensor) => tensor,
                Err(ArtifactError::MissingTensor(_)) => continue,
                Err(error) => return Err(ModelError::Artifact(error.to_string())),
            };
            return decode_f32_tensor(&name, &tensor, expected_shape).map(Some);
        }
        Ok(None)
    }

    fn artifact_count(&self) -> usize {
        self.weights.opened_shard_count()
    }
}
'''
text = text[:start] + replacement + text[end:]
text = text.replace("ArtifactCache<'_>", "ModelWeights<'_>")
old = "let mut cache = ArtifactCache::new(package);"
if old not in text:
    raise SystemExit("BERT cache constructor anchor not found")
text = text.replace(old, "let mut cache = ModelWeights::new(package);", 1)
for forbidden in ["ArtifactCache", "SafeTensorArtifact", "BTreeMap"]:
    if forbidden in text:
        raise SystemExit(f"private BERT shard cache token remains: {forbidden}")
bert.write_text(text)
