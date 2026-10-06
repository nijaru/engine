/// Identity of the coherent parameters used by a batch executor.
///
/// This is a version label, not ownership of an executable snapshot. The executor
/// must keep the corresponding weights alive and coordinate any replacement.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParameterVersion(u64);

impl ParameterVersion {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Artifact scalar metadata, independent of backend storage or kernel support.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ScalarType {
    F16,
    Bf16,
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Bool,
    Named(String),
}
