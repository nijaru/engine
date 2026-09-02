//! Typed model-state descriptions and lifecycle boundary.

use std::fmt;

use crate::device::DeviceId;
use crate::tensor::DataType;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateId(u64);

#[cfg_attr(not(test), allow(dead_code))]
impl StateId {
    pub(crate) const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateLocation {
    Device(DeviceId),
    Host,
}

/// Full-attention KV state is described independently from recurrent state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvStateSpec {
    layer_count: u16,
    kv_heads: u16,
    head_dim: u16,
    block_tokens: u32,
    dtype: DataType,
}

impl KvStateSpec {
    /// # Errors
    ///
    /// Returns [`StateSpecError::ZeroDimension`] when any state dimension is
    /// zero.
    pub fn new(
        layer_count: u16,
        kv_heads: u16,
        head_dim: u16,
        block_tokens: u32,
        dtype: DataType,
    ) -> Result<Self, StateSpecError> {
        if layer_count == 0 || kv_heads == 0 || head_dim == 0 || block_tokens == 0 {
            return Err(StateSpecError::ZeroDimension);
        }
        Ok(Self {
            layer_count,
            kv_heads,
            head_dim,
            block_tokens,
            dtype,
        })
    }

    #[must_use]
    pub const fn layer_count(self) -> u16 {
        self.layer_count
    }

    #[must_use]
    pub const fn kv_heads(self) -> u16 {
        self.kv_heads
    }

    #[must_use]
    pub const fn head_dim(self) -> u16 {
        self.head_dim
    }

    #[must_use]
    pub const fn block_tokens(self) -> u32 {
        self.block_tokens
    }

    #[must_use]
    pub const fn dtype(self) -> DataType {
        self.dtype
    }
}

/// Matrix dimensions for the recurrent/linear-attention state. Qwen3.8 uses
/// 16 key heads × 128 key width and 48 value heads × 128 value width.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecurrentMatrixShape {
    key_heads: u16,
    key_head_dim: u16,
    value_heads: u16,
    value_head_dim: u16,
}

impl RecurrentMatrixShape {
    #[must_use]
    pub const fn new(
        key_heads: u16,
        key_head_dim: u16,
        value_heads: u16,
        value_head_dim: u16,
    ) -> Option<Self> {
        if key_heads == 0 || key_head_dim == 0 || value_heads == 0 || value_head_dim == 0 {
            None
        } else {
            Some(Self {
                key_heads,
                key_head_dim,
                value_heads,
                value_head_dim,
            })
        }
    }

    #[must_use]
    pub const fn key_heads(self) -> u16 {
        self.key_heads
    }

    #[must_use]
    pub const fn key_head_dim(self) -> u16 {
        self.key_head_dim
    }

    #[must_use]
    pub const fn value_heads(self) -> u16 {
        self.value_heads
    }

    #[must_use]
    pub const fn value_head_dim(self) -> u16 {
        self.value_head_dim
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConvolutionStateShape {
    channels: u32,
    kernel: u16,
}

impl ConvolutionStateShape {
    #[must_use]
    pub const fn new(channels: u32, kernel: u16) -> Option<Self> {
        if channels == 0 || kernel == 0 {
            None
        } else {
            Some(Self { channels, kernel })
        }
    }

    #[must_use]
    pub const fn channels(self) -> u32 {
        self.channels
    }

    #[must_use]
    pub const fn kernel(self) -> u16 {
        self.kernel
    }
}

/// Recurrent/Gated-DeltaNet state includes the matrix state and convolution
/// history. It is not interchangeable with KV state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecurrentStateSpec {
    layer_count: u16,
    matrix: RecurrentMatrixShape,
    convolution: ConvolutionStateShape,
    matrix_dtype: DataType,
    convolution_dtype: DataType,
}

impl RecurrentStateSpec {
    /// # Errors
    ///
    /// Returns [`StateSpecError::ZeroDimension`] when `layer_count` is zero.
    pub fn new(
        layer_count: u16,
        matrix: RecurrentMatrixShape,
        convolution: ConvolutionStateShape,
        matrix_dtype: DataType,
        convolution_dtype: DataType,
    ) -> Result<Self, StateSpecError> {
        if layer_count == 0 {
            return Err(StateSpecError::ZeroDimension);
        }
        Ok(Self {
            layer_count,
            matrix,
            convolution,
            matrix_dtype,
            convolution_dtype,
        })
    }

    #[must_use]
    pub const fn layer_count(self) -> u16 {
        self.layer_count
    }

    #[must_use]
    pub const fn matrix(self) -> RecurrentMatrixShape {
        self.matrix
    }

    #[must_use]
    pub const fn convolution(self) -> ConvolutionStateShape {
        self.convolution
    }

    #[must_use]
    pub const fn matrix_dtype(self) -> DataType {
        self.matrix_dtype
    }

    #[must_use]
    pub const fn convolution_dtype(self) -> DataType {
        self.convolution_dtype
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateRequirement {
    FullAttentionKv(KvStateSpec),
    Recurrent(RecurrentStateSpec),
}

impl StateRequirement {
    #[must_use]
    pub const fn is_kv(self) -> bool {
        matches!(self, Self::FullAttentionKv(_))
    }

    #[must_use]
    pub const fn is_recurrent(self) -> bool {
        matches!(self, Self::Recurrent(_))
    }
}

/// A handle identifies backend-owned storage. Handles are manager-issued and
/// opaque to callers; the core does not pretend that a handle contains tensor
/// data or provide a fake allocator implementation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct StateHandle {
    id: StateId,
    requirement: StateRequirement,
    location: StateLocation,
    token_position: u32,
}

#[cfg_attr(not(test), allow(dead_code))]
impl StateHandle {
    pub(crate) const fn new(
        id: StateId,
        requirement: StateRequirement,
        location: StateLocation,
        token_position: u32,
    ) -> Self {
        Self {
            id,
            requirement,
            location,
            token_position,
        }
    }

    #[must_use]
    pub const fn id(&self) -> StateId {
        self.id
    }

    #[must_use]
    pub const fn requirement(&self) -> StateRequirement {
        self.requirement
    }

    #[must_use]
    pub const fn location(&self) -> StateLocation {
        self.location
    }

    #[must_use]
    pub const fn token_position(&self) -> u32 {
        self.token_position
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct KvState {
    handle: StateHandle,
    spec: KvStateSpec,
}

#[cfg_attr(not(test), allow(dead_code))]
impl KvState {
    pub(crate) fn new(handle: StateHandle, spec: KvStateSpec) -> Option<Self> {
        if matches!(handle.requirement(), StateRequirement::FullAttentionKv(actual) if actual == spec)
        {
            Some(Self { handle, spec })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn handle(&self) -> &StateHandle {
        &self.handle
    }

    #[must_use]
    pub const fn spec(&self) -> KvStateSpec {
        self.spec
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RecurrentState {
    handle: StateHandle,
    spec: RecurrentStateSpec,
}

#[cfg_attr(not(test), allow(dead_code))]
impl RecurrentState {
    pub(crate) fn new(handle: StateHandle, spec: RecurrentStateSpec) -> Option<Self> {
        if matches!(handle.requirement(), StateRequirement::Recurrent(actual) if actual == spec) {
            Some(Self { handle, spec })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn handle(&self) -> &StateHandle {
        &self.handle
    }

    #[must_use]
    pub const fn spec(&self) -> RecurrentStateSpec {
        self.spec
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum HybridState {
    Kv(KvState),
    Recurrent(RecurrentState),
}

#[cfg_attr(not(test), allow(dead_code))]
impl HybridState {
    #[must_use]
    pub const fn handle(&self) -> &StateHandle {
        match self {
            Self::Kv(state) => state.handle(),
            Self::Recurrent(state) => state.handle(),
        }
    }
}

/// The first Qwen3.8 path needs both families at once. Keeping them in named
/// optional fields prevents an accidental KV-only representation and validates
/// that a shared prefix boundary has one position and one placement.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct HybridStateSet {
    kv: Option<KvState>,
    recurrent: Option<RecurrentState>,
}

impl HybridStateSet {
    /// # Errors
    ///
    /// Returns a state error when both families are present but disagree on
    /// token position or placement.
    pub fn try_new(
        kv: Option<KvState>,
        recurrent: Option<RecurrentState>,
    ) -> Result<Self, StateError> {
        if let (Some(kv), Some(recurrent)) = (&kv, &recurrent) {
            if kv.handle().token_position() != recurrent.handle().token_position() {
                return Err(StateError::PositionMismatch);
            }
            if kv.handle().location() != recurrent.handle().location() {
                return Err(StateError::LocationMismatch);
            }
        }
        Ok(Self { kv, recurrent })
    }

    #[must_use]
    pub fn kv(&self) -> Option<&KvState> {
        self.kv.as_ref()
    }

    #[must_use]
    pub fn recurrent(&self) -> Option<&RecurrentState> {
        self.recurrent.as_ref()
    }

    #[must_use]
    pub fn token_position(&self) -> Option<u32> {
        self.kv()
            .map(|state| state.handle().token_position())
            .or_else(|| {
                self.recurrent()
                    .map(|state| state.handle().token_position())
            })
    }

    #[must_use]
    pub fn location(&self) -> Option<StateLocation> {
        self.kv()
            .map(|state| state.handle().location())
            .or_else(|| self.recurrent().map(|state| state.handle().location()))
    }

    #[must_use]
    pub fn contains(&self, requirement: StateRequirement) -> bool {
        match requirement {
            StateRequirement::FullAttentionKv(spec) => {
                self.kv().is_some_and(|state| state.spec() == spec)
            }
            StateRequirement::Recurrent(spec) => {
                self.recurrent().is_some_and(|state| state.spec() == spec)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateSpecError {
    ZeroDimension,
}

impl fmt::Display for StateSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => f.write_str("state dimensions must be greater than zero"),
        }
    }
}

impl std::error::Error for StateSpecError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateError {
    UnsupportedRequirement,
    InvalidHandle,
    PositionMismatch,
    LocationMismatch,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRequirement => f.write_str("state requirement is unsupported"),
            Self::InvalidHandle => f.write_str("state handle is invalid or already released"),
            Self::PositionMismatch => f.write_str("state families have different token positions"),
            Self::LocationMismatch => f.write_str("state families have different placements"),
        }
    }
}

impl std::error::Error for StateError {}

/// Storage/lifecycle implementations own allocation, transfer, checkpointing,
/// and reclamation. Separate methods preserve the type distinction at the API.
pub trait StateManager: Send {
    /// # Errors
    ///
    /// Returns [`StateError`] when storage cannot satisfy the KV allocation.
    fn allocate_kv(
        &mut self,
        spec: KvStateSpec,
        location: StateLocation,
    ) -> Result<KvState, StateError>;

    /// # Errors
    ///
    /// Returns [`StateError`] when storage cannot satisfy the recurrent
    /// allocation.
    fn allocate_recurrent(
        &mut self,
        spec: RecurrentStateSpec,
        location: StateLocation,
    ) -> Result<RecurrentState, StateError>;

    /// Commit a model-defined transition for all state families in one prefix
    /// boundary and return the manager-issued handles for that boundary.
    ///
    /// # Errors
    ///
    /// Returns [`StateError`] when the transition cannot be committed or the
    /// resulting families cannot share one position and placement.
    fn commit(
        &mut self,
        state: HybridStateSet,
        token_position: u32,
    ) -> Result<HybridStateSet, StateError>;

    /// # Errors
    ///
    /// Returns [`StateError::InvalidHandle`] when the handle is unknown or was
    /// already released.
    fn release(&mut self, handle: StateHandle) -> Result<(), StateError>;
}
