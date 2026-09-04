//! Typed model-state descriptions and lifecycle boundary.

use std::collections::HashMap;
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
    /// Returns [`StateSpecError::ZeroDimension`] when any state dimension is zero.
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

    #[must_use]
    pub fn byte_size(self) -> Option<u64> {
        let elements = u64::from(self.layer_count)
            .checked_mul(u64::from(self.kv_heads))?
            .checked_mul(u64::from(self.head_dim))?
            .checked_mul(u64::from(self.block_tokens))?
            .checked_mul(2)?;
        elements.checked_mul(self.dtype.byte_width())
    }
}

/// Storage geometry for a bank of recurrent state matrices.
///
/// This describes physical semantic state as `[matrix][row][column]`. Model
/// projection head counts are deliberately absent: a Gated-DeltaNet model can
/// tile projected key heads across state matrices without multiplying the
/// persistent state allocation by the projection-head count.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecurrentMatrixShape {
    matrix_count: u16,
    rows: u16,
    columns: u16,
}

impl RecurrentMatrixShape {
    #[must_use]
    pub const fn new(matrix_count: u16, rows: u16, columns: u16) -> Option<Self> {
        if matrix_count == 0 || rows == 0 || columns == 0 {
            None
        } else {
            Some(Self {
                matrix_count,
                rows,
                columns,
            })
        }
    }

    #[must_use]
    pub const fn matrix_count(self) -> u16 {
        self.matrix_count
    }

    #[must_use]
    pub const fn rows(self) -> u16 {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> u16 {
        self.columns
    }
}

/// Persistent causal-convolution history as `[channel][history_token]`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConvolutionStateShape {
    channels: u32,
    history_tokens: u16,
}

impl ConvolutionStateShape {
    #[must_use]
    pub const fn new(channels: u32, history_tokens: u16) -> Option<Self> {
        if channels == 0 || history_tokens == 0 {
            None
        } else {
            Some(Self {
                channels,
                history_tokens,
            })
        }
    }

    #[must_use]
    pub const fn channels(self) -> u32 {
        self.channels
    }

    #[must_use]
    pub const fn history_tokens(self) -> u16 {
        self.history_tokens
    }
}

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

    #[must_use]
    pub fn byte_size(self) -> Option<u64> {
        let matrix_elements = u64::from(self.matrix.matrix_count())
            .checked_mul(u64::from(self.matrix.rows()))?
            .checked_mul(u64::from(self.matrix.columns()))?
            .checked_mul(u64::from(self.layer_count))?;
        let convolution_elements = u64::from(self.convolution.channels())
            .checked_mul(u64::from(self.convolution.history_tokens()))?
            .checked_mul(u64::from(self.layer_count))?;
        matrix_elements
            .checked_mul(self.matrix_dtype.byte_width())?
            .checked_add(convolution_elements.checked_mul(self.convolution_dtype.byte_width())?)
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

    #[must_use]
    pub fn byte_size(self) -> Option<u64> {
        match self {
            Self::FullAttentionKv(spec) => spec.byte_size(),
            Self::Recurrent(spec) => spec.byte_size(),
        }
    }
}

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
pub enum InferenceState {
    Kv(KvState),
    Recurrent(RecurrentState),
}

impl InferenceState {
    #[must_use]
    pub const fn handle(&self) -> &StateHandle {
        match self {
            Self::Kv(state) => state.handle(),
            Self::Recurrent(state) => state.handle(),
        }
    }

    #[must_use]
    pub const fn requirement(&self) -> StateRequirement {
        self.handle().requirement()
    }

    fn with_position(&self, token_position: u32) -> Result<Self, StateError> {
        let handle = StateHandle::new(
            self.handle().id(),
            self.handle().requirement(),
            self.handle().location(),
            token_position,
        );
        match self {
            Self::Kv(state) => KvState::new(handle, state.spec())
                .map(Self::Kv)
                .ok_or(StateError::InvalidHandle),
            Self::Recurrent(state) => RecurrentState::new(handle, state.spec())
                .map(Self::Recurrent)
                .ok_or(StateError::InvalidHandle),
        }
    }
}

impl From<KvState> for InferenceState {
    fn from(value: KvState) -> Self {
        Self::Kv(value)
    }
}

impl From<RecurrentState> for InferenceState {
    fn from(value: RecurrentState) -> Self {
        Self::Recurrent(value)
    }
}

/// Typed state at one semantic prefix boundary. New state families extend
/// [`InferenceState`] rather than changing scheduler/backend signatures.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InferenceStateSet {
    states: Vec<InferenceState>,
}

impl InferenceStateSet {
    /// # Errors
    ///
    /// Returns a state error when states disagree on token position or location,
    /// or when the same state requirement appears more than once.
    pub fn new(states: Vec<InferenceState>) -> Result<Self, StateError> {
        if let Some(first) = states.first() {
            let position = first.handle().token_position();
            let location = first.handle().location();
            for (index, state) in states.iter().enumerate() {
                if state.handle().token_position() != position {
                    return Err(StateError::PositionMismatch);
                }
                if state.handle().location() != location {
                    return Err(StateError::LocationMismatch);
                }
                if states[..index]
                    .iter()
                    .any(|known| known.requirement() == state.requirement())
                {
                    return Err(StateError::DuplicateRequirement);
                }
            }
        }
        Ok(Self { states })
    }

    /// Convenience constructor for the current Qwen hybrid path.
    ///
    /// # Errors
    ///
    /// Returns a state error if the supplied states do not share a semantic
    /// prefix boundary or duplicate a state requirement.
    pub fn try_new(
        kv: Option<KvState>,
        recurrent: Option<RecurrentState>,
    ) -> Result<Self, StateError> {
        let mut states = Vec::with_capacity(2);
        if let Some(state) = kv {
            states.push(state.into());
        }
        if let Some(state) = recurrent {
            states.push(state.into());
        }
        Self::new(states)
    }

    #[must_use]
    pub fn states(&self) -> &[InferenceState] {
        &self.states
    }

    #[must_use]
    pub fn kv(&self) -> Option<&KvState> {
        self.states.iter().find_map(|state| match state {
            InferenceState::Kv(value) => Some(value),
            InferenceState::Recurrent(_) => None,
        })
    }

    #[must_use]
    pub fn recurrent(&self) -> Option<&RecurrentState> {
        self.states.iter().find_map(|state| match state {
            InferenceState::Kv(_) => None,
            InferenceState::Recurrent(value) => Some(value),
        })
    }

    #[must_use]
    pub fn token_position(&self) -> Option<u32> {
        self.states
            .first()
            .map(|state| state.handle().token_position())
    }

    #[must_use]
    pub fn location(&self) -> Option<StateLocation> {
        self.states.first().map(|state| state.handle().location())
    }

    #[must_use]
    pub fn contains(&self, requirement: StateRequirement) -> bool {
        self.states
            .iter()
            .any(|state| state.requirement() == requirement)
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
    UnsupportedLocation,
    CapacityExceeded {
        requested_bytes: u64,
        available_bytes: u64,
    },
    SizeOverflow,
    InvalidHandle,
    DuplicateRequirement,
    PositionMismatch,
    LocationMismatch,
    PositionRegression,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRequirement => f.write_str("state requirement is unsupported"),
            Self::UnsupportedLocation => {
                f.write_str("state location is unsupported by this manager")
            }
            Self::CapacityExceeded {
                requested_bytes,
                available_bytes,
            } => write!(
                f,
                "state allocation requires {requested_bytes} bytes, but only {available_bytes} are available"
            ),
            Self::SizeOverflow => f.write_str("state size calculation overflowed"),
            Self::InvalidHandle => f.write_str("state handle is invalid or already released"),
            Self::DuplicateRequirement => f.write_str("state requirement appears more than once"),
            Self::PositionMismatch => f.write_str("state families have different token positions"),
            Self::LocationMismatch => f.write_str("state families have different placements"),
            Self::PositionRegression => f.write_str("state token position cannot move backwards"),
        }
    }
}

impl std::error::Error for StateError {}

#[derive(Debug)]
pub struct LogicalStateManager {
    device: DeviceId,
    device_capacity_bytes: u64,
    host_capacity_bytes: u64,
    device_used_bytes: u64,
    host_used_bytes: u64,
    next_id: u64,
    allocations: HashMap<StateId, AllocationRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AllocationRecord {
    requirement: StateRequirement,
    location: StateLocation,
    token_position: u32,
    bytes: u64,
}

impl LogicalStateManager {
    #[must_use]
    pub fn new(device: DeviceId, device_capacity_bytes: u64, host_capacity_bytes: u64) -> Self {
        Self {
            device,
            device_capacity_bytes,
            host_capacity_bytes,
            device_used_bytes: 0,
            host_used_bytes: 0,
            next_id: 1,
            allocations: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    #[must_use]
    pub fn capacity_bytes(&self, location: StateLocation) -> Option<u64> {
        match location {
            StateLocation::Device(device) if device == self.device => {
                Some(self.device_capacity_bytes)
            }
            StateLocation::Host => Some(self.host_capacity_bytes),
            StateLocation::Device(_) => None,
        }
    }

    #[must_use]
    pub fn used_bytes(&self, location: StateLocation) -> Option<u64> {
        match location {
            StateLocation::Device(device) if device == self.device => Some(self.device_used_bytes),
            StateLocation::Host => Some(self.host_used_bytes),
            StateLocation::Device(_) => None,
        }
    }

    fn reserve(
        &mut self,
        requirement: StateRequirement,
        location: StateLocation,
    ) -> Result<StateHandle, StateError> {
        let bytes = requirement.byte_size().ok_or(StateError::SizeOverflow)?;
        let (capacity, used) = match location {
            StateLocation::Device(device) if device == self.device => {
                (self.device_capacity_bytes, self.device_used_bytes)
            }
            StateLocation::Host => (self.host_capacity_bytes, self.host_used_bytes),
            StateLocation::Device(_) => return Err(StateError::UnsupportedLocation),
        };
        let available = capacity.saturating_sub(used);
        if bytes > available {
            return Err(StateError::CapacityExceeded {
                requested_bytes: bytes,
                available_bytes: available,
            });
        }

        let id = StateId::new(self.next_id).ok_or(StateError::InvalidHandle)?;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(StateError::InvalidHandle)?;

        if matches!(location, StateLocation::Device(_)) {
            self.device_used_bytes += bytes;
        } else {
            self.host_used_bytes += bytes;
        }

        self.allocations.insert(
            id,
            AllocationRecord {
                requirement,
                location,
                token_position: 0,
                bytes,
            },
        );
        Ok(StateHandle::new(id, requirement, location, 0))
    }

    fn record_for(&self, handle: &StateHandle) -> Result<AllocationRecord, StateError> {
        let Some(record) = self.allocations.get(&handle.id()) else {
            return Err(StateError::InvalidHandle);
        };
        if record.requirement != handle.requirement()
            || record.location != handle.location()
            || record.token_position != handle.token_position()
        {
            return Err(StateError::InvalidHandle);
        }
        Ok(*record)
    }

    fn update_position(
        &mut self,
        state: &InferenceStateSet,
        token_position: u32,
    ) -> Result<(), StateError> {
        for value in state.states() {
            let record = self.record_for(value.handle())?;
            if token_position < record.token_position {
                return Err(StateError::PositionRegression);
            }
        }
        for value in state.states() {
            let record = self
                .allocations
                .get_mut(&value.handle().id())
                .ok_or(StateError::InvalidHandle)?;
            record.token_position = token_position;
        }
        Ok(())
    }
}

impl StateManager for LogicalStateManager {
    fn allocate_kv(
        &mut self,
        spec: KvStateSpec,
        location: StateLocation,
    ) -> Result<KvState, StateError> {
        let handle = self.reserve(StateRequirement::FullAttentionKv(spec), location)?;
        KvState::new(handle, spec).ok_or(StateError::UnsupportedRequirement)
    }

    fn allocate_recurrent(
        &mut self,
        spec: RecurrentStateSpec,
        location: StateLocation,
    ) -> Result<RecurrentState, StateError> {
        let handle = self.reserve(StateRequirement::Recurrent(spec), location)?;
        RecurrentState::new(handle, spec).ok_or(StateError::UnsupportedRequirement)
    }

    fn commit(
        &mut self,
        state: InferenceStateSet,
        token_position: u32,
    ) -> Result<InferenceStateSet, StateError> {
        self.update_position(&state, token_position)?;
        let states = state
            .states()
            .iter()
            .map(|value| value.with_position(token_position))
            .collect::<Result<Vec<_>, _>>()?;
        InferenceStateSet::new(states)
    }

    fn release(&mut self, handle: StateHandle) -> Result<(), StateError> {
        let record = self.record_for(&handle)?;
        self.allocations.remove(&handle.id());
        if matches!(record.location, StateLocation::Device(_)) {
            self.device_used_bytes -= record.bytes;
        } else {
            self.host_used_bytes -= record.bytes;
        }
        Ok(())
    }
}

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
    /// Returns [`StateError`] when storage cannot satisfy the recurrent allocation.
    fn allocate_recurrent(
        &mut self,
        spec: RecurrentStateSpec,
        location: StateLocation,
    ) -> Result<RecurrentState, StateError>;

    /// # Errors
    ///
    /// Returns [`StateError`] when the state transition cannot be committed.
    fn commit(
        &mut self,
        state: InferenceStateSet,
        token_position: u32,
    ) -> Result<InferenceStateSet, StateError>;

    /// # Errors
    ///
    /// Returns [`StateError::InvalidHandle`] when the handle is unknown or was released.
    fn release(&mut self, handle: StateHandle) -> Result<(), StateError>;
}
