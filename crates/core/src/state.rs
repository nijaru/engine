//! Typed model-state descriptions and lifecycle boundary.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ribn_foundation::{BytePool, PoolLease, ReserveError};

static NEXT_STATE_ID: AtomicU64 = AtomicU64::new(1);

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

    /// The same shape with a different token capacity.
    ///
    /// A declaration states a shape and the most this component can hold; a concrete
    /// sequence materializes the capacity its own work can reach. This is the only
    /// way to derive that concrete requirement, so capacity is never edited in place.
    ///
    /// # Errors
    /// Returns [`StateSpecError::ZeroDimension`] when `block_tokens` is zero.
    pub fn with_capacity(self, block_tokens: u32) -> Result<Self, StateSpecError> {
        Self::new(
            self.layer_count,
            self.kv_heads,
            self.head_dim,
            block_tokens,
            self.dtype,
        )
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

    /// This requirement with a concrete continuation capacity of `tokens`.
    ///
    /// A declaration covers a maximum capacity, so one declared schema derives the
    /// concrete requirement of any sequence that reaches no further than `tokens`.
    /// Recurrent state is a fixed-size summary of the whole prefix and has no
    /// capacity dimension, so it is returned unchanged.
    ///
    /// # Errors
    /// Returns [`StateSpecError::ZeroDimension`] when `tokens` is zero.
    pub fn with_capacity(self, tokens: u32) -> Result<Self, StateSpecError> {
        match self {
            Self::FullAttentionKv(spec) => Ok(Self::FullAttentionKv(spec.with_capacity(tokens)?)),
            Self::Recurrent(spec) => Ok(Self::Recurrent(spec)),
        }
    }

    /// Whether this concrete requirement fits `declaration`.
    ///
    /// Same component with a matching shape and no more capacity than the declared
    /// maximum is covered. A declaration is a schema plus a bound, so concrete state
    /// that reaches less of the context is valid, while a different shape, dtype or an
    /// exceeding capacity is not.
    #[must_use]
    pub fn within(self, declaration: Self) -> bool {
        match (self, declaration) {
            (Self::FullAttentionKv(concrete), Self::FullAttentionKv(declared)) => {
                concrete.layer_count == declared.layer_count
                    && concrete.kv_heads == declared.kv_heads
                    && concrete.head_dim == declared.head_dim
                    && concrete.dtype == declared.dtype
                    && concrete.block_tokens <= declared.block_tokens
            }
            (Self::Recurrent(concrete), Self::Recurrent(declared)) => concrete == declared,
            _ => false,
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

#[derive(Debug, Eq, Hash, PartialEq)]
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

#[derive(Debug, Eq, Hash, PartialEq)]
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

#[derive(Debug, Eq, Hash, PartialEq)]
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

    fn set_position(&mut self, token_position: u32) {
        match self {
            Self::Kv(state) => state.handle.token_position = token_position,
            Self::Recurrent(state) => state.handle.token_position = token_position,
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
/// The set transfers logical continuation ownership. The manager retains its
/// byte grant until explicit completion-safe release; inspecting a handle creates
/// neither another continuation owner nor a pool lease.
///
/// ```compile_fail
/// use engine_core::state::InferenceStateSet;
/// let state = InferenceStateSet::new(Vec::new()).unwrap();
/// let duplicate = state.clone();
/// ```
#[derive(Debug, Eq, Hash, PartialEq)]
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

    /// The concrete requirements this state satisfies, in state order.
    #[must_use]
    pub fn requirements(&self) -> Vec<StateRequirement> {
        self.states
            .iter()
            .map(InferenceState::requirement)
            .collect()
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
    PoolClosed,
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
            Self::PoolClosed => f.write_str("state reservation pool is closed"),
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

/// Logical provenance and positions over supplied reservation authorities.
/// This manager does not prove physical completion. Models must release backend
/// storage safely before releasing its records; abandoned records keep their charge.
#[derive(Debug)]
pub struct LogicalStateManager {
    device: DeviceId,
    device_pool: Arc<BytePool>,
    host_pool: Arc<BytePool>,
    allocations: HashMap<StateId, AllocationRecord>,
}

#[derive(Debug)]
struct AllocationRecord {
    requirement: StateRequirement,
    location: StateLocation,
    token_position: u32,
    // One aggregate grant shared by the bundle's records, never charged twice.
    reservation: Arc<PoolLease>,
}

impl LogicalStateManager {
    #[must_use]
    pub fn new(device: DeviceId, device_pool: Arc<BytePool>, host_pool: Arc<BytePool>) -> Self {
        Self {
            device,
            device_pool,
            host_pool,
            allocations: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    fn pool(&self, location: StateLocation) -> Option<&Arc<BytePool>> {
        match location {
            StateLocation::Device(device) if device == self.device => Some(&self.device_pool),
            StateLocation::Host => Some(&self.host_pool),
            StateLocation::Device(_) => None,
        }
    }

    #[must_use]
    pub fn capacity_bytes(&self, location: StateLocation) -> Option<u64> {
        self.pool(location).map(|pool| pool.capacity())
    }

    /// Register before allocation. Returning a grant and closing its authority
    /// publish; commit and unsuccessful allocation/release do not. Siblings share
    /// the source.
    #[must_use]
    pub fn capacity_wait(&self, location: StateLocation) -> Option<ribn_foundation::ReadinessWait> {
        self.pool(location).map(|pool| pool.capacity_wait())
    }

    /// Total grants from this authority, including sibling owners.
    #[must_use]
    pub fn used_bytes(&self, location: StateLocation) -> Option<u64> {
        self.pool(location).map(|pool| pool.granted())
    }

    fn new_handle(
        requirement: StateRequirement,
        location: StateLocation,
    ) -> Result<StateHandle, StateError> {
        // Backend allocation keys must not alias across managers sharing a pool.
        let raw_id = NEXT_STATE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| StateError::InvalidHandle)?;
        let id = StateId::new(raw_id).ok_or(StateError::InvalidHandle)?;
        Ok(StateHandle::new(id, requirement, location, 0))
    }

    fn grant(&self, bytes: u64, location: StateLocation) -> Result<Arc<PoolLease>, StateError> {
        let pool = self.pool(location).ok_or(StateError::UnsupportedLocation)?;
        let lease = pool.reserve(bytes).map_err(|error| match error {
            ReserveError::Exhausted {
                requested,
                available,
            } => StateError::CapacityExceeded {
                requested_bytes: requested,
                available_bytes: available,
            },
            ReserveError::TooLarge { requested, .. } => StateError::CapacityExceeded {
                requested_bytes: requested,
                available_bytes: pool.available(),
            },
            ReserveError::Closed => StateError::PoolClosed,
        })?;
        Ok(Arc::new(lease))
    }

    fn insert(&mut self, handle: &StateHandle, reservation: Arc<PoolLease>) {
        self.allocations.insert(
            handle.id(),
            AllocationRecord {
                requirement: handle.requirement(),
                location: handle.location(),
                token_position: handle.token_position(),
                reservation,
            },
        );
    }

    fn reserve(
        &mut self,
        requirement: StateRequirement,
        location: StateLocation,
    ) -> Result<StateHandle, StateError> {
        let bytes = requirement.byte_size().ok_or(StateError::SizeOverflow)?;
        let handle = Self::new_handle(requirement, location)?;
        let reservation = self.grant(bytes, location)?;
        self.insert(&handle, reservation);
        Ok(handle)
    }

    fn record_for(&self, handle: &StateHandle) -> Result<&AllocationRecord, StateError> {
        let Some(record) = self.allocations.get(&handle.id()) else {
            return Err(StateError::InvalidHandle);
        };
        if record.requirement != handle.requirement()
            || record.location != handle.location()
            || record.token_position != handle.token_position()
        {
            return Err(StateError::InvalidHandle);
        }
        Ok(record)
    }
}

impl Drop for LogicalStateManager {
    fn drop(&mut self) {
        // Logical-owner disappearance is not proof that backend storage retired.
        // Successful model teardown empties these records via explicit release.
        for (_, record) in self.allocations.drain() {
            std::mem::forget(record.reservation);
        }
    }
}

impl StateManager for LogicalStateManager {
    fn allocate_set(
        &mut self,
        requirements: &[StateRequirement],
        location: StateLocation,
    ) -> Result<InferenceStateSet, StateError> {
        let mut bytes = 0_u64;
        let mut states = Vec::with_capacity(requirements.len());
        for &requirement in requirements {
            bytes = bytes
                .checked_add(requirement.byte_size().ok_or(StateError::SizeOverflow)?)
                .ok_or(StateError::SizeOverflow)?;
            let handle = Self::new_handle(requirement, location)?;
            states.push(match requirement {
                StateRequirement::FullAttentionKv(spec) => {
                    InferenceState::Kv(KvState { handle, spec })
                }
                StateRequirement::Recurrent(spec) => {
                    InferenceState::Recurrent(RecurrentState { handle, spec })
                }
            });
        }
        // Shape/duplicate validation precedes the one atomic authority grant.
        // Failure therefore cannot publish a partial rollback that wakes itself.
        let state = InferenceStateSet::new(states)?;
        let reservation = self.grant(bytes, location)?;
        for value in state.states() {
            self.insert(value.handle(), Arc::clone(&reservation));
        }
        Ok(state)
    }

    fn validate(&self, state: &InferenceStateSet) -> Result<(), StateError> {
        for value in state.states() {
            self.record_for(value.handle())?;
        }
        Ok(())
    }

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

    fn commit_batch(
        &mut self,
        states: &mut [InferenceStateSet],
        token_positions: &[u32],
    ) -> Result<(), StateError> {
        if states.len() != token_positions.len() {
            return Err(StateError::PositionMismatch);
        }
        for (state, &position) in states.iter().zip(token_positions) {
            for value in state.states() {
                let record = self.record_for(value.handle())?;
                if position < record.token_position {
                    return Err(StateError::PositionRegression);
                }
            }
        }
        for (state, &position) in states.iter_mut().zip(token_positions) {
            for value in &mut state.states {
                self.allocations
                    .get_mut(&value.handle().id())
                    .expect("all allocation handles were validated")
                    .token_position = position;
                value.set_position(position);
            }
        }
        Ok(())
    }

    fn release_set(&mut self, state: &InferenceStateSet) -> Result<(), StateError> {
        for value in state.states() {
            self.record_for(value.handle())?;
        }
        for value in state.states() {
            self.release(value.handle().clone())
                .expect("all allocation handles were validated");
        }
        Ok(())
    }

    fn release(&mut self, handle: StateHandle) -> Result<(), StateError> {
        self.record_for(&handle)?;
        self.allocations.remove(&handle.id());
        Ok(())
    }
}

pub trait StateManager: Send {
    /// Allocate a coherent bundle with one aggregate grant. Nothing commits or
    /// publishes on failure. The charge persists until its last component releases.
    ///
    /// # Errors
    /// Rejects invalid/overflowing requirements, unsupported placement, a closed
    /// authority or insufficient aggregate capacity.
    fn allocate_set(
        &mut self,
        requirements: &[StateRequirement],
        location: StateLocation,
    ) -> Result<InferenceStateSet, StateError>;

    /// Validate allocation provenance and the current logical prefix.
    ///
    /// # Errors
    ///
    /// Returns an error for stale, released, or foreign allocation handles.
    fn validate(&self, state: &InferenceStateSet) -> Result<(), StateError>;

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

    /// Commit a batch atomically. On error, both manager records and all
    /// supplied state positions must remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid handles, positions, or mismatched lengths.
    fn commit_batch(
        &mut self,
        states: &mut [InferenceStateSet],
        token_positions: &[u32],
    ) -> Result<(), StateError>;

    /// Advance one state set without relinquishing ownership on failure.
    ///
    /// # Errors
    ///
    /// Returns an error when the transition cannot be committed.
    fn commit(
        &mut self,
        state: &mut InferenceStateSet,
        token_position: u32,
    ) -> Result<(), StateError> {
        self.commit_batch(std::slice::from_mut(state), &[token_position])
    }

    /// Release every allocation in a set atomically. On error, retain all
    /// allocations so the caller can retry with the same state owner.
    ///
    /// # Errors
    ///
    /// Returns an error when an allocation cannot be released.
    fn release_set(&mut self, state: &InferenceStateSet) -> Result<(), StateError>;

    /// # Errors
    ///
    /// Returns [`StateError::InvalidHandle`] when the handle is unknown or was released.
    fn release(&mut self, handle: StateHandle) -> Result<(), StateError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocate(manager: &mut LogicalStateManager) -> InferenceStateSet {
        let spec = KvStateSpec::new(1, 1, 2, 4, DataType::F16).expect("spec");
        let kv = manager
            .allocate_kv(spec, StateLocation::Device(DeviceId::new(0)))
            .expect("allocation");
        InferenceStateSet::try_new(Some(kv), None).expect("set")
    }

    fn recurrent_requirement() -> StateRequirement {
        let matrix = RecurrentMatrixShape::new(2, 4, 4).expect("matrix");
        let convolution = ConvolutionStateShape::new(8, 3).expect("convolution");
        StateRequirement::Recurrent(
            RecurrentStateSpec::new(2, matrix, convolution, DataType::F16, DataType::F32)
                .expect("recurrent spec"),
        )
    }

    #[test]
    fn a_declaration_covers_the_concrete_capacity_its_sequences_can_reach() {
        let declared = StateRequirement::FullAttentionKv(
            KvStateSpec::new(2, 4, 8, 64, DataType::F16).expect("declared spec"),
        );
        let concrete = declared.with_capacity(16).expect("concrete spec");
        assert!(concrete.within(declared));
        assert_eq!(
            concrete.byte_size(),
            declared.byte_size().map(|bytes| bytes / 4)
        );
        // Capacity above the bound, a different shape and a different dtype are all
        // still that model's declaration, just not one this state satisfies.
        assert!(!declared.within(concrete));
        assert!(!concrete.within(declared.with_capacity(8).expect("narrower")));
        let wider = StateRequirement::FullAttentionKv(
            KvStateSpec::new(2, 4, 8, 64, DataType::F32).expect("wider dtype"),
        );
        assert!(!concrete.within(wider));
        let deeper = StateRequirement::FullAttentionKv(
            KvStateSpec::new(3, 4, 8, 64, DataType::F16).expect("deeper"),
        );
        assert!(!concrete.within(deeper));
        // A recurrent summary has no capacity dimension: it is the same requirement
        // however far the sequence reached, and it never covers a KV declaration.
        let recurrent = recurrent_requirement();
        assert_eq!(recurrent.with_capacity(16).expect("unchanged"), recurrent);
        assert!(!recurrent.within(declared));
        assert!(!concrete.within(recurrent));
        assert!(recurrent.within(recurrent));
        assert!(matches!(
            declared.with_capacity(0),
            Err(StateSpecError::ZeroDimension)
        ));
    }

    #[test]
    fn bundle_reservation_is_atomic_and_failure_publishes_nothing() {
        let location = StateLocation::Device(DeviceId::new(0));
        let kv =
            StateRequirement::FullAttentionKv(KvStateSpec::new(1, 1, 2, 4, DataType::F16).unwrap());
        let recurrent = recurrent_requirement();
        let total = kv.byte_size().unwrap() + recurrent.byte_size().unwrap();
        let pool = BytePool::new(total - 1).shared();
        let mut manager = LogicalStateManager::new(
            DeviceId::new(0),
            Arc::clone(&pool),
            BytePool::new(0).shared(),
        );
        let wait = pool.capacity_wait();
        assert!(matches!(
            manager.allocate_set(&[kv, recurrent], location),
            Err(StateError::CapacityExceeded { .. })
        ));
        assert_eq!(
            manager.allocate_set(&[kv, kv], location).unwrap_err(),
            StateError::DuplicateRequirement
        );
        assert_eq!(pool.granted(), 0);
        assert!(
            !wait.changed(),
            "neither aggregate refusal nor validation rolls back a partial grant"
        );
        let state = manager.allocate_set(&[kv], location).unwrap();
        manager.release_set(&state).unwrap();
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn bundle_charge_survives_partial_component_release_and_invalid_set_release() {
        let location = StateLocation::Device(DeviceId::new(0));
        let requirements = [
            StateRequirement::FullAttentionKv(KvStateSpec::new(1, 1, 2, 4, DataType::F16).unwrap()),
            recurrent_requirement(),
        ];
        let total = requirements.iter().map(|r| r.byte_size().unwrap()).sum();
        let pool = BytePool::new(total).shared();
        let mut manager = LogicalStateManager::new(
            DeviceId::new(0),
            Arc::clone(&pool),
            BytePool::new(0).shared(),
        );
        let state = manager.allocate_set(&requirements, location).unwrap();
        let wait = pool.capacity_wait();
        manager
            .release(state.kv().unwrap().handle().clone())
            .unwrap();
        assert_eq!(
            pool.granted(),
            total,
            "one bundle grant stays with its surviving component"
        );
        assert!(!wait.changed());
        assert_eq!(manager.release_set(&state), Err(StateError::InvalidHandle));
        assert_eq!(pool.granted(), total);
        manager
            .release(state.recurrent().unwrap().handle().clone())
            .unwrap();
        assert_eq!(pool.granted(), 0);
        assert!(wait.changed());
    }

    #[test]
    fn supplied_authority_counts_siblings_and_closes_without_refunding_live_state() {
        let device = DeviceId::new(0);
        let location = StateLocation::Device(device);
        let pool = BytePool::new(64).shared();
        let host = BytePool::new(0).shared();
        let mut first = LogicalStateManager::new(device, Arc::clone(&pool), Arc::clone(&host));
        let mut peer = LogicalStateManager::new(device, Arc::clone(&pool), host);
        let state = allocate(&mut first);
        let sibling = pool.reserve(32).unwrap();
        assert_eq!(first.used_bytes(location), Some(64));
        assert_eq!(peer.used_bytes(location), Some(64));
        let wait = peer.capacity_wait(location).unwrap();
        assert!(matches!(
            peer.allocate_set(&state.requirements(), location),
            Err(StateError::CapacityExceeded { .. })
        ));
        assert!(!wait.changed());
        drop(sibling);
        assert!(wait.changed());
        let peer_state = peer.allocate_set(&state.requirements(), location).unwrap();
        assert_ne!(
            peer_state.kv().unwrap().handle().id(),
            state.kv().unwrap().handle().id()
        );
        let wait = peer.capacity_wait(location).unwrap();
        pool.close();
        assert!(wait.changed());
        assert_eq!(pool.granted(), 64);
        assert_eq!(
            peer.allocate_set(&state.requirements(), location)
                .unwrap_err(),
            StateError::PoolClosed
        );
        assert_eq!(peer.release_set(&state), Err(StateError::InvalidHandle));
        first.release_set(&state).unwrap();
        peer.release_set(&peer_state).unwrap();
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn dropping_an_unretired_logical_owner_does_not_refund_its_shared_charge() {
        let device = DeviceId::new(0);
        let pool = BytePool::new(32).shared();
        let mut manager =
            LogicalStateManager::new(device, Arc::clone(&pool), BytePool::new(0).shared());
        let state = allocate(&mut manager);
        drop(state);
        drop(manager);
        assert_eq!(
            pool.granted(),
            32,
            "logical disappearance cannot prove physical reuse"
        );
        assert!(matches!(
            pool.reserve(1),
            Err(ReserveError::Exhausted { .. })
        ));
    }

    #[test]
    fn independent_managers_reject_each_others_allocations() {
        let mut first = LogicalStateManager::new(
            DeviceId::new(0),
            ribn_foundation::BytePool::new(1024).shared(),
            ribn_foundation::BytePool::new(0).shared(),
        );
        let mut second = LogicalStateManager::new(
            DeviceId::new(0),
            ribn_foundation::BytePool::new(1024).shared(),
            ribn_foundation::BytePool::new(0).shared(),
        );
        let mut first_state = allocate(&mut first);
        let second_state = allocate(&mut second);
        assert_ne!(
            first_state.states()[0].handle().id(),
            second_state.states()[0].handle().id()
        );
        assert_eq!(
            second.commit(&mut first_state, 1),
            Err(StateError::InvalidHandle)
        );
        assert_eq!(
            second.release_set(&first_state),
            Err(StateError::InvalidHandle)
        );
        second
            .validate(&second_state)
            .expect("own allocation remains valid");
        first
            .validate(&first_state)
            .expect("foreign rejection leaves owner valid");
        first.release_set(&first_state).unwrap();
        second.release_set(&second_state).unwrap();
    }

    #[test]
    fn batch_commit_prevalidates_all_states_before_advancing_any() {
        let mut manager = LogicalStateManager::new(
            DeviceId::new(0),
            ribn_foundation::BytePool::new(1024).shared(),
            ribn_foundation::BytePool::new(0).shared(),
        );
        let first = allocate(&mut manager);
        let mut second = allocate(&mut manager);
        manager.commit(&mut second, 3).expect("advance second");
        let mut states = vec![first, second];
        assert_eq!(
            manager.commit_batch(&mut states, &[1, 2]),
            Err(StateError::PositionRegression)
        );
        assert_eq!(states[0].token_position(), Some(0));
        assert_eq!(states[1].token_position(), Some(3));
        for state in &states {
            manager.validate(state).expect("unchanged record");
        }
        manager.commit_batch(&mut states, &[1, 4]).expect("retry");
        assert_eq!(states[0].token_position(), Some(1));
        assert_eq!(states[1].token_position(), Some(4));
        for state in &states {
            manager.release_set(state).unwrap();
        }
    }
}
