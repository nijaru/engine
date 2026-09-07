use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use engine_core::{
    DataType, DeviceId, InferenceStateSet, KvState, KvStateSpec, RecurrentState,
    RecurrentStateSpec, StateId, StateLocation,
};

/// A typed device allocation for one inference-state family component.
/// Keeping the scalar representation visible lets a future kernel consume the
/// correct state dtype without treating recurrent state as KV storage.
pub enum CudaStateBuffer {
    F16(CudaSlice<u16>),
    F32(CudaSlice<f32>),
}

impl CudaStateBuffer {
    #[must_use]
    pub const fn dtype(&self) -> DataType {
        match self {
            Self::F16(_) => DataType::F16,
            Self::F32(_) => DataType::F32,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F16(buffer) => buffer.len(),
            Self::F32(buffer) => buffer.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The F16 device data, when this buffer is an F16 allocation.
    #[must_use]
    pub fn as_f16(&self) -> Option<&CudaSlice<u16>> {
        match self {
            Self::F16(buffer) => Some(buffer),
            Self::F32(_) => None,
        }
    }

    /// The mutable F16 device data, when this buffer is an F16 allocation.
    #[must_use]
    pub fn as_f16_mut(&mut self) -> Option<&mut CudaSlice<u16>> {
        match self {
            Self::F16(buffer) => Some(buffer),
            Self::F32(_) => None,
        }
    }

    /// The F32 device data, when this buffer is an F32 allocation.
    #[must_use]
    pub fn as_f32(&self) -> Option<&CudaSlice<f32>> {
        match self {
            Self::F32(buffer) => Some(buffer),
            Self::F16(_) => None,
        }
    }

    /// The mutable F32 device data, when this buffer is an F32 allocation.
    #[must_use]
    pub fn as_f32_mut(&mut self) -> Option<&mut CudaSlice<f32>> {
        match self {
            Self::F32(buffer) => Some(buffer),
            Self::F16(_) => None,
        }
    }

    #[must_use]
    pub fn byte_size(&self) -> Option<u64> {
        u64::try_from(self.len())
            .ok()?
            .checked_mul(self.dtype().byte_width())
    }

    fn zero(&mut self, stream: &Arc<CudaStream>) -> Result<(), CudaStateError> {
        match self {
            Self::F16(buffer) => stream
                .memset_zeros(buffer)
                .map_err(|error| CudaStateError::Driver(error.to_string())),
            Self::F32(buffer) => stream
                .memset_zeros(buffer)
                .map_err(|error| CudaStateError::Driver(error.to_string())),
        }
    }
}

/// Physical full-attention KV state. Keys and values remain separate device
/// buffers even when they share one allocation shape; each layer owns its
/// pair so kernels address one layer's cache without offset arithmetic.
pub struct CudaKvState {
    spec: KvStateSpec,
    /// Per-layer `[token][kv_head][head_dim]` key caches.
    keys: Vec<CudaStateBuffer>,
    /// Per-layer `[token][kv_head][head_dim]` value caches.
    values: Vec<CudaStateBuffer>,
}

impl CudaKvState {
    #[must_use]
    pub const fn spec(&self) -> KvStateSpec {
        self.spec
    }

    /// The device key cache for one full-attention layer.
    #[must_use]
    pub fn layer_keys(&self, layer: u32) -> Option<&CudaStateBuffer> {
        self.keys.get(usize::try_from(layer).ok()?)
    }

    /// The device value cache for one full-attention layer.
    #[must_use]
    pub fn layer_values(&self, layer: u32) -> Option<&CudaStateBuffer> {
        self.values.get(usize::try_from(layer).ok()?)
    }

    /// The mutable key and value caches for one full-attention layer.
    pub fn layer_mut(
        &mut self,
        layer: u32,
    ) -> Option<(&mut CudaStateBuffer, &mut CudaStateBuffer)> {
        let index = usize::try_from(layer).ok()?;
        let keys = self.keys.get_mut(index)?;
        let values = self.values.get_mut(index)?;
        Some((keys, values))
    }

    #[must_use]
    pub fn token_capacity(&self) -> u32 {
        self.spec.block_tokens()
    }

    #[must_use]
    pub fn byte_size(&self) -> Option<u64> {
        self.keys
            .iter()
            .try_fold(0_u64, |sum, buffer| sum.checked_add(buffer.byte_size()?))?
            .checked_add(
                self.values
                    .iter()
                    .try_fold(0_u64, |sum, buffer| sum.checked_add(buffer.byte_size()?))?,
            )
    }

    /// Write one token's keys and values into a token-major KV block. The
    /// caller commits the shared sequence position after all model regions
    /// have written their state for that token.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when the layer/token or vector shape is
    /// invalid, the state is not F16, or the device copy fails.
    pub fn write_token(
        &mut self,
        layer: u32,
        token: u32,
        keys: &[u16],
        values: &[u16],
        stream: &Arc<CudaStream>,
    ) -> Result<(), CudaStateError> {
        let spec = self.spec;
        let layer = usize::try_from(layer).map_err(|_| CudaStateError::IndexOverflow {
            family: "full-attention KV",
        })?;
        let token = usize::try_from(token).map_err(|_| CudaStateError::IndexOverflow {
            family: "full-attention KV",
        })?;
        let layers = usize::from(spec.layer_count());
        let block_tokens =
            usize::try_from(spec.block_tokens()).map_err(|_| CudaStateError::IndexOverflow {
                family: "full-attention KV",
            })?;
        let token_width = usize::from(spec.kv_heads())
            .checked_mul(usize::from(spec.head_dim()))
            .ok_or(CudaStateError::SizeOverflow {
                family: "full-attention KV token",
            })?;
        if layer >= layers {
            return Err(CudaStateError::IndexOutOfBounds {
                family: "full-attention KV layer",
                index: layer,
                length: layers,
            });
        }
        if token >= block_tokens {
            return Err(CudaStateError::IndexOutOfBounds {
                family: "full-attention KV token",
                index: token,
                length: block_tokens,
            });
        }
        if keys.len() != token_width || values.len() != token_width {
            return Err(CudaStateError::ShapeMismatch {
                family: "full-attention KV token",
                expected: token_width,
                actual: keys.len().max(values.len()),
            });
        }
        if spec.dtype() != DataType::F16 {
            return Err(CudaStateError::UnsupportedDtype {
                family: "full-attention KV token write",
                dtype: spec.dtype(),
            });
        }
        let token_offset = token
            .checked_mul(token_width)
            .ok_or(CudaStateError::SizeOverflow {
                family: "full-attention KV token offset",
            })?;
        let key_layers = self.keys.len();
        let layer_keys = self
            .keys
            .get_mut(layer)
            .ok_or(CudaStateError::IndexOutOfBounds {
                family: "full-attention KV layer",
                index: layer,
                length: key_layers,
            })?;
        let value_layers = self.values.len();
        let layer_values = self
            .values
            .get_mut(layer)
            .ok_or(CudaStateError::IndexOutOfBounds {
                family: "full-attention KV layer",
                index: layer,
                length: value_layers,
            })?;
        copy_u16_to_offset(
            stream,
            layer_keys,
            token_offset,
            keys,
            "full-attention KV keys",
        )?;
        copy_u16_to_offset(
            stream,
            layer_values,
            token_offset,
            values,
            "full-attention KV values",
        )
    }
}

/// Physical recurrent/Gated-DeltaNet state. The matrix and convolution
/// history have independent buffers and are never represented as KV pages.
pub struct CudaRecurrentState {
    spec: RecurrentStateSpec,
    /// Per-layer `[matrix][row][column]` recurrent state matrices.
    matrix: Vec<CudaStateBuffer>,
    /// Per-layer `[channel][history_token]` convolution histories.
    convolution: Vec<CudaStateBuffer>,
}

impl CudaRecurrentState {
    #[must_use]
    pub const fn spec(&self) -> RecurrentStateSpec {
        self.spec
    }

    /// The device state matrix for one recurrent layer.
    #[must_use]
    pub fn layer_matrix(&self, layer: u32) -> Option<&CudaStateBuffer> {
        self.matrix.get(usize::try_from(layer).ok()?)
    }

    /// The device convolution history for one recurrent layer.
    #[must_use]
    pub fn layer_convolution(&self, layer: u32) -> Option<&CudaStateBuffer> {
        self.convolution.get(usize::try_from(layer).ok()?)
    }

    /// The mutable matrix and convolution history for one recurrent layer.
    pub fn layer_mut(
        &mut self,
        layer: u32,
    ) -> Option<(&mut CudaStateBuffer, &mut CudaStateBuffer)> {
        let index = usize::try_from(layer).ok()?;
        let matrix = self.matrix.get_mut(index)?;
        let convolution = self.convolution.get_mut(index)?;
        Some((matrix, convolution))
    }

    #[must_use]
    pub fn byte_size(&self) -> Option<u64> {
        self.matrix
            .iter()
            .try_fold(0_u64, |sum, buffer| sum.checked_add(buffer.byte_size()?))?
            .checked_add(
                self.convolution
                    .iter()
                    .try_fold(0_u64, |sum, buffer| sum.checked_add(buffer.byte_size()?))?,
            )
    }

    /// Write one recurrent layer's matrix and convolution state. Each slice is
    /// a contiguous layer in the shape declared by [`RecurrentStateSpec`].
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when the layer or shape is invalid, the
    /// declared state dtypes are not F32, or the device copy fails.
    pub fn write_layer(
        &mut self,
        layer: u32,
        matrix: &[f32],
        convolution: &[f32],
        stream: &Arc<CudaStream>,
    ) -> Result<(), CudaStateError> {
        let spec = self.spec;
        let layer = usize::try_from(layer).map_err(|_| CudaStateError::IndexOverflow {
            family: "recurrent state",
        })?;
        let layers = usize::from(spec.layer_count());
        if layer >= layers {
            return Err(CudaStateError::IndexOutOfBounds {
                family: "recurrent state layer",
                index: layer,
                length: layers,
            });
        }
        if spec.matrix_dtype() != DataType::F32 {
            return Err(CudaStateError::UnsupportedDtype {
                family: "recurrent matrix write",
                dtype: spec.matrix_dtype(),
            });
        }
        if spec.convolution_dtype() != DataType::F32 {
            return Err(CudaStateError::UnsupportedDtype {
                family: "recurrent convolution write",
                dtype: spec.convolution_dtype(),
            });
        }
        let matrix_shape = spec.matrix();
        let matrix_width = checked_product(
            [
                u64::from(matrix_shape.matrix_count()),
                u64::from(matrix_shape.rows()),
                u64::from(matrix_shape.columns()),
            ],
            "recurrent matrix layer",
        )?;
        let convolution_width = checked_product(
            [
                u64::from(spec.convolution().channels()),
                u64::from(spec.convolution().history_tokens()),
            ],
            "recurrent convolution layer",
        )?;
        let matrix_width =
            usize::try_from(matrix_width).map_err(|_| CudaStateError::HostSizeOverflow {
                family: "recurrent matrix layer",
                elements: matrix_width,
            })?;
        let convolution_width =
            usize::try_from(convolution_width).map_err(|_| CudaStateError::HostSizeOverflow {
                family: "recurrent convolution layer",
                elements: convolution_width,
            })?;
        if matrix.len() != matrix_width {
            return Err(CudaStateError::ShapeMismatch {
                family: "recurrent matrix layer",
                expected: matrix_width,
                actual: matrix.len(),
            });
        }
        if convolution.len() != convolution_width {
            return Err(CudaStateError::ShapeMismatch {
                family: "recurrent convolution layer",
                expected: convolution_width,
                actual: convolution.len(),
            });
        }
        let matrix_layers = self.matrix.len();
        let layer_matrix = self
            .matrix
            .get_mut(layer)
            .ok_or(CudaStateError::IndexOutOfBounds {
                family: "recurrent state layer",
                index: layer,
                length: matrix_layers,
            })?;
        copy_f32_to_offset(stream, layer_matrix, 0, matrix, "recurrent matrix")?;
        let convolution_layers = self.convolution.len();
        let layer_convolution =
            self.convolution
                .get_mut(layer)
                .ok_or(CudaStateError::IndexOutOfBounds {
                    family: "recurrent state layer",
                    index: layer,
                    length: convolution_layers,
                })?;
        copy_f32_to_offset(
            stream,
            layer_convolution,
            0,
            convolution,
            "recurrent convolution history",
        )
    }
}

/// Identity of one logical state set, used to key backend-owned physical
/// state. Sorted [`StateId`] handles rather than request IDs keep scheduling
/// identity and model-state identity separate.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CudaStateKey(Vec<StateId>);

impl CudaStateKey {
    /// Derive the registry key for one logical state set.
    #[must_use]
    pub fn from_state_set(state: &InferenceStateSet) -> Self {
        let mut ids = state
            .states()
            .iter()
            .map(|value| value.handle().id())
            .collect::<Vec<_>>();
        ids.sort_unstable_by_key(|id| id.get());
        Self(ids)
    }
}

/// Persistent backend ownership for physical CUDA request state.
///
/// Logical [`StateId`] values are the identity boundary. Request IDs are not
/// used because scheduling/lifecycle identity and model state identity are
/// deliberately separate in core.
pub struct CudaStateRegistry {
    stream: Arc<CudaStream>,
    states: HashMap<CudaStateKey, CudaHybridState>,
}

impl CudaStateRegistry {
    #[must_use]
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            states: HashMap::new(),
        }
    }

    /// Materialize physical state on first use and reuse it on later steps.
    ///
    /// Fresh materialization is valid only at prefix zero. A future state
    /// restore/reuse path must supply the real KV and recurrent history for a
    /// nonzero logical prefix instead of allocating zeroed buffers and merely
    /// copying the logical token position.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when CUDA allocation fails, a nonzero prefix
    /// has no physical materialization, or the persistent physical prefix no
    /// longer matches the authoritative logical prefix.
    pub fn get_or_create(
        &mut self,
        state: &InferenceStateSet,
    ) -> Result<&mut CudaHybridState, CudaStateError> {
        let key = CudaStateKey::from_state_set(state);
        let expected = state.token_position().unwrap_or(0);
        if !self.states.contains_key(&key) {
            validate_fresh_materialization(state)?;
            let physical = CudaHybridState::from_state_set(self.stream.clone(), state)?;
            self.states.insert(key.clone(), physical);
        }
        let physical = self
            .states
            .get_mut(&key)
            .ok_or(CudaStateError::RegistryInvariant)?;
        if physical.token_position() != expected {
            return Err(CudaStateError::PositionMismatch {
                logical: expected,
                physical: physical.token_position(),
            });
        }
        Ok(physical)
    }

    /// Drop any physical allocation associated with this logical state set.
    /// Missing state is a valid no-op for requests cancelled before first use.
    pub fn release(&mut self, state: &InferenceStateSet) -> bool {
        self.release_key(&CudaStateKey::from_state_set(state))
    }

    /// Drop any physical allocation associated with a state key.
    /// Missing state is a valid no-op for requests cancelled before first use.
    pub fn release_key(&mut self, key: &CudaStateKey) -> bool {
        self.states.remove(key).is_some()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Remove the physical state for `key` and return it, if materialized.
    ///
    /// Batched execution needs several states borrowed mutably at once,
    /// which the one-at-a-time [`Self::get_or_create`] cannot express. The
    /// caller must [`Self::reinsert`] every removed state before returning
    /// to normal operation so no physical state is lost.
    pub fn take(&mut self, key: &CudaStateKey) -> Option<CudaHybridState> {
        self.states.remove(key)
    }

    /// Put a previously [`Self::take`]-removed physical state back. Returns
    /// the state untouched if an entry for its key somehow already exists,
    /// keeping the registry the single owner.
    pub fn reinsert(&mut self, key: CudaStateKey, state: CudaHybridState) {
        self.states.entry(key).or_insert(state);
    }
}

/// Backend-owned physical counterpart to a core [`InferenceStateSet`]. It is a
/// storage primitive only: core handles remain authoritative for identity,
/// placement, and lifecycle, while this value owns device allocations.
pub struct CudaHybridState {
    stream: Arc<CudaStream>,
    kv: Option<CudaKvState>,
    recurrent: Option<CudaRecurrentState>,
    token_position: u32,
}

impl CudaHybridState {
    /// Write one full-attention layer/token KV pair into physical state.
    /// State position is unchanged until [`Self::advance_to`] commits the
    /// completed model step.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when no KV state exists or the typed write
    /// validation/device copy fails.
    pub fn write_kv_token(
        &mut self,
        layer: u32,
        token: u32,
        keys: &[u16],
        values: &[u16],
    ) -> Result<(), CudaStateError> {
        let Some(kv) = &mut self.kv else {
            return Err(CudaStateError::MissingFamily {
                family: "full-attention KV",
            });
        };
        kv.write_token(layer, token, keys, values, &self.stream)
    }

    /// Write one recurrent layer's matrix and convolution state. State
    /// position is unchanged until [`Self::advance_to`] commits the completed
    /// model step.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when no recurrent state exists or the typed
    /// write validation/device copy fails.
    pub fn write_recurrent_layer(
        &mut self,
        layer: u32,
        matrix: &[f32],
        convolution: &[f32],
    ) -> Result<(), CudaStateError> {
        let Some(recurrent) = &mut self.recurrent else {
            return Err(CudaStateError::MissingFamily {
                family: "recurrent state",
            });
        };
        recurrent.write_layer(layer, matrix, convolution, &self.stream)
    }

    /// Allocate fresh physical buffers matching a core state set at prefix zero.
    /// Host-resident state is rejected because this type owns CUDA device
    /// storage; a later transfer tier can use a separate owner. Nonzero-prefix
    /// restoration requires a separate path carrying actual physical history.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when placement, prefix, dtype, dimensions, or
    /// CUDA allocation is unsupported.
    pub fn from_state_set(
        stream: Arc<CudaStream>,
        state: &InferenceStateSet,
    ) -> Result<Self, CudaStateError> {
        validate_fresh_materialization(state)?;
        if let Some(location) = state.location() {
            let StateLocation::Device(device) = location else {
                return Err(CudaStateError::UnsupportedLocation);
            };
            let actual = u16::try_from(stream.context().ordinal()).unwrap_or(u16::MAX);
            if device.get() != actual {
                return Err(CudaStateError::DeviceMismatch {
                    requested: device,
                    actual: DeviceId::new(actual),
                });
            }
        }
        Self::from_specs(
            stream,
            state.kv().map(KvState::spec),
            state.recurrent().map(RecurrentState::spec),
        )
    }

    /// Allocate physical buffers directly from typed state specifications.
    /// This constructor is useful while a dispatcher is being assembled; the
    /// core-handle-aware [`Self::from_state_set`] should be preferred once a
    /// request owns its state.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when a dtype, dimension, or CUDA allocation
    /// is unsupported.
    pub fn from_specs(
        stream: Arc<CudaStream>,
        kv: Option<KvStateSpec>,
        recurrent: Option<RecurrentStateSpec>,
    ) -> Result<Self, CudaStateError> {
        let kv = kv.map(|spec| allocate_kv(&stream, spec)).transpose()?;
        let recurrent = recurrent
            .map(|spec| allocate_recurrent(&stream, spec))
            .transpose()?;
        Ok(Self {
            stream,
            kv,
            recurrent,
            token_position: 0,
        })
    }

    #[must_use]
    pub fn kv(&self) -> Option<&CudaKvState> {
        self.kv.as_ref()
    }

    pub fn kv_mut(&mut self) -> Option<&mut CudaKvState> {
        self.kv.as_mut()
    }

    #[must_use]
    pub fn recurrent(&self) -> Option<&CudaRecurrentState> {
        self.recurrent.as_ref()
    }

    pub fn recurrent_mut(&mut self) -> Option<&mut CudaRecurrentState> {
        self.recurrent.as_mut()
    }

    #[must_use]
    pub const fn token_position(&self) -> u32 {
        self.token_position
    }

    #[must_use]
    pub fn byte_size(&self) -> Option<u64> {
        let kv_bytes = self
            .kv
            .as_ref()
            .and_then(CudaKvState::byte_size)
            .unwrap_or(0);
        let recurrent_bytes = self
            .recurrent
            .as_ref()
            .and_then(CudaRecurrentState::byte_size)
            .unwrap_or(0);
        kv_bytes.checked_add(recurrent_bytes)
    }

    /// Clear both state families before a new request/prefix is attached.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError::Driver`] when the device memset fails.
    pub fn zero(&mut self) -> Result<(), CudaStateError> {
        if let Some(kv) = &mut self.kv {
            for buffer in &mut kv.keys {
                buffer.zero(&self.stream)?;
            }
            for buffer in &mut kv.values {
                buffer.zero(&self.stream)?;
            }
        }
        if let Some(recurrent) = &mut self.recurrent {
            for buffer in &mut recurrent.matrix {
                buffer.zero(&self.stream)?;
            }
            for buffer in &mut recurrent.convolution {
                buffer.zero(&self.stream)?;
            }
        }
        self.stream
            .synchronize()
            .map_err(|error| CudaStateError::Driver(error.to_string()))
    }

    /// Advance the physical prefix boundary without permitting a regression.
    /// Core state-manager commit remains the authoritative lifecycle update;
    /// this mirror protects backend callers from accidentally reusing a stale
    /// device position.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError::PositionRegression`] when the requested
    /// position is older than the current boundary.
    pub fn advance_to(&mut self, token_position: u32) -> Result<(), CudaStateError> {
        if token_position < self.token_position {
            return Err(CudaStateError::PositionRegression {
                current: self.token_position,
                requested: token_position,
            });
        }
        self.token_position = token_position;
        Ok(())
    }
}

fn validate_fresh_materialization(state: &InferenceStateSet) -> Result<(), CudaStateError> {
    let position = state.token_position().unwrap_or(0);
    if position == 0 {
        Ok(())
    } else {
        Err(CudaStateError::UnmaterializedPrefix { position })
    }
}

fn allocate_kv(stream: &Arc<CudaStream>, spec: KvStateSpec) -> Result<CudaKvState, CudaStateError> {
    // Each layer's cache is a contiguous [token][kv_head][head_dim] buffer,
    // making one token contiguous for state append and attention reads.
    let layer_elements = checked_product(
        [
            u64::from(spec.block_tokens()),
            u64::from(spec.kv_heads()),
            u64::from(spec.head_dim()),
        ],
        "full-attention KV layer",
    )?;
    let layer_count = usize::from(spec.layer_count());
    let mut keys = Vec::with_capacity(layer_count);
    let mut values = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        keys.push(allocate_buffer(
            stream,
            spec.dtype(),
            layer_elements,
            "full-attention KV keys",
        )?);
        values.push(allocate_buffer(
            stream,
            spec.dtype(),
            layer_elements,
            "full-attention KV values",
        )?);
    }
    Ok(CudaKvState { spec, keys, values })
}

fn allocate_recurrent(
    stream: &Arc<CudaStream>,
    spec: RecurrentStateSpec,
) -> Result<CudaRecurrentState, CudaStateError> {
    let matrix_shape = spec.matrix();
    let matrix_elements = checked_product(
        [
            u64::from(matrix_shape.matrix_count()),
            u64::from(matrix_shape.rows()),
            u64::from(matrix_shape.columns()),
        ],
        "recurrent matrix",
    )?;
    let convolution_shape = spec.convolution();
    let convolution_elements = checked_product(
        [
            u64::from(convolution_shape.channels()),
            u64::from(convolution_shape.history_tokens()),
        ],
        "recurrent convolution history",
    )?;
    let layer_count = usize::from(spec.layer_count());
    let mut matrix = Vec::with_capacity(layer_count);
    let mut convolution = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        matrix.push(allocate_buffer(
            stream,
            spec.matrix_dtype(),
            matrix_elements,
            "recurrent matrix",
        )?);
        convolution.push(allocate_buffer(
            stream,
            spec.convolution_dtype(),
            convolution_elements,
            "recurrent convolution history",
        )?);
    }
    Ok(CudaRecurrentState {
        spec,
        matrix,
        convolution,
    })
}

fn copy_u16_to_offset(
    stream: &Arc<CudaStream>,
    destination: &mut CudaStateBuffer,
    offset: usize,
    source: &[u16],
    family: &'static str,
) -> Result<(), CudaStateError> {
    let CudaStateBuffer::F16(destination) = destination else {
        return Err(CudaStateError::UnsupportedDtype {
            family,
            dtype: DataType::F32,
        });
    };
    copy_slice_to_offset(stream, destination, offset, source, family)
}

fn copy_f32_to_offset(
    stream: &Arc<CudaStream>,
    destination: &mut CudaStateBuffer,
    offset: usize,
    source: &[f32],
    family: &'static str,
) -> Result<(), CudaStateError> {
    let CudaStateBuffer::F32(destination) = destination else {
        return Err(CudaStateError::UnsupportedDtype {
            family,
            dtype: DataType::F16,
        });
    };
    copy_slice_to_offset(stream, destination, offset, source, family)
}

fn copy_slice_to_offset<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    destination: &mut CudaSlice<T>,
    offset: usize,
    source: &[T],
    family: &'static str,
) -> Result<(), CudaStateError> {
    let Some(end) = offset.checked_add(source.len()) else {
        return Err(CudaStateError::SizeOverflow { family });
    };
    let Some(mut view) = destination.try_slice_mut(offset..end) else {
        return Err(CudaStateError::IndexOutOfBounds {
            family,
            index: end,
            length: destination.len(),
        });
    };
    stream
        .memcpy_htod(source, &mut view)
        .map_err(|error| CudaStateError::Driver(error.to_string()))
}

fn checked_product(
    values: impl IntoIterator<Item = u64>,
    family: &'static str,
) -> Result<u64, CudaStateError> {
    values.into_iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(value)
            .ok_or(CudaStateError::SizeOverflow { family })
    })
}

fn allocate_buffer(
    stream: &Arc<CudaStream>,
    dtype: DataType,
    elements: u64,
    family: &'static str,
) -> Result<CudaStateBuffer, CudaStateError> {
    let elements = usize::try_from(elements)
        .map_err(|_| CudaStateError::HostSizeOverflow { family, elements })?;
    match dtype {
        DataType::F16 => stream
            .alloc_zeros::<u16>(elements)
            .map(CudaStateBuffer::F16)
            .map_err(|error| CudaStateError::Allocation {
                family,
                message: error.to_string(),
            }),
        DataType::F32 => stream
            .alloc_zeros::<f32>(elements)
            .map(CudaStateBuffer::F32)
            .map_err(|error| CudaStateError::Allocation {
                family,
                message: error.to_string(),
            }),
        dtype => Err(CudaStateError::UnsupportedDtype { family, dtype }),
    }
}

#[derive(Debug)]
pub enum CudaStateError {
    UnsupportedLocation,
    DeviceMismatch {
        requested: DeviceId,
        actual: DeviceId,
    },
    UnsupportedDtype {
        family: &'static str,
        dtype: DataType,
    },
    SizeOverflow {
        family: &'static str,
    },
    HostSizeOverflow {
        family: &'static str,
        elements: u64,
    },
    Allocation {
        family: &'static str,
        message: String,
    },
    Driver(String),
    UnmaterializedPrefix {
        position: u32,
    },
    PositionRegression {
        current: u32,
        requested: u32,
    },
    PositionMismatch {
        logical: u32,
        physical: u32,
    },
    RegistryInvariant,
    MissingFamily {
        family: &'static str,
    },
    IndexOverflow {
        family: &'static str,
    },
    IndexOutOfBounds {
        family: &'static str,
        index: usize,
        length: usize,
    },
    ShapeMismatch {
        family: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for CudaStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedLocation => {
                f.write_str("CUDA physical state requires a device-resident core state")
            }
            Self::DeviceMismatch { requested, actual } => write!(
                f,
                "state requests {requested}, but CUDA stream belongs to {actual}"
            ),
            Self::UnsupportedDtype { family, dtype } => {
                write!(f, "{family} state dtype {dtype} is not supported")
            }
            Self::SizeOverflow { family } => write!(f, "{family} state size overflowed"),
            Self::HostSizeOverflow { family, elements } => write!(
                f,
                "{family} state element count {elements} does not fit this host"
            ),
            Self::Allocation { family, message } => {
                write!(f, "could not allocate {family} state: {message}")
            }
            Self::Driver(message) => write!(f, "CUDA state driver error: {message}"),
            Self::UnmaterializedPrefix { position } => write!(
                f,
                "cannot allocate fresh CUDA state at nonzero prefix {position} without restoring physical history"
            ),
            Self::PositionRegression { current, requested } => write!(
                f,
                "physical state position cannot move from {current} back to {requested}"
            ),
            Self::PositionMismatch { logical, physical } => write!(
                f,
                "logical state is at position {logical}, but persistent CUDA state is at {physical}"
            ),
            Self::RegistryInvariant => f.write_str("CUDA state registry lost a materialized entry"),
            Self::MissingFamily { family } => write!(f, "physical state has no {family} family"),
            Self::IndexOverflow { family } => write!(f, "{family} index overflowed"),
            Self::IndexOutOfBounds {
                family,
                index,
                length,
            } => write!(f, "{family} index {index} is outside length {length}"),
            Self::ShapeMismatch {
                family,
                expected,
                actual,
            } => write!(f, "{family} has {actual} elements, expected {expected}"),
        }
    }
}

impl std::error::Error for CudaStateError {}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{LogicalStateManager, StateManager};

    #[test]
    fn fresh_materialization_rejects_nonzero_logical_prefix() {
        let device = DeviceId::new(0);
        let spec = KvStateSpec::new(1, 1, 2, 8, DataType::F16).expect("KV spec");
        let mut manager = LogicalStateManager::new(device, 1024, 0);
        let kv = manager
            .allocate_kv(spec, StateLocation::Device(device))
            .expect("logical KV state");
        let mut zero = InferenceStateSet::try_new(Some(kv), None).expect("state set");
        assert!(validate_fresh_materialization(&zero).is_ok());

        manager
            .commit(&mut zero, 3)
            .expect("advance logical prefix");
        assert!(matches!(
            validate_fresh_materialization(&zero),
            Err(CudaStateError::UnmaterializedPrefix { position: 3 })
        ));
    }
}
