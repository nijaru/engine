use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use engine_core::{
    DataType, DeviceId, HybridStateSet, KvState, KvStateSpec, RecurrentState, RecurrentStateSpec,
    StateLocation,
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
/// buffers even when they share one allocation shape.
pub struct CudaKvState {
    spec: KvStateSpec,
    keys: CudaStateBuffer,
    values: CudaStateBuffer,
}

impl CudaKvState {
    #[must_use]
    pub const fn spec(&self) -> KvStateSpec {
        self.spec
    }

    #[must_use]
    pub fn keys(&self) -> &CudaStateBuffer {
        &self.keys
    }

    pub fn keys_mut(&mut self) -> &mut CudaStateBuffer {
        &mut self.keys
    }

    #[must_use]
    pub fn values(&self) -> &CudaStateBuffer {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut CudaStateBuffer {
        &mut self.values
    }

    #[must_use]
    pub fn token_capacity(&self) -> u32 {
        self.spec.block_tokens()
    }
}

/// Physical recurrent/Gated-DeltaNet state. The matrix and convolution
/// history have independent buffers and are never represented as KV pages.
pub struct CudaRecurrentState {
    spec: RecurrentStateSpec,
    matrix: CudaStateBuffer,
    convolution: CudaStateBuffer,
}

impl CudaRecurrentState {
    #[must_use]
    pub const fn spec(&self) -> RecurrentStateSpec {
        self.spec
    }

    #[must_use]
    pub fn matrix(&self) -> &CudaStateBuffer {
        &self.matrix
    }

    pub fn matrix_mut(&mut self) -> &mut CudaStateBuffer {
        &mut self.matrix
    }

    #[must_use]
    pub fn convolution(&self) -> &CudaStateBuffer {
        &self.convolution
    }

    pub fn convolution_mut(&mut self) -> &mut CudaStateBuffer {
        &mut self.convolution
    }
}

/// Backend-owned physical counterpart to a core [`HybridStateSet`]. It is a
/// storage primitive only: core handles remain authoritative for identity,
/// placement, and lifecycle, while this value owns device allocations.
pub struct CudaHybridState {
    stream: Arc<CudaStream>,
    kv: Option<CudaKvState>,
    recurrent: Option<CudaRecurrentState>,
    token_position: u32,
}

impl CudaHybridState {
    /// Allocate physical buffers matching the families present in a core state
    /// set. Host-resident state is rejected because this type owns CUDA device
    /// storage; a later transfer tier can use a separate owner.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError`] when placement, dtype, dimensions, or CUDA
    /// allocation is unsupported.
    pub fn from_state_set(
        stream: Arc<CudaStream>,
        state: &HybridStateSet,
    ) -> Result<Self, CudaStateError> {
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
        let physical = Self::from_specs(
            stream,
            state.kv().map(KvState::spec),
            state.recurrent().map(RecurrentState::spec),
        )?;
        Ok(Self {
            token_position: state.token_position().unwrap_or(0),
            ..physical
        })
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

    /// Clear both state families before a new request/prefix is attached.
    ///
    /// # Errors
    ///
    /// Returns [`CudaStateError::Driver`] when the device memset fails.
    pub fn zero(&mut self) -> Result<(), CudaStateError> {
        if let Some(kv) = &mut self.kv {
            kv.keys.zero(&self.stream)?;
            kv.values.zero(&self.stream)?;
        }
        if let Some(recurrent) = &mut self.recurrent {
            recurrent.matrix.zero(&self.stream)?;
            recurrent.convolution.zero(&self.stream)?;
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

fn allocate_kv(stream: &Arc<CudaStream>, spec: KvStateSpec) -> Result<CudaKvState, CudaStateError> {
    let elements = checked_product(
        [
            u64::from(spec.layer_count()),
            u64::from(spec.kv_heads()),
            u64::from(spec.head_dim()),
            u64::from(spec.block_tokens()),
        ],
        "full-attention KV",
    )?;
    let keys = allocate_buffer(stream, spec.dtype(), elements, "full-attention KV keys")?;
    let values = allocate_buffer(stream, spec.dtype(), elements, "full-attention KV values")?;
    Ok(CudaKvState { spec, keys, values })
}

fn allocate_recurrent(
    stream: &Arc<CudaStream>,
    spec: RecurrentStateSpec,
) -> Result<CudaRecurrentState, CudaStateError> {
    let matrix_shape = spec.matrix();
    let matrix_elements = checked_product(
        [
            u64::from(spec.layer_count()),
            u64::from(matrix_shape.key_heads()),
            u64::from(matrix_shape.key_head_dim()),
            u64::from(matrix_shape.value_heads()),
            u64::from(matrix_shape.value_head_dim()),
        ],
        "recurrent matrix",
    )?;
    let convolution_shape = spec.convolution();
    let convolution_elements = checked_product(
        [
            u64::from(spec.layer_count()),
            u64::from(convolution_shape.channels()),
            u64::from(convolution_shape.kernel()),
        ],
        "recurrent convolution history",
    )?;
    let matrix = allocate_buffer(
        stream,
        spec.matrix_dtype(),
        matrix_elements,
        "recurrent matrix",
    )?;
    let convolution = allocate_buffer(
        stream,
        spec.convolution_dtype(),
        convolution_elements,
        "recurrent convolution history",
    )?;
    Ok(CudaRecurrentState {
        spec,
        matrix,
        convolution,
    })
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
    PositionRegression {
        current: u32,
        requested: u32,
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
            Self::PositionRegression { current, requested } => write!(
                f,
                "physical state position cannot move from {current} back to {requested}"
            ),
        }
    }
}

impl std::error::Error for CudaStateError {}
