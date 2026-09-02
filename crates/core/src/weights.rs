//! Prepared model-weight metadata and bounded scalar streams.
//!
//! The core describes which model tensors a prepared execution may use, but
//! does not own checkpoint parsing or device allocations. Format adapters can
//! expose a tensor as a bounded stream of canonical `f32` blocks, while a
//! backend owns the physical materialization.

use std::fmt;

use crate::device::DeviceId;
use crate::model::ModelId;
use crate::tensor::DataType;

/// Metadata for one tensor in the stream's scalar ordering.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WeightTensorSpec {
    name: String,
    dimensions: Vec<u64>,
    dtype: DataType,
    element_count: u64,
}

impl WeightTensorSpec {
    /// # Errors
    ///
    /// Returns [`WeightSpecError`] when the name or shape is invalid or the
    /// element count overflows.
    pub fn new(
        name: impl Into<String>,
        dimensions: Vec<u64>,
        dtype: DataType,
    ) -> Result<Self, WeightSpecError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(WeightSpecError::EmptyName);
        }
        if dimensions.is_empty() {
            return Err(WeightSpecError::EmptyShape);
        }
        if dimensions.contains(&0) {
            return Err(WeightSpecError::ZeroDimension);
        }
        let element_count = dimensions
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
            .ok_or(WeightSpecError::ElementCountOverflow)?;
        Ok(Self {
            name,
            dimensions,
            dtype,
            element_count,
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Dimensions are expressed in the stream's scalar ordering. A format
    /// adapter must not silently present serialized dimensions as a different
    /// execution layout.
    #[must_use]
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    #[must_use]
    pub const fn dtype(&self) -> DataType {
        self.dtype
    }

    #[must_use]
    pub const fn element_count(&self) -> u64 {
        self.element_count
    }

    #[must_use]
    pub fn byte_size(&self) -> Option<u64> {
        self.element_count.checked_mul(self.dtype.byte_width())
    }
}

/// A bounded source of canonical F32 tensor blocks.
///
/// Implementations must return exactly `spec().element_count()` scalar values
/// before returning `None`. Blocks may be any non-zero size; consumers enforce
/// the declared total and own the destination allocation.
pub trait F32BlockStream {
    type Error;

    fn spec(&self) -> &WeightTensorSpec;

    /// Return the next non-empty block, or `None` after the declared tensor is
    /// exhausted.
    ///
    /// # Errors
    ///
    /// Returns the source-specific error when the next block cannot be read or
    /// decoded.
    fn next_block(&mut self) -> Result<Option<Vec<f32>>, Self::Error>;
}

/// Logical model/device association for a prepared set of weight tensors.
/// Physical buffers remain owned by the selected backend.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WeightBinding {
    model: ModelId,
    device: DeviceId,
    tensors: Vec<WeightTensorSpec>,
}

impl WeightBinding {
    /// # Errors
    ///
    /// Returns [`WeightBindingError::DuplicateTensor`] when two entries have
    /// the same tensor name.
    pub fn new(
        model: ModelId,
        device: DeviceId,
        tensors: Vec<WeightTensorSpec>,
    ) -> Result<Self, WeightBindingError> {
        for (index, tensor) in tensors.iter().enumerate() {
            if tensors[..index]
                .iter()
                .any(|previous| previous.name() == tensor.name())
            {
                return Err(WeightBindingError::DuplicateTensor(
                    tensor.name().to_owned(),
                ));
            }
        }
        Ok(Self {
            model,
            device,
            tensors,
        })
    }

    #[must_use]
    pub fn empty(model: ModelId, device: DeviceId) -> Self {
        Self {
            model,
            device,
            tensors: Vec::new(),
        }
    }

    #[must_use]
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    #[must_use]
    pub fn tensors(&self) -> &[WeightTensorSpec] {
        &self.tensors
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&WeightTensorSpec> {
        self.tensors.iter().find(|tensor| tensor.name() == name)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WeightSpecError {
    EmptyName,
    EmptyShape,
    ZeroDimension,
    ElementCountOverflow,
}

impl fmt::Display for WeightSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => f.write_str("weight tensor name must not be empty"),
            Self::EmptyShape => f.write_str("weight tensor shape must not be empty"),
            Self::ZeroDimension => f.write_str("weight tensor dimensions must be non-zero"),
            Self::ElementCountOverflow => f.write_str("weight tensor element count overflowed"),
        }
    }
}

impl std::error::Error for WeightSpecError {}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum WeightBindingError {
    DuplicateTensor(String),
}

impl fmt::Display for WeightBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateTensor(name) => {
                write!(f, "weight binding contains duplicate tensor {name:?}")
            }
        }
    }
}

impl std::error::Error for WeightBindingError {}
