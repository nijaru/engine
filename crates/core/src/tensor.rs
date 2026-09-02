//! Tensor and weight metadata shared by model providers and backends.

use std::fmt;

/// Scalar types that a backend may expose for model weights or state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DataType {
    F16,
    BF16,
    F32,
    I8,
    U8,
}

/// Weight quantization is explicit because similarly named formats are not
/// interchangeable for correctness or benchmark comparisons.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Quantization {
    None,
    GgufQ4Km,
    CompressedTensorsW4A16 { group_size: u32 },
    Other,
}

/// The loader/provider boundary keeps checkpoint representation out of the
/// execution and state APIs.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum WeightFormat {
    Gguf,
    Safetensors,
    Vendor(String),
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::I8 => "i8",
            Self::U8 => "u8",
        };
        f.write_str(name)
    }
}
