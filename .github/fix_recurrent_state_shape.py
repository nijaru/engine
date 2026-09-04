from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


# Core recurrent state is a bank of matrices, not a Cartesian product of
# model q/k and v head counts. Name the actual storage axes so byte accounting
# and backend allocation cannot accidentally over-allocate Qwen GDN state.
p = Path("crates/core/src/state.rs")
s = p.read_text()
old = '''#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
'''
new = '''/// Storage geometry for a bank of recurrent state matrices.
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
'''
s = replace_once(s, old, new, "core recurrent shape types")
s = replace_once(
    s,
    '''        let matrix_elements = u64::from(self.matrix.key_heads())
            .checked_mul(u64::from(self.matrix.key_head_dim()))?
            .checked_mul(u64::from(self.matrix.value_heads()))?
            .checked_mul(u64::from(self.matrix.value_head_dim()))?
            .checked_mul(u64::from(self.layer_count))?;
        let convolution_elements = u64::from(self.convolution.channels())
            .checked_mul(u64::from(self.convolution.kernel()))?
            .checked_mul(u64::from(self.layer_count))?;
''',
    '''        let matrix_elements = u64::from(self.matrix.matrix_count())
            .checked_mul(u64::from(self.matrix.rows()))?
            .checked_mul(u64::from(self.matrix.columns()))?
            .checked_mul(u64::from(self.layer_count))?;
        let convolution_elements = u64::from(self.convolution.channels())
            .checked_mul(u64::from(self.convolution.history_tokens()))?
            .checked_mul(u64::from(self.layer_count))?;
''',
    "core recurrent byte accounting",
)
p.write_text(s)

# Make the Qwen provider describe the actual recurrent state used by the
# verified llama.cpp equations: 48 independent 128x128 state matrices and
# d_conv-1 (=3) history values for each convolution channel.
p = Path("crates/gguf/src/lib.rs")
s = p.read_text()
s = replace_once(
    s,
    '''    let matrix = RecurrentMatrixShape::new(
        narrow_u16(config.ssm_group_count(), "recurrent key head count")?,
        narrow_u16(config.ssm_state_size(), "recurrent key head dimension")?,
        narrow_u16(config.ssm_time_step_rank(), "recurrent value head count")?,
        narrow_u16(
            config.ssm_inner_size() / config.ssm_time_step_rank(),
            "recurrent value head dimension",
        )?,
    )
''',
    '''    let matrix = RecurrentMatrixShape::new(
        narrow_u16(config.ssm_time_step_rank(), "recurrent matrix count")?,
        narrow_u16(config.ssm_state_size(), "recurrent matrix row dimension")?,
        narrow_u16(
            config.ssm_inner_size() / config.ssm_time_step_rank(),
            "recurrent matrix column dimension",
        )?,
    )
''',
    "Qwen recurrent matrix geometry",
)
s = replace_once(
    s,
    '''    let recurrent_spec = RecurrentStateSpec::new(
        narrow_u16(recurrent_layers, "recurrent layer count")?,
        matrix,
        ConvolutionStateShape::new(
            convolution_channels,
            narrow_u16(config.ssm_conv_kernel(), "recurrent convolution kernel")?,
        )
''',
    '''    let convolution_history = config
        .ssm_conv_kernel()
        .checked_sub(1)
        .ok_or(GgufError::InvalidModelConfiguration(
            "recurrent convolution kernel has no history",
        ))?;
    let recurrent_spec = RecurrentStateSpec::new(
        narrow_u16(recurrent_layers, "recurrent layer count")?,
        matrix,
        ConvolutionStateShape::new(
            convolution_channels,
            narrow_u16(convolution_history, "recurrent convolution history")?,
        )
''',
    "Qwen recurrent convolution history",
)
# Strengthen the existing model-description test so the provider and CUDA
# executor cannot silently diverge again.
marker = '''        assert_eq!(description.state_requirements().len(), 2);
'''
addition = '''        assert_eq!(description.state_requirements().len(), 2);
        let recurrent = description
            .state_requirements()
            .iter()
            .find_map(|requirement| match requirement {
                StateRequirement::Recurrent(spec) => Some(*spec),
                StateRequirement::FullAttentionKv(_) => None,
            })
            .expect("recurrent state");
        assert_eq!(recurrent.matrix().matrix_count(), 48);
        assert_eq!(recurrent.matrix().rows(), 128);
        assert_eq!(recurrent.matrix().columns(), 128);
        assert_eq!(recurrent.convolution().channels(), 10_240);
        assert_eq!(recurrent.convolution().history_tokens(), 3);
'''
s = replace_once(s, marker, addition, "Qwen state geometry assertions")
p.write_text(s)

# CUDA storage follows the same three-axis matrix bank and explicit history
# length. This removes the accidental 16x state multiplication.
p = Path("crates/nvidia/src/state.rs")
s = p.read_text()
s = s.replace(
    '    /// Per-layer `[k_heads][k_dim][v_heads][v_dim]` state matrices.\n',
    '    /// Per-layer `[matrix][row][column]` recurrent state matrices.\n',
)
s = s.replace(
    '    /// Per-layer `[channels][kernel - 1]` convolution histories.\n',
    '    /// Per-layer `[channel][history_token]` convolution histories.\n',
)
s = s.replace(
    '''            [
                u64::from(matrix_shape.key_heads()),
                u64::from(matrix_shape.key_head_dim()),
                u64::from(matrix_shape.value_heads()),
                u64::from(matrix_shape.value_head_dim()),
            ],
''',
    '''            [
                u64::from(matrix_shape.matrix_count()),
                u64::from(matrix_shape.rows()),
                u64::from(matrix_shape.columns()),
            ],
''',
)
s = s.replace(
    '                u64::from(spec.convolution().kernel()),\n',
    '                u64::from(spec.convolution().history_tokens()),\n',
)
s = s.replace(
    '''        [
            u64::from(matrix_shape.key_heads()),
            u64::from(matrix_shape.key_head_dim()),
            u64::from(matrix_shape.value_heads()),
            u64::from(matrix_shape.value_head_dim()),
        ],
''',
    '''        [
            u64::from(matrix_shape.matrix_count()),
            u64::from(matrix_shape.rows()),
            u64::from(matrix_shape.columns()),
        ],
''',
)
s = s.replace(
    '            u64::from(convolution_shape.kernel()),\n',
    '            u64::from(convolution_shape.history_tokens()),\n',
)
p.write_text(s)

# Decode validation now speaks the storage semantics directly.
p = Path("crates/nvidia/src/decode.rs")
s = p.read_text()
s = replace_once(
    s,
    '''                || usize::from(matrix.key_heads()) != 1
                || usize::from(matrix.key_head_dim()) != GDN_HEAD_DIM
                || usize::from(matrix.value_heads()) != GDN_V_HEADS
                || usize::from(matrix.value_head_dim()) != GDN_HEAD_DIM
                || u64::from(convolution.channels()) != GDN_QKV_DIM as u64
                || usize::from(convolution.kernel()) != GDN_D_CONV - 1
''',
    '''                || usize::from(matrix.matrix_count()) != GDN_V_HEADS
                || usize::from(matrix.rows()) != GDN_HEAD_DIM
                || usize::from(matrix.columns()) != GDN_HEAD_DIM
                || u64::from(convolution.channels()) != GDN_QKV_DIM as u64
                || usize::from(convolution.history_tokens()) != GDN_D_CONV - 1
''',
    "CUDA recurrent geometry validation",
)
p.write_text(s)

# Update the handful of construction sites to the corrected 3-axis shape.
replacements = {
    "crates/core/src/lib.rs": [
        ("RecurrentMatrixShape::new(16, 128, 48, 128)", "RecurrentMatrixShape::new(48, 128, 128)"),
        ("ConvolutionStateShape::new(10_240, 4)", "ConvolutionStateShape::new(10_240, 3)"),
    ],
    "crates/nvidia/examples/qwen_decode_bench.rs": [
        ("RecurrentMatrixShape::new(1, 128, 48, 128)", "RecurrentMatrixShape::new(48, 128, 128)"),
    ],
    "crates/nvidia/tests/cuda_reference.rs": [
        ("RecurrentMatrixShape::new(1, 2, 2, 2)", "RecurrentMatrixShape::new(2, 2, 2)"),
        ("RecurrentMatrixShape::new(1, 128, 48, 128)", "RecurrentMatrixShape::new(48, 128, 128)"),
    ],
}
for filename, changes in replacements.items():
    p = Path(filename)
    s = p.read_text()
    for old, new in changes:
        if old not in s:
            raise SystemExit(f"{filename}: missing {old}")
        s = s.replace(old, new)
    p.write_text(s)

# EOS can coincide with the request's max-output transition. Only explicitly
# finish a request that the scheduler has not already terminalized.
p = Path("crates/server/src/local.rs")
s = p.read_text()
s = replace_once(
    s,
    '''        if reached_eos && u32::try_from(output.len()).unwrap_or(u32::MAX) < max_tokens {
            serving.finish(request).map_err(|error| error.to_string())?;
        }
''',
    '''        if reached_eos && serving.scheduler().counts().terminal() == 0 {
            serving.finish(request).map_err(|error| error.to_string())?;
        }
''',
    "EOS terminal transition",
)
# max_tokens is enforced by the scheduler; the generate helper no longer needs
# it after fixing the terminal-state check.
s = s.replace(
    '''        tokenizer.eos_token_id(),
        options.max_tokens,
    )?;
''',
    '''        tokenizer.eos_token_id(),
    )?;
''',
    1,
)
s = s.replace(
    '''    eos_token: u32,
    max_tokens: u32,
) -> Result<Vec<u32>, String>
''',
    '''    eos_token: u32,
) -> Result<Vec<u32>, String>
''',
    1,
)
p.write_text(s)
