from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


# Split the greedy selector into an asynchronous launch that writes caller-owned
# device storage and the existing blocking convenience wrapper. The serving
# decode path can then reuse one output slot instead of allocating device memory
# on every token.
p = Path("crates/nvidia/src/model_ops.rs")
s = p.read_text()
old = '''    /// This is a blocking host read because token selection is the boundary
    /// between device logits and the next request token. The launch itself is
    /// submitted asynchronously before the result is copied back.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when the context, input, or launch is
    /// invalid.
    pub fn argmax(&self, logits: &CudaSlice<f32>) -> Result<u32, CudaModelKernelError> {
        if self.stream.context().as_ref() != logits.context().as_ref() {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if logits.is_empty() {
            return Err(CudaModelKernelError::EmptyInput);
        }
        let length =
            u32::try_from(logits.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let mut selected = self
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        // Safety: cudarc allocated both slices, the length is checked, and the
        // single-thread launch writes exactly one result element.
        unsafe {
            self.stream
                .launch_builder(&self.argmax)
                .arg(logits)
                .arg(&mut selected)
                .arg(&length)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let selected = self
            .stream
            .clone_dtoh(&selected)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        selected
            .first()
            .copied()
            .ok_or(CudaModelKernelError::EmptyInput)
    }
'''
new = '''    /// Launch greedy token selection into one caller-owned device result slot.
    ///
    /// The launch is asynchronous with respect to the host. Keeping result
    /// storage caller-owned lets a long-lived decoder reuse it across steps
    /// instead of allocating device memory in the steady-state token path.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when the context, input/output shape,
    /// or launch is invalid.
    pub fn argmax_into(
        &self,
        logits: &CudaSlice<f32>,
        selected: &mut CudaSlice<u32>,
    ) -> Result<(), CudaModelKernelError> {
        if self.stream.context().as_ref() != logits.context().as_ref()
            || self.stream.context().as_ref() != selected.context().as_ref()
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        if logits.is_empty() {
            return Err(CudaModelKernelError::EmptyInput);
        }
        if selected.len() != 1 {
            return Err(CudaModelKernelError::OutputLength {
                expected: 1,
                actual: selected.len(),
            });
        }
        let length =
            u32::try_from(logits.len()).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        // Safety: cudarc allocated both slices, lengths are checked, and the
        // single-thread launch writes exactly one result element.
        unsafe {
            self.stream
                .launch_builder(&self.argmax)
                .arg(logits)
                .arg(selected)
                .arg(&length)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    /// Select the greedy token and return it to the host.
    ///
    /// This convenience path is blocking because token selection crosses the
    /// device/host boundary. Long-lived execution should prefer
    /// [`Self::argmax_into`] with reusable result storage.
    ///
    /// # Errors
    ///
    /// Returns [`CudaModelKernelError`] when allocation, launch, synchronization,
    /// or the device-to-host copy fails.
    pub fn argmax(&self, logits: &CudaSlice<f32>) -> Result<u32, CudaModelKernelError> {
        let mut selected = self
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        self.argmax_into(logits, &mut selected)?;
        self.stream
            .synchronize()
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let selected = self
            .stream
            .clone_dtoh(&selected)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        selected
            .first()
            .copied()
            .ok_or(CudaModelKernelError::EmptyInput)
    }
'''
s = replace_once(s, old, new, "argmax reusable output API")
p.write_text(s)

p = Path("crates/nvidia/src/decode.rs")
s = p.read_text()
s = replace_once(
    s,
    '''    ffn_act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    /// Attention score scratch, `[q_heads][token capacity]`, allocated on
''',
    '''    ffn_act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    selected_token: CudaSlice<u32>,
    /// Attention score scratch, `[q_heads][token capacity]`, allocated on
''',
    "decode selected-token field",
)
s = replace_once(
    s,
    '''        let scratch = alloc_scratch(&stream, vocab)?;

        Ok(Self {
''',
    '''        let scratch = alloc_scratch(&stream, vocab)?;
        let selected_token = stream
            .alloc_zeros::<u32>(1)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;

        Ok(Self {
''',
    "decode selected-token allocation",
)
s = replace_once(
    s,
    '''            ffn_act: scratch.ffn_act,
            logits: scratch.logits,
        })
''',
    '''            ffn_act: scratch.ffn_act,
            logits: scratch.logits,
            selected_token,
        })
''',
    "decode selected-token initialization",
)
s = replace_once(
    s,
    '''        gemv(
            &self.weights,
            "output.weight",
            &self.normed,
            &mut self.logits,
        )?;
        Ok(self.ops.argmax(&self.logits)?)
    }
''',
    '''        gemv(
            &self.weights,
            "output.weight",
            &self.normed,
            &mut self.logits,
        )?;
        self.ops
            .argmax_into(&self.logits, &mut self.selected_token)?;
        self.stream
            .synchronize()
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        let selected = self
            .stream
            .clone_dtoh(&self.selected_token)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        selected.first().copied().ok_or_else(|| {
            CudaDecodeError::InvalidPlan("greedy token result slot was empty".to_owned())
        })
    }
''',
    "decode reusable argmax output",
)
p.write_text(s)

# Exercise the reusable API independently of the blocking convenience wrapper
# on CUDA qualification hosts.
p = Path("crates/nvidia/tests/cuda_reference.rs")
s = p.read_text()
old = '''    assert_eq!(
        ops.argmax(&logits_device).expect("select greedy token"),
        401
    );
}
'''
new = '''    assert_eq!(
        ops.argmax(&logits_device).expect("select greedy token"),
        401
    );
    let mut selected = stream.alloc_zeros::<u32>(1).expect("argmax output");
    ops.argmax_into(&logits_device, &mut selected)
        .expect("launch reusable argmax");
    stream.synchronize().expect("argmax synchronization");
    assert_eq!(
        stream.clone_dtoh(&selected).expect("read argmax output"),
        vec![401]
    );
}
'''
s = replace_once(s, old, new, "reusable argmax CUDA test")
p.write_text(s)
