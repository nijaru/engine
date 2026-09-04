from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


decode = Path("crates/nvidia/src/decode.rs")
s = decode.read_text()
old = '''    /// Run one full forward pass for `token` at absolute sequence
    /// `position` and return the greedy token choice.
    ///
    /// All kernel launches are stream-ordered; the returned token comes
    /// from a synchronized argmax, so state mutations from this step are
    /// visible to subsequent steps.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when the physical state does not match
    /// the layer plan geometry, the position exceeds the KV capacity, or
    /// any launch fails.
    pub fn decode_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<u32, CudaDecodeError> {
'''
new = '''    /// Advance model state for one known prompt token without producing logits.
    ///
    /// Intermediate prefill tokens do not need an output projection or sampled
    /// token: the next input is already supplied by the prompt. This path
    /// therefore skips output RMSNorm, vocabulary projection, argmax, and the
    /// device-to-host synchronization. Launches remain ordered on the decoder
    /// stream, so a following prefill/decode step observes the updated state.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when the physical state does not match the
    /// layer plan geometry, the position exceeds KV capacity, or a launch fails.
    pub fn prefill_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
        self.run_step(state, token, position)
    }

    /// Run one full forward pass for `token` at absolute sequence `position`
    /// and return the greedy token choice.
    ///
    /// The output-token readback is an intentional host synchronization on the
    /// current host-driven autoregressive path.
    ///
    /// # Errors
    ///
    /// Returns [`CudaDecodeError`] when state/geometry is invalid or execution
    /// or output selection fails.
    pub fn decode_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<u32, CudaDecodeError> {
        self.run_step(state, token, position)?;
        self.select_greedy_token()
    }

    fn run_step(
        &mut self,
        state: &mut CudaHybridState,
        token: u32,
        position: u32,
    ) -> Result<(), CudaDecodeError> {
'''
s = replace_once(s, old, new, "decode step split")
old = '''        self.ops.rms_norm(
            &self.hidden,
            f32_slice(&self.weights, "output_norm.weight")?,
            &mut self.normed,
            self.epsilon,
        )?;
        gemv(
            &self.weights,
            "output.weight",
            &self.normed,
            &mut self.logits,
        )?;
        self.ops
            .argmax_into(&self.logits, &mut self.selected_token)?;
        // The host needs the selected token before it can schedule the next
        // autoregressive step, so this remains an intentional blocking
        // completion boundary. Copy into stack storage rather than allocating
        // a new Vec for every token.
        let mut selected = [0_u32; 1];
        self.stream
            .memcpy_dtoh(&self.selected_token, &mut selected)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        Ok(selected[0])
    }
'''
new = '''        Ok(())
    }

    fn select_greedy_token(&mut self) -> Result<u32, CudaDecodeError> {
        self.ops.rms_norm(
            &self.hidden,
            f32_slice(&self.weights, "output_norm.weight")?,
            &mut self.normed,
            self.epsilon,
        )?;
        gemv(
            &self.weights,
            "output.weight",
            &self.normed,
            &mut self.logits,
        )?;
        self.ops
            .argmax_into(&self.logits, &mut self.selected_token)?;
        // The host needs the selected token before it can schedule the next
        // autoregressive step, so this remains an intentional blocking
        // completion boundary. Copy into stack storage rather than allocating
        // a new Vec for every token.
        let mut selected = [0_u32; 1];
        self.stream
            .memcpy_dtoh(&self.selected_token, &mut selected)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        Ok(selected[0])
    }
'''
s = replace_once(s, old, new, "separate output head")
decode.write_text(s)

serving = Path("crates/nvidia/src/serving.rs")
s = serving.read_text()
old = '''                    let chosen = self
                        .executor
                        .decode_step(physical, token, position)
                        .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                    let next = position.checked_add(1).ok_or_else(|| {
                        BackendError::ExecutionFailed("prefill position overflowed".to_owned())
                    })?;
                    physical
                        .advance_to(next)
                        .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                    if segment.requests_sampling() && offset + 1 == segment.token_count() {
                        output_token = Some(chosen);
                    }
'''
new = '''                    let requests_output =
                        segment.requests_sampling() && offset + 1 == segment.token_count();
                    if requests_output {
                        let chosen = self
                            .executor
                            .decode_step(physical, token, position)
                            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                        output_token = Some(chosen);
                    } else {
                        self.executor
                            .prefill_step(physical, token, position)
                            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
                    }
                    let next = position.checked_add(1).ok_or_else(|| {
                        BackendError::ExecutionFailed("prefill position overflowed".to_owned())
                    })?;
                    physical
                        .advance_to(next)
                        .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
'''
s = replace_once(s, old, new, "serving output-free prefill")
serving.write_text(s)

bench = Path("crates/nvidia/examples/qwen_decode_bench.rs")
s = bench.read_text()
old = '''    let mut chosen = 0_u32;
    for (position, token) in PROMPT.iter().enumerate() {
        chosen = executor
            .decode_step(
                &mut state,
                *token,
                u32::try_from(position).expect("fits u32"),
            )
            .expect("prefill step");
    }
'''
new = '''    let mut chosen = 0_u32;
    let last_prompt_index = PROMPT.len() - 1;
    for (position, token) in PROMPT.iter().enumerate() {
        let position_u32 = u32::try_from(position).expect("fits u32");
        if position == last_prompt_index {
            chosen = executor
                .decode_step(&mut state, *token, position_u32)
                .expect("final prefill step");
        } else {
            executor
                .prefill_step(&mut state, *token, position_u32)
                .expect("prefill step");
        }
    }
'''
s = replace_once(s, old, new, "benchmark output-free prefill")
bench.write_text(s)
