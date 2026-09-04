from pathlib import Path

p = Path("crates/nvidia/src/decode.rs")
s = p.read_text()
old = '''        self.ops
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
'''
new = '''        self.ops
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
'''
if old not in s:
    raise SystemExit("greedy host-copy target not found")
p.write_text(s.replace(old, new, 1))
