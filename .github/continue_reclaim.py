from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


# A slot may expose its terminal state to the serving owner only after it has
# already been removed from RequestSlots. This keeps live-slot ownership intact.
p = Path("crates/core/src/serving.rs")
s = p.read_text()
s = replace_once(
    s,
    """    #[must_use]
    pub const fn progress(&self) -> RequestProgress {
        self.progress
    }

    /// # Errors
""",
    """    #[must_use]
    pub const fn progress(&self) -> RequestProgress {
        self.progress
    }

    /// Take the state owned by a terminal slot after scheduler reclamation.
    ///
    /// This is crate-private so live request state cannot be detached through
    /// the public serving API. [`RequestSlots::remove`] verifies terminal state
    /// ownership before returning the slot to its serving owner.
    ///
    /// # Errors
    ///
    /// Returns [`RequestSlotError::InvalidTransition`] for a non-terminal slot
    /// or [`RequestSlotError::StateUnavailable`] when state was already taken.
    pub(crate) fn take_terminal_state(&mut self) -> Result<InferenceStateSet, RequestSlotError> {
        if !self.lifecycle.is_terminal() {
            return Err(RequestSlotError::InvalidTransition);
        }
        self.state.take().ok_or(RequestSlotError::StateUnavailable)
    }

    /// # Errors
""",
    "terminal state accessor",
)
p.write_text(s)

# Runtime owns the StateManager, so it is the only layer that should release
# logical state handles after serving has finished with a request.
p = Path("crates/core/src/runtime.rs")
s = p.read_text()
s = replace_once(
    s,
    """    #[must_use]
    pub const fn state_manager_mut(&mut self) -> &mut S {
        &mut self.state_manager
    }

    /// Submit a batch while preserving caller-owned logical state if model,
""",
    """    #[must_use]
    pub const fn state_manager_mut(&mut self) -> &mut S {
        &mut self.state_manager
    }

    /// Release every logical state allocation owned by one finished request.
    ///
    /// Physical backend state remains a backend concern; this closes the core
    /// allocation lifecycle so request reclamation cannot leak StateManager
    /// capacity.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::State`] when a state handle is invalid or its
    /// manager cannot release it.
    pub fn release_state_set(&mut self, state: &InferenceStateSet) -> Result<(), RuntimeError> {
        for value in state.states() {
            self.state_manager.release(value.handle().clone())?;
        }
        Ok(())
    }

    /// Submit a batch while preserving caller-owned logical state if model,
""",
    "runtime release state set",
)
p.write_text(s)

# Serving removes terminal scheduler bookkeeping, releases the underlying
# logical state allocations, then drops prompt/decode-frontier payloads.
p = Path("crates/core/src/serving_runtime.rs")
s = p.read_text()
s = replace_once(
    s,
    """    pub fn reclaim_next(&mut self) -> Result<Option<ActiveRequestSlot>, ServingRuntimeError> {
        let reclaimed = self.scheduler.reclaim_next()?;
        if let Some(slot) = &reclaimed {
            let request = slot.request().id();
            self.prompt_tokens.remove(&request);
            self.next_tokens.remove(&request);
        }
        Ok(reclaimed)
    }
""",
    """    pub fn reclaim_next(&mut self) -> Result<Option<ActiveRequestSlot>, ServingRuntimeError> {
        let Some(mut reclaimed) = self.scheduler.reclaim_next()? else {
            return Ok(None);
        };
        let request = reclaimed.request().id();
        let state = reclaimed.take_terminal_state()?;
        self.runtime.release_state_set(&state)?;
        self.prompt_tokens.remove(&request);
        self.next_tokens.remove(&request);
        Ok(Some(reclaimed))
    }
""",
    "serving reclaim release",
)
# Existing failure test should now observe that returned reclaimed metadata no
# longer carries live state handles.
s = replace_once(
    s,
    """        assert!(
            serving
                .reclaim_next()
                .expect("reclaim")
                .expect("terminal request")
                .state()
                .is_some()
        );
""",
    """        assert!(
            serving
                .reclaim_next()
                .expect("reclaim")
                .expect("terminal request")
                .state()
                .is_none()
        );
""",
    "reclaim test expectation",
)
marker = """    #[test]
    fn failed_backend_submit_restores_states_before_terminalizing() {
"""
test = """    #[test]
    fn terminal_reclaim_releases_logical_state_capacity() {
        let mut serving = fixture(false);
        let device = DeviceId::new(0);
        let spec = crate::state::KvStateSpec::new(1, 1, 2, 4, crate::tensor::DataType::F16)
            .expect("KV spec");
        let kv = serving
            .runtime_mut()
            .state_manager_mut()
            .allocate_kv(spec, crate::state::StateLocation::Device(device))
            .expect("KV allocation");
        let state = InferenceStateSet::try_new(Some(kv), None).expect("state set");
        assert!(
            serving
                .runtime()
                .state_manager()
                .used_bytes(crate::state::StateLocation::Device(device))
                .is_some_and(|bytes| bytes > 0)
        );

        let request_id = RequestId::new(1).expect("request ID");
        serving
            .admit(request(1), state, prompt(&[11]))
            .expect("admit");
        serving.cancel(request_id).expect("cancel");
        let reclaimed = serving
            .reclaim_next()
            .expect("reclaim")
            .expect("terminal request");

        assert!(reclaimed.state().is_none());
        assert_eq!(
            serving
                .runtime()
                .state_manager()
                .used_bytes(crate::state::StateLocation::Device(device)),
            Some(0)
        );
    }

"""
if test not in s:
    s = replace_once(s, marker, test + marker, "state reclamation test")
p.write_text(s)
