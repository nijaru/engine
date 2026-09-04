from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label}: target not found")
    return text.replace(old, new, 1)


execution = Path("crates/core/src/execution.rs")
s = execution.read_text()
s = replace_once(
    s,
    "    state_requirements: Vec<StateRequirement>,\n",
    "    state_requirements: Arc<[StateRequirement]>,\n",
    "segment state field",
)
s = replace_once(
    s,
    "            state_requirements,\n            token_input: None,\n",
    "            state_requirements: state_requirements.into(),\n            token_input: None,\n",
    "segment vec constructor",
)
marker = "    /// Attach the token payload consumed by this segment.\n"
shared_ctor = '''    /// Construct a segment over an already shared state-requirement set.\n    ///\n    /// This is the steady-state serving constructor: prepared plans can share\n    /// one immutable requirement array across every request/iteration instead\n    /// of allocating a new `Vec` for each segment.\n    ///\n    /// # Errors\n    ///\n    /// Returns [`PlanError::InvalidSegmentBatchSize`] unless `batch_size` is\n    /// one or [`PlanError::ZeroWork`] when `token_count` is zero.\n    pub fn new_shared(\n        request: RequestId,\n        phase: ExecutionPhase,\n        batch_size: u32,\n        token_count: u32,\n        state_position: u32,\n        state_requirements: Arc<[StateRequirement]>,\n    ) -> Result<Self, PlanError> {\n        if batch_size != 1 {\n            return Err(PlanError::InvalidSegmentBatchSize);\n        }\n        if token_count == 0 {\n            return Err(PlanError::ZeroWork);\n        }\n        Ok(Self {\n            request,\n            phase,\n            token_count,\n            state_position,\n            state_requirements,\n            token_input: None,\n            sampling: None,\n        })\n    }\n\n'''
s = replace_once(s, marker, shared_ctor + marker, "shared segment constructor")
# Only the ExecutionPlan field remains as Vec after replacing the first occurrence.
s = replace_once(
    s,
    "    state_requirements: Vec<StateRequirement>,\n    weights: WeightBinding,\n",
    "    state_requirements: Arc<[StateRequirement]>,\n    weights: WeightBinding,\n",
    "plan state field",
)
s = replace_once(
    s,
    "            state_requirements,\n            weights,\n            residency,\n",
    "            state_requirements: state_requirements.into(),\n            weights,\n            residency,\n",
    "plan vec constructor",
)
marker = '''    #[must_use]\n    pub fn state_requirements(&self) -> &[StateRequirement] {\n        &self.state_requirements\n    }\n'''
replacement = '''    #[must_use]\n    pub fn state_requirements(&self) -> &[StateRequirement] {\n        &self.state_requirements\n    }\n\n    /// Clone the immutable prepared state schema without copying its entries.\n    #[must_use]\n    pub fn shared_state_requirements(&self) -> Arc<[StateRequirement]> {\n        Arc::clone(&self.state_requirements)\n    }\n'''
s = replace_once(s, marker, replacement, "plan shared requirement getter")
execution.write_text(s)

serving = Path("crates/core/src/serving_runtime.rs")
s = serving.read_text()
s = replace_once(
    s,
    '''                let mut segment = ExecutionSegment::new(\n                    item.request(),\n                    item.phase(),\n                    1,\n                    item.token_count(),\n                    item.state_position(),\n                    self.plan.state_requirements().to_vec(),\n                )?;\n''',
    '''                let mut segment = ExecutionSegment::new_shared(\n                    item.request(),\n                    item.phase(),\n                    1,\n                    item.token_count(),\n                    item.state_position(),\n                    self.plan.shared_state_requirements(),\n                )?;\n''',
    "serving shared requirement path",
)
serving.write_text(s)
