use engine_core::{
    DeviceId, InferenceState, InferenceStateSet, LogicalStateManager, StateError, StateLocation,
    StateManager, StateRequirement,
};

/// Allocate the old Qwen bundle transactionally while it remains in use.
/// A rejected admission must not leave even a logical capacity reservation.
pub(crate) fn allocate(
    manager: &mut LogicalStateManager,
    requirements: &[StateRequirement],
) -> Result<InferenceStateSet, StateError> {
    let location = StateLocation::Device(manager.device());
    let mut states: Vec<InferenceState> = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let result = match *requirement {
            StateRequirement::FullAttentionKv(spec) => manager
                .allocate_kv(spec, location)
                .map(InferenceState::from),
            StateRequirement::Recurrent(spec) => manager
                .allocate_recurrent(spec, location)
                .map(InferenceState::from),
        };
        match result {
            Ok(state) => states.push(state),
            Err(error) => {
                for state in &states {
                    manager
                        .release(state.handle().clone())
                        .expect("owned fresh allocation");
                }
                return Err(error);
            }
        }
    }
    let handles = states
        .iter()
        .map(|state| state.handle().clone())
        .collect::<Vec<_>>();
    match InferenceStateSet::new(states) {
        Ok(set) => Ok(set),
        Err(error) => {
            for handle in handles {
                manager.release(handle).expect("owned fresh allocation");
            }
            Err(error)
        }
    }
}

pub(crate) fn capacity(requirements: &[StateRequirement], sequences: usize) -> Option<u64> {
    requirements
        .iter()
        .try_fold(0_u64, |bytes, requirement| {
            bytes.checked_add(requirement.byte_size()?)
        })?
        .checked_mul(u64::try_from(sequences).ok()?)
}

pub(crate) fn manager(device: DeviceId, bytes: u64) -> LogicalStateManager {
    LogicalStateManager::new(device, bytes, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{DataType, KvStateSpec};

    fn kv(tokens: u32) -> StateRequirement {
        StateRequirement::FullAttentionKv(KvStateSpec::new(1, 1, 2, tokens, DataType::F16).unwrap())
    }

    #[test]
    fn allocation_and_bundle_validation_failures_restore_capacity() {
        let location = StateLocation::Device(DeviceId::new(0));
        let mut manager = manager(DeviceId::new(0), 32);
        assert!(matches!(
            allocate(&mut manager, &[kv(2), kv(4)]),
            Err(StateError::CapacityExceeded { .. })
        ));
        assert_eq!(manager.used_bytes(location), Some(0));
        assert_eq!(
            allocate(&mut manager, &[kv(2), kv(2)]),
            Err(StateError::DuplicateRequirement)
        );
        assert_eq!(manager.used_bytes(location), Some(0));
        let state = allocate(&mut manager, &[kv(4)]).unwrap();
        assert_eq!(manager.used_bytes(location), Some(32));
        manager.release_set(&state).unwrap();
        assert_eq!(manager.used_bytes(location), Some(0));
        assert_eq!(capacity(&[kv(4)], 2), Some(64));
    }
}
