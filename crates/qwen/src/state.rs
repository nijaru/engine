use engine_core::{DeviceId, LogicalStateManager, StateRequirement};

pub(crate) fn capacity(requirements: &[StateRequirement], sequences: usize) -> Option<u64> {
    requirements
        .iter()
        .try_fold(0_u64, |bytes, requirement| {
            bytes.checked_add(requirement.byte_size()?)
        })?
        .checked_mul(u64::try_from(sequences).ok()?)
}

pub(crate) fn manager(device: DeviceId, bytes: u64) -> LogicalStateManager {
    LogicalStateManager::new(
        device,
        ribn_foundation::BytePool::new(bytes).shared(),
        ribn_foundation::BytePool::new(0).shared(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{DataType, KvStateSpec, StateError, StateLocation, StateManager};

    fn kv(tokens: u32) -> StateRequirement {
        StateRequirement::FullAttentionKv(KvStateSpec::new(1, 1, 2, tokens, DataType::F16).unwrap())
    }

    #[test]
    fn capacity_readiness_tracks_successful_release_not_commit_or_invalid_release() {
        let location = StateLocation::Device(DeviceId::new(0));
        let mut manager = manager(DeviceId::new(0), 32);
        let wait = manager.capacity_wait(location).unwrap();
        let host_wait = manager.capacity_wait(StateLocation::Host).unwrap();
        let mut state = manager.allocate_set(&[kv(4)], location).unwrap();
        let old_handle = state.kv().unwrap().handle().clone();
        manager.commit(&mut state, 1).unwrap();
        assert!(!wait.changed());
        assert!(manager.release(old_handle).is_err());
        assert!(!wait.changed());
        manager.release_set(&state).unwrap();
        assert!(wait.changed());
        assert!(!host_wait.changed());
    }

    #[test]
    fn refused_and_invalid_bundles_preserve_capacity_without_publication() {
        let location = StateLocation::Device(DeviceId::new(0));
        let mut manager = manager(DeviceId::new(0), 32);
        let wait = manager.capacity_wait(location).unwrap();
        assert!(matches!(
            manager.allocate_set(&[kv(2), kv(4)], location),
            Err(StateError::CapacityExceeded { .. })
        ));
        assert_eq!(manager.used_bytes(location), Some(0));
        assert_eq!(
            manager.allocate_set(&[kv(2), kv(2)], location),
            Err(StateError::DuplicateRequirement)
        );
        assert_eq!(manager.used_bytes(location), Some(0));
        assert!(
            !wait.changed(),
            "refused and invalid bundles never create a rollback wake"
        );
        let state = manager.allocate_set(&[kv(4)], location).unwrap();
        assert_eq!(manager.used_bytes(location), Some(32));
        manager.release_set(&state).unwrap();
        assert_eq!(manager.used_bytes(location), Some(0));
        assert_eq!(capacity(&[kv(4)], 2), Some(64));
    }
}
