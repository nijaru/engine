//! Ownership of accepted submissions across terminal device failures.
use engine_core::BackendSubmissionId;
use std::collections::HashMap;

pub(crate) struct Submissions<T> {
    pub(crate) pending: HashMap<BackendSubmissionId, T>,
    quarantined: Vec<T>,
    fault: Option<String>,
    undrained: bool,
}

impl<T> Default for Submissions<T> {
    fn default() -> Self {
        Self {
            pending: HashMap::new(),
            quarantined: Vec::new(),
            fault: None,
            undrained: false,
        }
    }
}

impl<T> Submissions<T> {
    pub(crate) fn fault(&self) -> Option<&str> {
        self.fault.as_deref()
    }
    pub(crate) fn release_is_safe(&self) -> bool {
        !self.undrained
    }
    pub(crate) fn quarantine(&mut self, value: T) {
        self.quarantined.push(value);
    }

    /// Stop polling every affected submission. A successful stream drain is
    /// the only evidence permitting retirement; otherwise ownership stays here.
    pub(crate) fn fail(&mut self, error: String, drained: bool) -> Vec<T> {
        self.fault = Some(error);
        self.undrained = !drained;
        self.quarantined
            .extend(self.pending.drain().map(|(_, value)| value));
        if drained {
            std::mem::take(&mut self.quarantined)
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn failed_drain_retains_all_resources_until_teardown() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut submissions = Submissions::default();
        submissions.pending.insert(
            BackendSubmissionId::new(1).unwrap(),
            Resource(dropped.clone()),
        );
        submissions.quarantine(Resource(dropped.clone()));
        assert!(
            submissions
                .fail("event query failed".into(), false)
                .is_empty()
        );
        assert!(submissions.pending.is_empty());
        assert!(submissions.fault().is_some());
        assert!(!submissions.release_is_safe());
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(submissions);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn successful_drain_retires_resources_but_keeps_dispatcher_poisoned() {
        let mut submissions = Submissions::default();
        submissions
            .pending
            .insert(BackendSubmissionId::new(1).unwrap(), 7);
        submissions.quarantine(8);
        let mut retired = submissions.fail("pinned read failed".into(), true);
        retired.sort_unstable();
        assert_eq!(retired, vec![7, 8]);
        assert!(submissions.pending.is_empty());
        assert!(submissions.release_is_safe());
        assert!(submissions.fault().is_some());
    }
}
