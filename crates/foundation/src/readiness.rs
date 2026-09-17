//! Registration for a changing resource condition, without a waiter queue.
//!
//! Register before testing the condition; publish only after changing it. A wait
//! remains changed even if publication precedes parking. Callers must observe it
//! at a bounded interval until their execution owner has a notification transport.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Resource-owner publication source. Clones name the same condition, not a
/// second accounting authority. This value grants no resources.
#[derive(Clone, Debug, Default)]
pub struct Readiness(Arc<AtomicU64>);

impl Readiness {
    /// Register before attempting the operation that may need to wait.
    #[must_use]
    pub fn register(&self) -> ReadinessWait {
        ReadinessWait {
            source: self.clone(),
            observed: self.0.load(Ordering::SeqCst),
        }
    }

    /// Publish after the condition has changed. This is not a wake notification
    /// or a promise that the next operation succeeds.
    pub fn publish(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A retained registration, tied to its source rather than a bare epoch number.
#[derive(Clone, Debug)]
pub struct ReadinessWait {
    source: Readiness,
    observed: u64,
}

impl ReadinessWait {
    /// Whether this attempt should be retried. It remains true until the caller
    /// replaces this registration, including when publication precedes parking.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.source.0.load(Ordering::SeqCst) != self.observed
    }
}

impl PartialEq for ReadinessWait {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.source.0, &other.source.0) && self.observed == other.observed
    }
}
impl Eq for ReadinessWait {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_before_parking_remains_observable() {
        let source = Readiness::default();
        let wait = source.register();
        assert!(!wait.changed());
        source.publish();
        assert!(wait.changed());
        assert!(wait.changed());
        assert!(!source.register().changed());
    }

    #[test]
    fn unrelated_sources_do_not_reactivate_a_wait() {
        let source = Readiness::default();
        let other = Readiness::default();
        let wait = source.register();
        other.publish();
        assert!(!wait.changed());
        assert_ne!(wait, other.register());
        source.clone().publish();
        assert!(wait.changed());
    }
}
