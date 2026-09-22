//! Epoch registration with optional one-shot wake notifications.
//!
//! Register before testing the condition; publish only after changing it. A wait
//! remains changed even if publication precedes parking. Execution owners can arm
//! their notification transport before parking without losing that publication.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::Waker;

#[derive(Debug, Default)]
struct Source {
    epoch: AtomicU64,
    notifications: Mutex<Vec<Weak<Notification>>>,
}

#[derive(Debug)]
struct Notification(Waker);

/// Resource-owner publication source. Clones name the same condition, not a
/// second accounting authority. This value grants no resources.
#[derive(Clone, Debug, Default)]
pub struct Readiness(Arc<Source>);

impl Readiness {
    /// Register before attempting the operation that may need to wait.
    #[must_use]
    pub fn register(&self) -> ReadinessWait {
        ReadinessWait {
            source: self.clone(),
            observed: self.0.epoch.load(Ordering::SeqCst),
            notification: None,
        }
    }

    /// Publish after the condition changes and wake armed owners once. This is
    /// permission to retry, not a promise that the next operation succeeds.
    /// If a callback unwinds, notify the remaining owners before propagating it.
    pub fn publish(&self) {
        let notifications = {
            let mut registered = self
                .0
                .notifications
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Keep epoch advancement and detachment atomic with arming. Otherwise
            // a new-epoch subscriber could consume its wake on this old publication.
            self.0.epoch.fetch_add(1, Ordering::SeqCst);
            std::mem::take(&mut *registered)
        };
        // Waking (or dropping a waker) may reenter this source. Never do it while
        // holding the registry lock. Weak entries do not keep cancelled waits alive.
        let mut failure = None;
        for notification in notifications {
            if let Some(notification) = notification.upgrade()
                && let Err(error) = std::panic::catch_unwind(move || notification.0.wake_by_ref())
                && failure.is_none()
            {
                failure = Some(error);
            }
        }
        if let Some(error) = failure {
            std::panic::resume_unwind(error);
        }
    }
}

/// A retained registration, tied to its source rather than a bare epoch number.
#[derive(Clone, Debug)]
pub struct ReadinessWait {
    source: Readiness,
    observed: u64,
    notification: Option<Arc<Notification>>,
}

impl ReadinessWait {
    /// Whether this attempt should be retried. It remains true until the caller
    /// replaces this registration, including when publication precedes parking.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.source.0.epoch.load(Ordering::SeqCst) != self.observed
    }

    /// Arm an owner's wake transport before parking. Repeated calls with the same
    /// waker do not add subscriptions. First arming after a change wakes immediately;
    /// otherwise publication consumes the subscription. Retrying needs a new wait.
    ///
    /// Clones may share a subscription; it is released when its last owner drops
    /// or replaces it. Stale weak entries are pruned on subsequent arming/publication,
    /// bounding registry storage by the peak number of concurrently armed waits.
    pub fn wake_on_change(&mut self, waker: &Waker) {
        if self
            .notification
            .as_ref()
            .is_some_and(|current| current.0.will_wake(waker))
        {
            return;
        }
        let notification = Arc::new(Notification(waker.clone()));
        let changed = {
            let mut registered = self
                .source
                .0
                .notifications
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let changed = self.changed();
            if !changed {
                if registered.len() == registered.capacity() {
                    registered.retain(|entry| entry.strong_count() != 0);
                    // Batch compaction rather than scanning N live subscriptions
                    // for every new one. Leave slack for near-full-set churn.
                    let live = registered.len();
                    registered.reserve(live);
                }
                registered.push(Arc::downgrade(&notification));
            }
            changed
        };
        self.notification = Some(notification);
        if changed {
            waker.wake_by_ref();
        }
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
    use std::sync::atomic::AtomicUsize;
    use std::task::Wake;

    #[derive(Default)]
    struct Counter(AtomicUsize);
    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn publication_on_either_side_of_arming_wakes_once() {
        for publish_first in [false, true] {
            let source = Readiness::default();
            let mut wait = source.register();
            let counter = Arc::new(Counter::default());
            let waker = Waker::from(counter.clone());
            if publish_first {
                source.publish();
            }
            wait.wake_on_change(&waker);
            wait.wake_on_change(&waker);
            if !publish_first {
                source.publish();
            }
            assert!(wait.changed());
            assert_eq!(counter.0.load(Ordering::SeqCst), 1);
            source.publish();
            assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn dropped_waits_release_wakers_and_stale_entries_are_bounded() {
        let source = Readiness::default();
        let counter = Arc::new(Counter::default());
        let waker = Waker::from(counter.clone());
        let mut initial_capacity = 0;
        for index in 0..1000 {
            let mut wait = source.register();
            wait.wake_on_change(&waker);
            let registered = source.0.notifications.lock().unwrap();
            if index == 0 {
                initial_capacity = registered.capacity();
            }
            assert!(registered.len() <= initial_capacity);
        }
        assert_eq!(Arc::strong_count(&counter), 2); // counter + local waker only
        source.publish();
        assert_eq!(counter.0.load(Ordering::SeqCst), 0);
        assert!(source.0.notifications.lock().unwrap().is_empty());
    }

    #[test]
    fn cloned_registration_and_distinct_owners_keep_their_notifications() {
        let source = Readiness::default();
        let first = Arc::new(Counter::default());
        let second = Arc::new(Counter::default());
        let mut wait = source.register();
        wait.wake_on_change(&Waker::from(first.clone()));
        let mut clone = wait.clone();
        clone.wake_on_change(&Waker::from(second.clone()));
        source.publish();
        assert_eq!(first.0.load(Ordering::SeqCst), 1);
        assert_eq!(second.0.load(Ordering::SeqCst), 1);
        assert_eq!(wait, clone); // transport does not alter registration identity
    }

    #[test]
    fn racing_publication_and_arming_cannot_lose_a_wake() {
        for _ in 0..128 {
            let source = Readiness::default();
            let mut wait = source.register();
            let counter = Arc::new(Counter::default());
            let waker = Waker::from(counter.clone());
            let barrier = std::sync::Barrier::new(2);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    barrier.wait();
                    source.publish();
                });
                barrier.wait();
                wait.wake_on_change(&waker);
            });
            assert!(wait.changed());
            assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn a_wake_can_arm_the_next_epoch_without_deadlock_or_consuming_its_signal() {
        struct Rearm {
            source: Readiness,
            target: Waker,
            next: Mutex<Option<ReadinessWait>>,
        }
        impl Rearm {
            fn arm_next(&self) {
                let mut next = self.source.register();
                next.wake_on_change(&self.target);
                *self.next.lock().unwrap() = Some(next);
            }
        }
        impl Wake for Rearm {
            fn wake(self: Arc<Self>) {
                self.arm_next();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.arm_next();
            }
        }
        let source = Readiness::default();
        let counter = Arc::new(Counter::default());
        let rearm = Arc::new(Rearm {
            source: source.clone(),
            target: Waker::from(counter.clone()),
            next: Mutex::new(None),
        });
        let mut wait = source.register();
        wait.wake_on_change(&Waker::from(rearm.clone()));
        source.publish();
        assert_eq!(counter.0.load(Ordering::SeqCst), 0);
        assert!(!rearm.next.lock().unwrap().as_ref().unwrap().changed());
        source.publish();
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_panicking_callback_does_not_strand_other_owners() {
        struct Panicker;
        impl Wake for Panicker {
            fn wake(self: Arc<Self>) {
                panic!("injected wake panic");
            }
        }
        let source = Readiness::default();
        let mut broken = source.register();
        broken.wake_on_change(&Waker::from(Arc::new(Panicker)));
        let counter = Arc::new(Counter::default());
        let mut healthy = source.register();
        healthy.wake_on_change(&Waker::from(counter.clone()));
        assert!(std::panic::catch_unwind(|| source.publish()).is_err());
        assert!(healthy.changed());
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_shared_source_wakes_all_live_registrations() {
        let source = Readiness::default();
        let counter = Arc::new(Counter::default());
        let waker = Waker::from(counter.clone());
        let mut waits: Vec<_> = (0..256).map(|_| source.register()).collect();
        for wait in &mut waits {
            wait.wake_on_change(&waker);
        }
        source.publish();
        assert!(waits.iter().all(ReadinessWait::changed));
        assert_eq!(counter.0.load(Ordering::SeqCst), waits.len());
    }

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
