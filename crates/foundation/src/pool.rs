//! Shared physical-pool accounting.
//!
//! One authority grants reservations for each shared physical pool. A reservation is
//! an owning, non-duplicable lease: moving it between owners does not reserve again,
//! and physical storage plus its charge stay together until safe reuse. This module
//! deliberately contains no trait objects and no generic storage type; a backend that
//! materializes device memory for a lease keeps the two in one owner.
//!
//! Readiness is published as a release epoch, not as a wakeup mechanism. The runtime
//! that parks waiting work owns registration-and-recheck against that epoch, so a
//! release between a capacity check and parking cannot be lost.

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Identity of one granted allocation.
///
/// A lease carries this so a later compatibility check can tell "the same allocation"
/// from "the same byte count". It is unique per grant; a released allocation is never
/// reissued under the same identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocationId(u64);

impl AllocationId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Why a reservation could not be granted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReserveError {
    /// The pool can grant this amount, but not while other leases are alive.
    /// The caller may register for the pool's current epoch and retry after a
    /// release; this is ordinary backpressure, not a request defect.
    Exhausted {
        /// Bytes the caller asked for.
        requested: u64,
        /// Bytes currently available.
        available: u64,
    },
    /// The request can never be granted by this pool, whatever is released.
    /// Permanent request-local infeasibility, not a capacity wait.
    TooLarge {
        /// Bytes the caller asked for.
        requested: u64,
        /// Bytes this pool can ever grant.
        capacity: u64,
    },
    /// The pool is closed; no new reservation is granted.
    Closed,
}

impl fmt::Display for ReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted {
                requested,
                available,
            } => write!(
                f,
                "pool has {available} bytes available, {requested} requested"
            ),
            Self::TooLarge {
                requested,
                capacity,
            } => write!(
                f,
                "pool capacity {capacity} bytes cannot grant {requested} bytes"
            ),
            Self::Closed => f.write_str("pool is closed"),
        }
    }
}

impl Error for ReserveError {}

/// Authority for one shared physical pool of bytes.
///
/// The pool grants leases; it does not own storage. A backend attaches the storage it
/// materialized to the lease it was granted, and releases both together.
pub struct BytePool {
    capacity: u64,
    state: Mutex<PoolState>,
    epoch: AtomicU64,
}

#[derive(Debug)]
struct PoolState {
    granted: u64,
    next_allocation: u64,
    closed: bool,
}

impl BytePool {
    /// Create a pool that may grant at most `capacity` bytes in total.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            state: Mutex::new(PoolState {
                granted: 0,
                next_allocation: 1,
                closed: false,
            }),
            epoch: AtomicU64::new(0),
        }
    }

    /// Total bytes this pool can ever grant, counting every lease it has issued.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The authority is shared, since one pool serves every runtime drawing on it.
    #[must_use]
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn lock(&self) -> MutexGuard<'_, PoolState> {
        // A panic while holding this lock can only happen inside the few arithmetic
        // statements below, which cannot observe a half-applied reservation because
        // each is a single mutation of a `u64` field. Recovering the guard keeps the
        // accounting usable instead of turning one panic into a permanent outage.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Bytes not currently covered by a live lease.
    #[must_use]
    pub fn available(&self) -> u64 {
        let state = self.lock();
        self.capacity.saturating_sub(state.granted)
    }

    /// Bytes covered by live leases.
    #[must_use]
    pub fn granted(&self) -> u64 {
        self.lock().granted
    }

    /// Current release epoch. It advances whenever a lease releases bytes, so a
    /// runtime that parked on capacity can tell "something changed" from "the pool
    /// is unchanged" without polling for a specific byte count.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Reserve `bytes` and return the owning lease.
    ///
    /// # Errors
    /// Returns [`ReserveError::TooLarge`] for an amount this pool can never grant,
    /// [`ReserveError::Exhausted`] while other leases hold the capacity, or
    /// [`ReserveError::Closed`] after [`BytePool::close`].
    pub fn reserve(self: &Arc<Self>, bytes: u64) -> Result<PoolLease, ReserveError> {
        if bytes > self.capacity {
            return Err(ReserveError::TooLarge {
                requested: bytes,
                capacity: self.capacity,
            });
        }
        let mut state = self.lock();
        if state.closed {
            return Err(ReserveError::Closed);
        }
        let available = self.capacity.saturating_sub(state.granted);
        if bytes > available {
            return Err(ReserveError::Exhausted {
                requested: bytes,
                available,
            });
        }
        state.granted += bytes;
        let allocation = AllocationId(state.next_allocation);
        state.next_allocation += 1;
        drop(state);
        Ok(PoolLease {
            pool: Arc::clone(self),
            allocation,
            bytes,
        })
    }

    /// Refuse new reservations. Existing leases keep their charge and release
    /// normally, so shutdown does not pretend that live device state is free.
    pub fn close(&self) {
        self.lock().closed = true;
    }

    /// Whether this pool still grants reservations.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    fn release(&self, bytes: u64) {
        let mut state = self.lock();
        state.granted = state.granted.saturating_sub(bytes);
        drop(state);
        // Publish after the bytes are actually available, so a waiter that observes
        // this epoch and rechecks capacity never sees the old accounting.
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }
}

impl fmt::Debug for BytePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("BytePool")
            .field("capacity", &self.capacity)
            .field("granted", &state.granted)
            .field("closed", &state.closed)
            .field("epoch", &self.epoch.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

/// An owning reservation of pool bytes.
///
/// Dropping a lease releases its charge. Transferring the lease moves the charge with
/// it, so a result handed to another owner keeps its bytes reserved, and popping a
/// result from a queue does not free them.
#[derive(Debug)]
pub struct PoolLease {
    pool: Arc<BytePool>,
    allocation: AllocationId,
    bytes: u64,
}

impl PoolLease {
    /// Identity of this granted allocation, stable for the lifetime of the lease.
    #[must_use]
    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    /// Bytes this lease keeps reserved.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Reduce the lease to `bytes`, releasing the remainder immediately.
    ///
    /// A backend uses this when it learns the real demand is smaller than its
    /// preparation-time estimate. It cannot grow, because growing would have to
    /// negotiate with the authority again.
    ///
    /// # Errors
    /// Returns [`ReserveError::TooLarge`] when `bytes` exceeds the current charge.
    pub fn shrink_to(&mut self, bytes: u64) -> Result<(), ReserveError> {
        if bytes > self.bytes {
            return Err(ReserveError::TooLarge {
                requested: bytes,
                capacity: self.bytes,
            });
        }
        let released = self.bytes - bytes;
        self.bytes = bytes;
        if released > 0 {
            self.pool.release(released);
        }
        Ok(())
    }
}

impl Drop for PoolLease {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.pool.release(self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grants_exact_capacity_and_refuses_beyond_it() {
        let pool = BytePool::new(1024).shared();
        let lease = pool.reserve(1024).unwrap();
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.granted(), 1024);
        assert_eq!(
            pool.reserve(1).unwrap_err(),
            ReserveError::Exhausted {
                requested: 1,
                available: 0
            }
        );
        assert_eq!(
            pool.reserve(1025).unwrap_err(),
            ReserveError::TooLarge {
                requested: 1025,
                capacity: 1024
            }
        );
        assert_eq!(lease.bytes(), 1024);
        drop(lease);
        assert_eq!(pool.available(), 1024);
    }

    #[test]
    fn unsatisfiable_requests_are_rejected_not_waited_out() {
        let pool = BytePool::new(64).shared();
        // Occupying the whole pool cannot make an oversized request satisfiable,
        // so rejection must not depend on current availability.
        let _held = pool.reserve(64).unwrap();
        assert!(matches!(
            pool.reserve(65).unwrap_err(),
            ReserveError::TooLarge { .. }
        ));
    }

    #[test]
    fn epoch_advances_only_on_release() {
        let pool = BytePool::new(256).shared();
        let start = pool.epoch();
        let first = pool.reserve(128).unwrap();
        assert_eq!(pool.epoch(), start);
        let second = pool.reserve(128).unwrap();
        assert_eq!(pool.epoch(), start);
        assert!(pool.reserve(1).is_err());
        drop(first);
        let after_release = pool.epoch();
        assert!(after_release > start);
        drop(second);
        assert!(pool.epoch() > after_release);
    }

    #[test]
    fn allocations_are_identifiable_and_never_reissued() {
        let pool = BytePool::new(256).shared();
        let first = pool.reserve(8).unwrap();
        let first_id = first.allocation();
        let second = pool.reserve(8).unwrap();
        assert_ne!(first_id, second.allocation());
        drop(first);
        let third = pool.reserve(8).unwrap();
        assert_ne!(first_id, third.allocation());
        assert_ne!(second.allocation(), third.allocation());
    }

    #[test]
    fn a_moved_lease_keeps_its_charge() {
        let pool = BytePool::new(64).shared();
        let lease = pool.reserve(64).unwrap();
        let handed_off = lease;
        assert_eq!(pool.granted(), 64);
        assert!(pool.reserve(1).is_err());
        drop(handed_off);
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn shrink_releases_only_the_surplus_and_cannot_grow() {
        let pool = BytePool::new(64).shared();
        let mut lease = pool.reserve(48).unwrap();
        lease.shrink_to(16).unwrap();
        assert_eq!(lease.bytes(), 16);
        assert_eq!(pool.granted(), 16);
        assert_eq!(pool.available(), 48);
        assert!(matches!(
            lease.shrink_to(17).unwrap_err(),
            ReserveError::TooLarge { .. }
        ));
        let allocation = lease.allocation();
        lease.shrink_to(16).unwrap();
        assert_eq!(lease.allocation(), allocation);
        lease.shrink_to(0).unwrap();
        assert_eq!(pool.granted(), 0);
        drop(lease);
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn closing_refuses_new_work_but_releases_existing_leases() {
        let pool = BytePool::new(64).shared();
        let held = pool.reserve(32).unwrap();
        pool.close();
        assert_eq!(pool.reserve(1).unwrap_err(), ReserveError::Closed);
        drop(held);
        assert_eq!(pool.available(), 64);
        assert_eq!(pool.reserve(1).unwrap_err(), ReserveError::Closed);
    }
}
