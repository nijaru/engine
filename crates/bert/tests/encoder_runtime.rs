//! Run the device encoder through the non-autoregressive batching runtime.
//!
//! These tests are the wiring evidence for the prepared-resource contract: the
//! pool bounds how many requests run, a device-resident result keeps its charge
//! after it leaves the runtime, and a request the encoder can never execute is
//! rejected without stalling a peer. Compiling is not device evidence, so they are
//! ignored by default and must be run serially with an idle GPU.
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use engine_bert::{CudaBertEncoder, EncoderCompletion, EncoderConstraint, EncoderRequest};
use ribn_batch::{BatchConfig, BatchRuntime, BlockReason, CancelOutcome, Rejection, StepOutcome};
use ribn_foundation::BytePool;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../batch/tests/fixtures/bert-tiny")
}

fn config() -> BatchConfig {
    BatchConfig {
        max_waiting_requests: 8,
        max_retained_results: 4,
    }
}

fn encoder() -> CudaBertEncoder {
    CudaBertEncoder::load(fixture(), 0).expect("prepare encoder")
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn the_shared_pool_bounds_how_many_requests_execute() {
    let encoder = encoder();
    let sequence = 4;
    let envelope = encoder.request_bytes(sequence).expect("envelope");
    assert!(envelope > 0);
    // Exactly two requests' worth of device bytes.
    let pool = BytePool::new(2 * envelope).shared();
    let mut runtime = BatchRuntime::new(encoder, Arc::clone(&pool), config()).expect("runtime");
    for _ in 0..3 {
        runtime
            .submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
            .expect("request");
    }

    // The accepted range is the prefix whose real envelopes the pool can cover.
    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 2 }
    );
    assert_eq!(pool.granted(), 2 * envelope);
    assert_eq!(
        runtime.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: envelope,
            available: 0
        })
    );

    // The envelope the pool was charged for is the storage that actually exists.
    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("first completion"));
    let EncoderCompletion::Result(result) = completion else {
        panic!("an executing request is not a rejection");
    };
    assert_eq!(result.device_bytes(), envelope);
    assert_eq!(result.retained_bytes(), envelope);
    assert_eq!(
        pool.granted(),
        2 * envelope,
        "a popped device result keeps its charge until it is released"
    );

    // The consumer awaits the producer's completion dependency before reading.
    result.synchronize().expect("synchronize");
    assert!(result.is_complete().expect("completion"));
    let output = result.read().expect("read");
    let hidden = result.submission().shape().hidden();
    assert_eq!(output.last_hidden_state.len(), sequence * hidden);
    assert_eq!(output.pooled.len(), hidden);
    assert!(
        output.pooled.iter().all(|value| value.is_finite()),
        "pooled output must stay finite"
    );

    // Releasing the result frees exactly its own envelope, admitting the queued
    // request.
    drop(result);
    assert_eq!(pool.granted(), envelope);
    drop(runtime.pop_completed().expect("second completion"));
    assert_eq!(pool.granted(), 0);
    assert_eq!(
        runtime.step().expect("step after release"),
        StepOutcome::Executed { results: 1 }
    );

    // The last request still owns its charge and completes normally.
    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("third completion"));
    let EncoderCompletion::Result(result) = completion else {
        panic!("an executing request is not a rejection");
    };
    result.synchronize().expect("synchronize");
    assert!(result.read().is_ok());
    assert_eq!(
        runtime.executor().quarantined_requests(),
        0,
        "a successful run must not leave the encoder holding device work"
    );
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn a_request_longer_than_the_model_is_rejected_while_a_peer_progresses() {
    let encoder = encoder();
    let max = encoder.config().max_position_embeddings;
    let pool = BytePool::new(1 << 24).shared();
    let mut runtime = BatchRuntime::new(encoder, Arc::clone(&pool), config()).expect("runtime");

    let oversized = runtime
        .submit(EncoderRequest::single_segment(vec![1; max + 1]))
        .expect("oversized request");
    let healthy = runtime
        .submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
        .expect("healthy request");

    // Permanent request-local infeasibility is delivered without waiting, and the
    // peer keeps moving.
    assert_eq!(
        runtime.step().expect("rejecting step"),
        StepOutcome::Rejected { request: oversized }
    );
    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("rejection entry"));
    let EncoderCompletion::Rejected { request, rejection } = completion else {
        panic!("an infeasible request never executes");
    };
    assert_eq!(request, oversized);
    assert!(matches!(
        rejection,
        Rejection::Executor(EncoderConstraint::SequenceTooLong {
            max_position_embeddings,
            ..
        }) if max_position_embeddings == max
    ));

    assert_eq!(
        runtime.step().expect("healthy peer"),
        StepOutcome::Executed { results: 1 }
    );
    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("healthy completion"));
    let EncoderCompletion::Result(result) = completion else {
        panic!("the peer executed");
    };
    assert_eq!(result.request(), healthy);
    result.synchronize().expect("synchronize");
    assert!(result.read().is_ok());
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn a_request_larger_than_the_whole_pool_is_rejected_while_a_peer_progresses() {
    let encoder = encoder();
    let small = encoder.request_bytes(4).expect("small envelope");
    let large = encoder.request_bytes(12).expect("large envelope");
    assert!(large > small);
    // A pool that can never grant the large request, whatever is released.
    let pool = BytePool::new(large - 1).shared();
    let mut runtime = BatchRuntime::new(encoder, Arc::clone(&pool), config()).expect("runtime");

    let oversized = runtime
        .submit(EncoderRequest::single_segment(vec![1; 12]))
        .expect("oversized request");
    let healthy = runtime
        .submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
        .expect("healthy request");
    assert!(small <= pool.capacity());

    assert_eq!(
        runtime.step().expect("rejecting step"),
        StepOutcome::Rejected { request: oversized }
    );
    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("rejection entry"));
    let EncoderCompletion::Rejected { rejection, .. } = completion else {
        panic!("an oversized request never executes");
    };
    assert!(matches!(
        rejection,
        Rejection::RetainedOutputTooLarge { requested, capacity }
            if requested == large && capacity == large - 1
    ));

    assert_eq!(
        runtime.step().expect("healthy peer"),
        StepOutcome::Executed { results: 1 }
    );
    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("healthy completion"));
    let EncoderCompletion::Result(result) = completion else {
        panic!("the peer executed");
    };
    assert_eq!(result.request(), healthy);
    assert_eq!(result.device_bytes(), small);
    result.synchronize().expect("synchronize");
    let output = result.read().expect("read");
    assert!(output.pooled.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn cancelling_a_request_that_is_waiting_for_capacity_runs_nothing() {
    let encoder = encoder();
    let envelope = encoder.request_bytes(4).expect("envelope");
    // A sibling holds the whole pool, so the request can only wait.
    let pool = BytePool::new(envelope).shared();
    let held = pool.reserve(envelope).expect("sibling reservation");
    let mut runtime = BatchRuntime::new(encoder, Arc::clone(&pool), config()).expect("runtime");
    let request = runtime
        .submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
        .expect("request");
    assert!(matches!(
        runtime.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::Pool { .. })
    ));

    assert_eq!(runtime.cancel(request), CancelOutcome::Queued);
    assert_eq!(pool.granted(), envelope, "the sibling keeps its charge");
    assert_eq!(
        runtime.step().expect("cancelling step"),
        StepOutcome::Cancelled { request }
    );
    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("cancellation entry"));
    let EncoderCompletion::Cancelled { request: cancelled } = completion else {
        panic!("a cancelled request never executes");
    };
    assert_eq!(cancelled, request);
    assert_eq!(runtime.executor().quarantined_requests(), 0);

    drop(held);
    assert_eq!(
        runtime.step().expect("idle step"),
        StepOutcome::Idle,
        "a cancelled request is never executed"
    );
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn cancelling_a_retained_result_leaves_its_charge_with_its_storage() {
    let encoder = encoder();
    let envelope = encoder.request_bytes(4).expect("envelope");
    let pool = BytePool::new(2 * envelope).shared();
    let mut runtime = BatchRuntime::new(encoder, Arc::clone(&pool), config()).expect("runtime");
    let request = runtime
        .submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
        .expect("request");
    assert_eq!(
        runtime.step().expect("step"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(pool.granted(), envelope);

    assert_eq!(runtime.cancel(request), CancelOutcome::Discarded);
    // The runtime hands the result and its charge to the encoder, so the two never
    // separate: either the encoder still holds both, or a proven drain released both.
    let quarantined = runtime.executor().quarantined_requests();
    let held_charge = runtime.executor().quarantined_charge();
    let expected_charge = if quarantined == 0 { 0 } else { envelope };
    assert_eq!(held_charge, expected_charge);
    assert_eq!(
        pool.granted(),
        expected_charge,
        "storage and the charge covering it are never separated"
    );

    let completion =
        EncoderCompletion::from_completed(runtime.pop_completed().expect("cancellation entry"));
    assert!(
        matches!(completion, EncoderCompletion::Cancelled { .. }),
        "the cancellation is delivered through the ordinary terminal queue"
    );

    // Whatever the drain did, the charge must not outlive the storage.
    runtime
        .executor_mut()
        .drain_retirement()
        .expect("drain after cancellation");
    assert_eq!(runtime.executor().quarantined_requests(), 0);
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires an idle CUDA GPU; run serially with --test-threads=1"]
fn a_lagging_consumer_holds_its_charge_while_a_peer_still_progresses() {
    let envelope = encoder().request_bytes(4).expect("envelope");
    // Room for one result per runtime, so the lagging consumer is what holds the
    // peer back rather than the pool being unable to run anything.
    let pool = BytePool::new(2 * envelope).shared();
    let mut lagging =
        BatchRuntime::new(encoder(), Arc::clone(&pool), config()).expect("lagging runtime");
    let mut peer = BatchRuntime::new(encoder(), Arc::clone(&pool), config()).expect("peer runtime");

    lagging
        .submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
        .expect("lagging request");
    peer.submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
        .expect("peer request");
    assert_eq!(
        lagging.step().expect("lagging step"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(
        peer.step().expect("peer step"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(pool.granted(), 2 * envelope);

    // The lagging runtime never awaits its result or releases it, so the pool is
    // fully committed and a third request cannot start anywhere.
    lagging
        .submit(EncoderRequest::single_segment(vec![4, 1, 9, 3]))
        .expect("lagging request");
    assert_eq!(
        lagging.step().expect("blocked step"),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: envelope,
            available: 0
        })
    );
    assert_eq!(pool.available(), 0);

    // Consuming the peer's result admits exactly that one request: the lagging
    // consumer keeps its own charge, and progress does not require its cooperation.
    drop(peer.pop_completed().expect("peer entry"));
    assert_eq!(pool.granted(), envelope);
    assert_eq!(
        peer.step().expect("peer step after consuming"),
        StepOutcome::Executed { results: 1 }
    );
    assert_eq!(pool.granted(), 2 * envelope);

    // The lagging consumer's result is still intact and still charged: nothing
    // released it while it waited.
    let entry = lagging.pop_completed().expect("lagging entry");
    let completion = EncoderCompletion::from_completed(entry);
    assert!(matches!(completion, EncoderCompletion::Result(_)));
    let EncoderCompletion::Result(result) = completion else {
        panic!("the lagging result is an output");
    };
    result.synchronize().expect("synchronize");
    assert!(result.read().is_ok());
    drop(result);
    assert_eq!(
        pool.available(),
        envelope,
        "consuming the lagging result releases exactly its own charge"
    );
}
