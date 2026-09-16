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
use ribn_batch::{BatchConfig, BatchRuntime, BlockReason, Rejection, StepOutcome};
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
