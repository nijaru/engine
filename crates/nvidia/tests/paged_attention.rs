#![cfg(feature = "cuda")]

//! Block-table KV addressing must not change attention arithmetic.
//!
//! The paged entry point resolves each logical token through a block table and
//! then runs the same per-row device body as the contiguous entry point. A
//! sequence whose blocks are contiguous, or whose table permutes them, must
//! therefore produce bit-identical scores and outputs. This is the addressing
//! prerequisite for block-granular continuation; it does not exercise a cache,
//! reuse policy, or recurrent checkpoints.

use cudarc::driver::CudaContext;
use engine_nvidia::CudaQwen35Ops;

const BLOCK_TOKENS: usize = 4;
const KV_HEADS: usize = 2;
const Q_HEADS: usize = 4;
const HEAD_DIM: usize = 64;

/// Distinct finite F16 patterns: keys are small positive normals, values are
/// small negative normals, so every dot product and softmax weight is finite
/// while remaining sensitive to a wrong token address.
fn patterns(count: usize, seed: u16) -> Vec<u16> {
    (0..count)
        .map(|index| {
            let mantissa = u16::try_from((index * 37 + usize::from(seed)) % 512).unwrap();
            0x3000 + mantissa
        })
        .collect()
}

/// Physically scatter `logical` tokens into `table.len()` blocks.
fn scatter(logical: &[u16], table: &[i32], block_tokens: usize) -> Vec<u16> {
    let width = KV_HEADS * HEAD_DIM;
    let mut physical = vec![0_u16; table.len() * block_tokens * width];
    for (logical_block, &physical_block) in table.iter().enumerate() {
        let blocks = logical.len().div_ceil(width * block_tokens);
        assert!(
            logical_block < blocks,
            "table names a block outside the data"
        );
        let physical_block = usize::try_from(physical_block).expect("non-negative block index");
        for offset in 0..block_tokens {
            let source = logical_block * block_tokens + offset;
            if source >= logical.len() / width {
                continue;
            }
            let from = source * width;
            let to = (physical_block * block_tokens + offset) * width;
            physical[to..to + width].copy_from_slice(&logical[from..from + width]);
        }
    }
    physical
}

#[test]
#[ignore = "requires a CUDA device"]
fn block_table_addressing_matches_contiguous_attention_bit_for_bit() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");
    let width = KV_HEADS * HEAD_DIM;
    let max_tokens = 10;
    let logical_keys = patterns(max_tokens * width, 0);
    let logical_values: Vec<u16> = patterns(max_tokens * width, 0)
        .into_iter()
        .map(|bits| bits | 0x8000)
        .collect();
    let contiguous_keys = stream.clone_htod(&logical_keys).expect("upload keys");
    let contiguous_values = stream.clone_htod(&logical_values).expect("upload values");

    // A shuffled table and the identity table read the same logical tokens.
    for table in [vec![2_i32, 0, 1], vec![0_i32, 1, 2]] {
        let table_device = stream.clone_htod(&table).expect("upload block table");
        let pool_keys = stream
            .clone_htod(&scatter(&logical_keys, &table, BLOCK_TOKENS))
            .expect("upload paged keys");
        let pool_values = stream
            .clone_htod(&scatter(&logical_values, &table, BLOCK_TOKENS))
            .expect("upload paged values");
        // Single token, an aligned block, a partial tail block, and causal
        // multi-row prefill chunks that share the cache.
        for (tokens, rows) in [(1_usize, 1_usize), (4, 1), (5, 1), (8, 1), (7, 3), (10, 10)] {
            let elements = rows * Q_HEADS * HEAD_DIM;
            let stride = max_tokens;
            let q = stream
                .clone_htod(&vec![0.25_f32; elements])
                .expect("upload q");
            let gate = stream
                .clone_htod(&vec![0.1_f32; 2 * elements])
                .expect("upload gate");
            let mut expected_out = stream.alloc_zeros::<f32>(elements).expect("out");
            let mut actual_out = stream.alloc_zeros::<f32>(elements).expect("out");
            let mut expected_scores = stream.alloc_zeros::<f32>(rows * Q_HEADS * stride).unwrap();
            let mut actual_scores = stream.alloc_zeros::<f32>(rows * Q_HEADS * stride).unwrap();
            ops.attn_score_gqa(
                &q,
                &contiguous_keys,
                &contiguous_values,
                &gate,
                &mut expected_scores,
                &mut expected_out,
                tokens,
                rows,
                stride,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
            )
            .expect("contiguous attention");
            ops.attn_score_gqa_paged(
                &q,
                &pool_keys,
                &pool_values,
                &table_device,
                BLOCK_TOKENS,
                &gate,
                &mut actual_scores,
                &mut actual_out,
                tokens,
                rows,
                stride,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
            )
            .expect("paged attention");
            for (name, expected, actual) in [
                (
                    "output",
                    stream.clone_dtoh(&expected_out).unwrap(),
                    stream.clone_dtoh(&actual_out).unwrap(),
                ),
                (
                    "scores",
                    stream.clone_dtoh(&expected_scores).unwrap(),
                    stream.clone_dtoh(&actual_scores).unwrap(),
                ),
            ] {
                assert_eq!(expected.len(), actual.len());
                for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                    assert!(
                        expected.is_finite() && actual.is_finite(),
                        "{name} not finite at {index}"
                    );
                    assert_eq!(
                        expected.to_bits(),
                        actual.to_bits(),
                        "{name} differs: table={table:?} tokens={tokens} rows={rows} index={index}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn block_table_geometry_is_validated_before_launch() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, stream.clone()).expect("compile Qwen ops");
    let width = KV_HEADS * HEAD_DIM;
    let keys = stream
        .clone_htod(&patterns(2 * BLOCK_TOKENS * width, 0))
        .unwrap();
    let values = stream
        .clone_htod(&patterns(2 * BLOCK_TOKENS * width, 0))
        .unwrap();
    // Two logical blocks are required for nine tokens; the table names one.
    let short_table = stream.clone_htod(&vec![0_i32]).unwrap();
    let q = stream
        .clone_htod(&vec![0.25_f32; Q_HEADS * HEAD_DIM])
        .unwrap();
    let gate = stream
        .clone_htod(&vec![0.1_f32; 2 * Q_HEADS * HEAD_DIM])
        .unwrap();
    let mut scores = stream.alloc_zeros::<f32>(Q_HEADS * 16).unwrap();
    let mut out = stream.alloc_zeros::<f32>(Q_HEADS * HEAD_DIM).unwrap();
    assert!(
        ops.attn_score_gqa_paged(
            &q,
            &keys,
            &values,
            &short_table,
            BLOCK_TOKENS,
            &gate,
            &mut scores,
            &mut out,
            9,
            1,
            16,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
        )
        .is_err()
    );
    // A zero block size cannot resolve any token.
    let table = stream.clone_htod(&vec![0_i32, 1]).unwrap();
    assert!(
        ops.attn_score_gqa_paged(
            &q,
            &keys,
            &values,
            &table,
            0,
            &gate,
            &mut scores,
            &mut out,
            1,
            1,
            16,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
        )
        .is_err()
    );
}
