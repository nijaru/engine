//! Host-only algebra check for the chunked gated-delta rule used by Qwen prefill.
//!
//! This intentionally uses tiny matrices rather than the pinned Qwen geometry. It
//! proves the chunk transform against the token-recurrent definition before a CUDA
//! chunk kernel exists. Device qualification remains a separate gate.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "the tiny mathematical reference keeps matrix indices and tensor roles explicit"
)]

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn attention_scale(key_dim: usize) -> f32 {
    f32::from(u16::try_from(key_dim).expect("reference key dimension fits u16"))
        .sqrt()
        .recip()
}

fn recurrent_rule(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    initial_state: &[f32],
    sequence: usize,
    key_dim: usize,
    value_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0_f32; sequence * value_dim];
    let scale = attention_scale(key_dim);

    for token in 0..sequence {
        let q = &query[token * key_dim..(token + 1) * key_dim];
        let k = &key[token * key_dim..(token + 1) * key_dim];
        let v = &value[token * value_dim..(token + 1) * value_dim];
        let decay = log_decay[token].exp();
        for state_value in &mut state {
            *state_value *= decay;
        }

        let mut prediction = vec![0.0_f32; value_dim];
        for key_index in 0..key_dim {
            for value_index in 0..value_dim {
                prediction[value_index] +=
                    state[key_index * value_dim + value_index] * k[key_index];
            }
        }
        let correction = (0..value_dim)
            .map(|value_index| (v[value_index] - prediction[value_index]) * beta[token])
            .collect::<Vec<_>>();
        for key_index in 0..key_dim {
            for value_index in 0..value_dim {
                state[key_index * value_dim + value_index] +=
                    k[key_index] * correction[value_index];
            }
        }
        for value_index in 0..value_dim {
            output[token * value_dim + value_index] = (0..key_dim)
                .map(|key_index| state[key_index * value_dim + value_index] * q[key_index])
                .sum::<f32>()
                * scale;
        }
    }

    (output, state)
}

fn chunk_rule(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    initial_state: &[f32],
    sequence: usize,
    key_dim: usize,
    value_dim: usize,
    chunk_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(chunk_size > 0);
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0_f32; sequence * value_dim];
    let scale = attention_scale(key_dim);

    let mut chunk_start = 0;
    while chunk_start < sequence {
        let chunk_len = chunk_size.min(sequence - chunk_start);
        let mut cumulative_decay = vec![0.0_f32; chunk_len];
        let mut cumulative = 0.0_f32;
        for local in 0..chunk_len {
            cumulative += log_decay[chunk_start + local];
            cumulative_decay[local] = cumulative;
        }

        // A is unit-lower-triangular for the solve. Only its strict lower
        // triangle is stored: A[i,j] = beta_i * <k_i,k_j> * decay(j -> i).
        let mut lower = vec![0.0_f32; chunk_len * chunk_len];
        let mut intra = vec![0.0_f32; chunk_len * chunk_len];
        for row in 0..chunk_len {
            let token_row = chunk_start + row;
            let q = &query[token_row * key_dim..(token_row + 1) * key_dim];
            let k_row = &key[token_row * key_dim..(token_row + 1) * key_dim];
            for column in 0..=row {
                let token_column = chunk_start + column;
                let k_column = &key[token_column * key_dim..(token_column + 1) * key_dim];
                let decay = (cumulative_decay[row] - cumulative_decay[column]).exp();
                intra[row * chunk_len + column] = dot(q, k_column) * scale * decay;
                if column < row {
                    lower[row * chunk_len + column] =
                        beta[token_row] * dot(k_row, k_column) * decay;
                }
            }
        }

        // Solve A * X = beta*V and A * Kc = beta*K*exp(cumulative_decay)
        // by forward substitution. This is the UT transform used by chunked
        // DeltaNet implementations, written directly for the host oracle.
        let mut new_values = vec![0.0_f32; chunk_len * value_dim];
        let mut decayed_keys = vec![0.0_f32; chunk_len * key_dim];
        for row in 0..chunk_len {
            let token = chunk_start + row;
            for value_index in 0..value_dim {
                let mut solved = value[token * value_dim + value_index] * beta[token];
                for previous in 0..row {
                    solved -= lower[row * chunk_len + previous]
                        * new_values[previous * value_dim + value_index];
                }
                new_values[row * value_dim + value_index] = solved;
            }
            for key_index in 0..key_dim {
                let mut solved =
                    key[token * key_dim + key_index] * beta[token] * cumulative_decay[row].exp();
                for previous in 0..row {
                    solved -= lower[row * chunk_len + previous]
                        * decayed_keys[previous * key_dim + key_index];
                }
                decayed_keys[row * key_dim + key_index] = solved;
            }
        }

        // Fold the old state out of each transformed value.
        let mut value_update = new_values;
        for row in 0..chunk_len {
            for value_index in 0..value_dim {
                let old_prediction = (0..key_dim)
                    .map(|key_index| {
                        decayed_keys[row * key_dim + key_index]
                            * state[key_index * value_dim + value_index]
                    })
                    .sum::<f32>();
                value_update[row * value_dim + value_index] -= old_prediction;
            }
        }

        // Output = read of the old state with a cumulatively-decayed query,
        // plus all causal within-chunk transformed updates.
        for row in 0..chunk_len {
            let token = chunk_start + row;
            let query_decay = cumulative_decay[row].exp();
            for value_index in 0..value_dim {
                let old_read = (0..key_dim)
                    .map(|key_index| {
                        query[token * key_dim + key_index]
                            * scale
                            * query_decay
                            * state[key_index * value_dim + value_index]
                    })
                    .sum::<f32>();
                let within_chunk = (0..=row)
                    .map(|column| {
                        intra[row * chunk_len + column]
                            * value_update[column * value_dim + value_index]
                    })
                    .sum::<f32>();
                output[token * value_dim + value_index] = old_read + within_chunk;
            }
        }

        // Carry only one recurrent matrix between chunks.
        let chunk_decay = cumulative_decay[chunk_len - 1].exp();
        for state_value in &mut state {
            *state_value *= chunk_decay;
        }
        for row in 0..chunk_len {
            let key_decay = (cumulative_decay[chunk_len - 1] - cumulative_decay[row]).exp();
            let token = chunk_start + row;
            for key_index in 0..key_dim {
                let key_value = key[token * key_dim + key_index] * key_decay;
                for value_index in 0..value_dim {
                    state[key_index * value_dim + value_index] +=
                        key_value * value_update[row * value_dim + value_index];
                }
            }
        }

        chunk_start += chunk_len;
    }

    (output, state)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (*actual - *expected).abs();
        assert!(
            difference <= 2.0e-5,
            "value {index} differs: actual={actual}, expected={expected}, abs={difference}"
        );
    }
}

#[test]
fn chunked_gated_delta_rule_matches_token_recurrence() {
    const SEQUENCE: usize = 7;
    const KEY_DIM: usize = 3;
    const VALUE_DIM: usize = 2;

    let query = [
        0.2, -0.1, 0.7, 0.5, 0.3, -0.4, -0.6, 0.8, 0.1, 0.9, -0.2, 0.4, 0.3, 0.6, -0.5, -0.7, -0.1,
        0.8, 0.4, -0.9, 0.2,
    ];
    let key = [
        0.4, -0.3, 0.5, -0.2, 0.7, 0.1, 0.6, 0.2, -0.4, -0.5, 0.3, 0.8, 0.1, -0.8, 0.6, 0.7, 0.4,
        -0.2, -0.3, 0.9, 0.2,
    ];
    let value = [
        0.7, -0.2, 0.1, 0.8, -0.6, 0.5, 0.4, -0.9, 0.3, 0.2, -0.1, 0.6, 0.9, -0.4,
    ];
    let log_decay = [-0.11, -0.27, -0.04, -0.33, -0.19, -0.08, -0.22];
    let beta = [0.25, 0.73, 0.41, 0.88, 0.36, 0.64, 0.52];
    let initial_state = [0.15, -0.2, 0.05, 0.31, -0.17, 0.09];

    let (expected_output, expected_state) = recurrent_rule(
        &query,
        &key,
        &value,
        &log_decay,
        &beta,
        &initial_state,
        SEQUENCE,
        KEY_DIM,
        VALUE_DIM,
    );

    for chunk_size in 1..=SEQUENCE + 2 {
        let (actual_output, actual_state) = chunk_rule(
            &query,
            &key,
            &value,
            &log_decay,
            &beta,
            &initial_state,
            SEQUENCE,
            KEY_DIM,
            VALUE_DIM,
            chunk_size,
        );
        assert_close(&actual_output, &expected_output);
        assert_close(&actual_state, &expected_state);
    }
}
