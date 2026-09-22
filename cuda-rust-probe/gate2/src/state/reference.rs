//! Independent row-major f64 GDN equations and a forward f32 error envelope.
//! Bounds depend on inputs and operation counts, never on observed device error.
#[derive(Clone, Copy, Debug)]
pub(super) struct Value {
    value: f64,
    error: f64,
}

impl Value {
    pub(super) fn exact(value: f32) -> Self {
        assert!(value.is_finite() && (value == 0.0 || value.is_normal()));
        Self {
            value: f64::from(value),
            error: 0.0,
        }
    }

    // One full epsilon (twice unit roundoff) conservatively covers rounding,
    // including the f64 evaluation of this envelope. MIN_POSITIVE is an absolute
    // underflow floor, not general coverage for flushing subnormal operands.
    // This reference is fixture-scoped: normal/zero operands, no overflow or
    // arbitrary reassociation. Device arithmetic uses non-FTZ add/mul/FMA.
    fn add(self, rhs: Self) -> Self {
        let magnitude = self.value.abs() + self.error + rhs.value.abs() + rhs.error;
        Self {
            value: self.value + rhs.value,
            error: self.error
                + rhs.error
                + f64::from(f32::EPSILON) * magnitude
                + f64::from(f32::MIN_POSITIVE),
        }
    }

    fn mul(self, rhs: Self) -> Self {
        let magnitude = (self.value.abs() + self.error) * (rhs.value.abs() + rhs.error);
        Self {
            value: self.value * rhs.value,
            error: self.value.abs() * rhs.error
                + rhs.value.abs() * self.error
                + self.error * rhs.error
                + f64::from(f32::EPSILON) * magnitude
                + f64::from(f32::MIN_POSITIVE),
        }
    }

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            error: self.error,
        }
    }

    pub(super) fn accepts(self, actual: f32) -> bool {
        actual.is_finite()
            && self.value.is_finite()
            && self.error.is_finite()
            && (f64::from(actual) - self.value).abs() <= self.error
    }
}

pub(super) struct Inputs<'a> {
    pub heads: usize,
    pub key_heads: usize,
    pub dim: usize,
    pub offset: usize,
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub decay: &'a [f32],
    pub beta: &'a [f32],
}

impl Inputs<'_> {
    pub(super) fn advance(&self, matrices: &mut [Vec<Value>]) -> Vec<Value> {
        let Self {
            heads,
            key_heads,
            dim,
            offset,
            q,
            k,
            v,
            decay,
            beta,
        } = *self;
        assert!(dim > 0 && key_heads > 0 && heads > 0);
        assert_eq!(q.len(), matrices.len() * key_heads * dim);
        assert_eq!(k.len(), q.len());
        assert_eq!(v.len(), matrices.len() * (offset + heads * dim));
        assert_eq!(decay.len(), matrices.len() * heads);
        assert_eq!(beta.len(), decay.len());
        let mut output = Vec::with_capacity(matrices.len() * heads * dim);
        // PTX ISA 9.4 rsqrt.approx.f32: maximum relative error 2^-22.9.
        // Two f32 epsilons exceed that bound; this is NOT a measured tolerance.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt
        let scale = 1.0 / (dim as f64).sqrt();
        let scale = Value {
            value: scale,
            error: scale * 2.0 * f64::from(f32::EPSILON),
        };
        for (member, matrix) in matrices.iter_mut().enumerate() {
            assert_eq!(matrix.len(), heads * dim * dim);
            for (head, state) in matrix.chunks_exact_mut(dim * dim).enumerate() {
                let start = (member * key_heads + head % key_heads) * dim;
                let q = &q[start..start + dim];
                let k = &k[start..start + dim];
                let start = member * (offset + heads * dim) + offset + head * dim;
                let values = &v[start..start + dim];
                let decay = Value::exact(decay[member * heads + head]);
                let beta = Value::exact(beta[member * heads + head]);
                for value in state.iter_mut() {
                    *value = value.mul(decay);
                }
                let mut projected = vec![Value::exact(0.0); dim];
                for (row, key) in state.chunks_exact(dim).zip(k) {
                    for (sum, value) in projected.iter_mut().zip(row) {
                        *sum = sum.add(value.mul(Value::exact(*key)));
                    }
                }
                let delta: Vec<_> = values
                    .iter()
                    .zip(projected)
                    .map(|(v, sum)| Value::exact(*v).add(sum.neg()).mul(beta))
                    .collect();
                for (row, key) in state.chunks_exact_mut(dim).zip(k) {
                    for (value, delta) in row.iter_mut().zip(&delta) {
                        // Separate multiply/add bounds also cover a contracted FMA.
                        *value = value.add(Value::exact(*key).mul(*delta));
                    }
                }
                let mut result = vec![Value::exact(0.0); dim];
                for (row, query) in state.chunks_exact(dim).zip(q) {
                    for (sum, value) in result.iter_mut().zip(row) {
                        *sum = sum.add(value.mul(Value::exact(*query)));
                    }
                }
                output.extend(result.into_iter().map(|value| value.mul(scale)));
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_mapping_orientation_members_and_value_offset() {
        let inputs = Inputs {
            heads: 3,
            key_heads: 2,
            dim: 2,
            offset: 3,
            q: &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0],
            k: &[1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0],
            v: &[
                -999.0, -999.0, -999.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -999.0, -999.0, -999.0, 7.0,
                8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            decay: &[0.0; 6],
            beta: &[1.0; 6],
        };
        let mut states = vec![vec![Value::exact(17.0); 12]; 2];
        let output = inputs.advance(&mut states);
        // Decay zero erases old state; beta one gives the outer product k*v^T.
        let expected = [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0, 5.0, 6.0, 0.0, 0.0],
            [
                21.0, 24.0, 0.0, 0.0, 0.0, 0.0, 36.0, 40.0, 33.0, 36.0, 0.0, 0.0,
            ],
        ];
        for (state, expected) in states.iter().zip(expected) {
            for (value, expected) in state.iter().zip(expected) {
                assert_eq!(value.value, expected);
            }
        }
        // q selects the first or second row; head two reuses key/query head zero.
        let expected = [
            1.0, 2.0, 6.0, 8.0, 5.0, 6.0, -21.0, -24.0, 36.0, 40.0, -33.0, -36.0,
        ];
        for (value, expected) in output.iter().zip(expected) {
            assert!(
                (value.value - expected / 2.0_f64.sqrt()).abs() <= f64::EPSILON * expected.abs()
            );
        }
    }

    #[test]
    fn envelope_covers_fused_and_unfused_cancellation() {
        let values = [0.0_f32, 1e-12, -1e-12, 0.315, -0.73, 17.0, -4096.0];
        for a in values {
            for b in values {
                for c in values.into_iter().chain([-a * b]) {
                    let reference = Value::exact(a).mul(Value::exact(b)).add(Value::exact(c));
                    assert!(reference.accepts(a * b + c));
                    assert!(reference.accepts(a.mul_add(b, c)));
                }
            }
        }
    }

    #[test]
    fn scalar_equations_and_envelope_reject_corruption() {
        let inputs = Inputs {
            heads: 1,
            key_heads: 1,
            dim: 1,
            offset: 0,
            q: &[0.5],
            k: &[0.25],
            v: &[0.75],
            decay: &[0.5],
            beta: &[0.5],
        };
        let mut states = vec![vec![Value::exact(1.0)]];
        let output = inputs.advance(&mut states);
        // S'=0.5 + 0.25 * ((0.75 - 0.5*0.25)*0.5) = 0.578125.
        assert_eq!(states[0][0].value, 0.578125);
        assert_eq!(output[0].value, 0.2890625);
        assert!(states[0][0].accepts(0.578125));
        assert!(output[0].accepts(0.2890625));
        assert!(!states[0][0].accepts(0.578225));
        assert!(!output[0].accepts(0.0));
        assert!(!output[0].accepts(f32::NAN));
        let output = inputs.advance(&mut states);
        assert_eq!(states[0][0].value, 0.373779296875);
        assert!(output[0].accepts(0.186_889_65));
    }
}
