use std::error::Error;
use std::fmt;

use ribn_foundation::BackendId;

#[derive(Clone, Copy)]
struct RmsNormSpec {
    width: usize,
    epsilon: f32,
}

#[derive(Debug)]
enum OpError {
    NoImplementation,
    InvalidShape,
    WidthTooLarge,
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation => {
                f.write_str("no RMSNorm implementation supports this preparation")
            }
            Self::InvalidShape => f.write_str("RMSNorm input/weight shape does not match its spec"),
            Self::WidthTooLarge => f.write_str("RMSNorm reference fixture width is too large"),
        }
    }
}

impl Error for OpError {}

type RmsNormKernel = fn(&RmsNormSpec, &[f32], &[f32]) -> Result<Vec<f32>, OpError>;

/// Result of preparation: semantic matching is already finished. Execution does
/// not walk a registry or ask support predicates again.
struct PreparedRmsNorm {
    implementation: &'static str,
    spec: RmsNormSpec,
    kernel: RmsNormKernel,
}

impl PreparedRmsNorm {
    fn execute(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>, OpError> {
        (self.kernel)(&self.spec, input, weight)
    }
}

trait RmsNormFactory {
    fn supports(&self, backend: &BackendId, spec: RmsNormSpec) -> bool;
    fn prepare(&self, spec: RmsNormSpec) -> PreparedRmsNorm;
}

struct SpecializedFixture;

impl RmsNormFactory for SpecializedFixture {
    fn supports(&self, backend: &BackendId, spec: RmsNormSpec) -> bool {
        backend.as_str() == "fixture-gpu" && spec.width.is_multiple_of(4)
    }

    fn prepare(&self, spec: RmsNormSpec) -> PreparedRmsNorm {
        PreparedRmsNorm {
            implementation: "fixture-gpu-rmsnorm",
            spec,
            kernel: specialized_kernel,
        }
    }
}

struct ReferenceFixture;

impl RmsNormFactory for ReferenceFixture {
    fn supports(&self, _backend: &BackendId, _spec: RmsNormSpec) -> bool {
        true
    }

    fn prepare(&self, spec: RmsNormSpec) -> PreparedRmsNorm {
        PreparedRmsNorm {
            implementation: "reference-rmsnorm",
            spec,
            kernel: reference_kernel,
        }
    }
}

fn prepare_rmsnorm(
    backend: &BackendId,
    spec: RmsNormSpec,
    factories: &[&dyn RmsNormFactory],
) -> Result<PreparedRmsNorm, OpError> {
    factories
        .iter()
        .find(|factory| factory.supports(backend, spec))
        .map(|factory| factory.prepare(spec))
        .ok_or(OpError::NoImplementation)
}

fn rmsnorm(spec: &RmsNormSpec, input: &[f32], weight: &[f32]) -> Result<Vec<f32>, OpError> {
    if input.len() != spec.width || weight.len() != spec.width || spec.width == 0 {
        return Err(OpError::InvalidShape);
    }
    let width = u16::try_from(spec.width).map_err(|_| OpError::WidthTooLarge)?;
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / f32::from(width);
    let inverse_rms = (mean_square + spec.epsilon).sqrt().recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, scale)| value * inverse_rms * scale)
        .collect())
}

fn reference_kernel(
    spec: &RmsNormSpec,
    input: &[f32],
    weight: &[f32],
) -> Result<Vec<f32>, OpError> {
    rmsnorm(spec, input, weight)
}

fn specialized_kernel(
    spec: &RmsNormSpec,
    input: &[f32],
    weight: &[f32],
) -> Result<Vec<f32>, OpError> {
    // This fixture uses the same arithmetic oracle. A real backend would return
    // a prepared kernel/launch object rather than duplicate semantic selection.
    rmsnorm(spec, input, weight)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((*actual - *expected).abs() <= 1.0e-6);
    }
}

#[test]
fn preparation_selects_backend_implementation_once() {
    let gpu = BackendId::new("fixture-gpu").expect("gpu backend");
    let cpu = BackendId::new("cpu").expect("cpu backend");
    let spec = RmsNormSpec {
        width: 4,
        epsilon: 1.0e-6,
    };
    let specialized = SpecializedFixture;
    let reference = ReferenceFixture;
    let factories: [&dyn RmsNormFactory; 2] = [&specialized, &reference];

    let gpu_op = prepare_rmsnorm(&gpu, spec, &factories).expect("gpu preparation");
    let cpu_op = prepare_rmsnorm(&cpu, spec, &factories).expect("cpu preparation");
    assert_eq!(gpu_op.implementation, "fixture-gpu-rmsnorm");
    assert_eq!(cpu_op.implementation, "reference-rmsnorm");

    let input = [1.0, -2.0, 0.5, 3.0];
    let weight = [1.0, 0.5, 2.0, 1.5];
    let expected = reference_kernel(&spec, &input, &weight).expect("reference output");
    assert_close(
        &gpu_op.execute(&input, &weight).expect("gpu output"),
        &expected,
    );
    assert_close(
        &cpu_op.execute(&input, &weight).expect("cpu output"),
        &expected,
    );
}

#[test]
fn unsupported_shape_falls_back_without_changing_semantics() {
    let gpu = BackendId::new("fixture-gpu").expect("gpu backend");
    let spec = RmsNormSpec {
        width: 3,
        epsilon: 1.0e-6,
    };
    let specialized = SpecializedFixture;
    let reference = ReferenceFixture;
    let factories: [&dyn RmsNormFactory; 2] = [&specialized, &reference];

    let prepared = prepare_rmsnorm(&gpu, spec, &factories).expect("fallback preparation");
    assert_eq!(prepared.implementation, "reference-rmsnorm");
}
