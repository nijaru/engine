//! Semantic-operator dispatch fixture: selecting an implementation once must not
//! become an assumption that the selection stays valid for every runtime shape.
//!
//! Preparation chooses between a specialized and a reference implementation from
//! backend plus parameter spec. A real runtime then executes varying shapes - row
//! counts per step, sequence lengths, ragged batches - and an optimized kernel is
//! usually qualified for only some of them. The fixture therefore separates two
//! questions that are easy to collapse into one:
//!
//! - *selection*: which implementation serves this backend and parameter spec? It
//!   happens once, at preparation, and execution must not repeat it;
//! - *shape validity*: is the selected implementation qualified for this step's
//!   shape? It is asked per execution, and an unqualified shape uses the reference
//!   implementation instead of being silently served by the optimized path.
//!
//! The kernels here share one arithmetic oracle; a real backend would launch
//! different kernels. What the fixture tests is the dispatch and qualification
//! contract around them, not numerical work.

use std::cell::Cell;
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
    /// The shape is valid for the operation but not for the implementation the
    /// caller chose. It is a routing error, not a malformed request.
    UnsupportedShape {
        rows: usize,
    },
}

impl fmt::Display for OpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation => {
                formatter.write_str("no RMSNorm implementation supports this preparation")
            }
            Self::InvalidShape => {
                formatter.write_str("RMSNorm input/weight shape does not match its spec")
            }
            Self::WidthTooLarge => {
                formatter.write_str("RMSNorm reference fixture width is too large")
            }
            Self::UnsupportedShape { rows } => {
                write!(formatter, "implementation is not qualified for {rows} rows")
            }
        }
    }
}

impl Error for OpError {}

/// Rows one execution covers, so a step's shape is runtime data.
type RmsNormKernel = fn(&RmsNormSpec, usize, &[f32], &[f32]) -> Result<Vec<f32>, OpError>;

/// Result of preparation: semantic matching is already finished. Execution does
/// not walk a registry or ask support predicates again.
struct PreparedRmsNorm {
    implementation: &'static str,
    spec: RmsNormSpec,
    /// Which runtime row counts this prepared implementation may serve. A step
    /// asks this instead of re-running selection.
    supports_rows: fn(usize) -> bool,
    kernel: RmsNormKernel,
}

impl PreparedRmsNorm {
    fn supports_rows(&self, rows: usize) -> bool {
        (self.supports_rows)(rows)
    }

    fn execute(&self, rows: usize, input: &[f32], weight: &[f32]) -> Result<Vec<f32>, OpError> {
        if !self.supports_rows(rows) {
            return Err(OpError::UnsupportedShape { rows });
        }
        (self.kernel)(&self.spec, rows, input, weight)
    }
}

trait RmsNormFactory {
    fn supports(&self, backend: &BackendId, spec: RmsNormSpec) -> bool;
    fn prepare(&self, spec: RmsNormSpec) -> PreparedRmsNorm;
}

/// Optimized implementation, qualified for a single row and for groups of four
/// rows or more. It cannot serve two or three rows, which is exactly the case a
/// preparation-time decision would get wrong if it assumed every shape.
struct SpecializedFixture {
    preparations: Cell<usize>,
}

impl RmsNormFactory for SpecializedFixture {
    fn supports(&self, backend: &BackendId, spec: RmsNormSpec) -> bool {
        backend.as_str() == "fixture-gpu" && spec.width.is_multiple_of(4)
    }

    fn prepare(&self, spec: RmsNormSpec) -> PreparedRmsNorm {
        self.preparations.set(self.preparations.get() + 1);
        PreparedRmsNorm {
            implementation: "fixture-gpu-rmsnorm",
            spec,
            supports_rows: |rows| rows == 1 || rows >= 4,
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
            supports_rows: |rows| rows >= 1,
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

fn rmsnorm(
    spec: &RmsNormSpec,
    rows: usize,
    input: &[f32],
    weight: &[f32],
) -> Result<Vec<f32>, OpError> {
    if rows == 0
        || spec.width == 0
        || weight.len() != spec.width
        || input.len() != rows.checked_mul(spec.width).ok_or(OpError::InvalidShape)?
    {
        return Err(OpError::InvalidShape);
    }
    let width = u16::try_from(spec.width).map_err(|_| OpError::WidthTooLarge)?;
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(spec.width) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / f32::from(width);
        let inverse_rms = (mean_square + spec.epsilon).sqrt().recip();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(value, scale)| value * inverse_rms * scale),
        );
    }
    Ok(output)
}

fn reference_kernel(
    spec: &RmsNormSpec,
    rows: usize,
    input: &[f32],
    weight: &[f32],
) -> Result<Vec<f32>, OpError> {
    rmsnorm(spec, rows, input, weight)
}

fn specialized_kernel(
    spec: &RmsNormSpec,
    rows: usize,
    input: &[f32],
    weight: &[f32],
) -> Result<Vec<f32>, OpError> {
    // This fixture uses the same arithmetic oracle. A real backend would return
    // a prepared kernel/launch object rather than duplicate semantic selection.
    rmsnorm(spec, rows, input, weight)
}

/// What one executed step used, so a caller can observe routing decisions.
struct StepResult {
    implementation: &'static str,
    output: Vec<f32>,
}

/// One runtime step: use the selected implementation when it is qualified for this
/// step's row count, and the reference implementation otherwise. Selection is
/// already finished; this only routes an execution.
fn execute_step(
    selected: &PreparedRmsNorm,
    reference: &PreparedRmsNorm,
    rows: usize,
    input: &[f32],
    weight: &[f32],
) -> Result<StepResult, OpError> {
    if selected.supports_rows(rows) {
        return Ok(StepResult {
            implementation: selected.implementation,
            output: selected.execute(rows, input, weight)?,
        });
    }
    Ok(StepResult {
        implementation: reference.implementation,
        output: reference.execute(rows, input, weight)?,
    })
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((*actual - *expected).abs() <= 1.0e-6);
    }
}

fn fixture() -> (BackendId, RmsNormSpec, SpecializedFixture, ReferenceFixture) {
    (
        BackendId::new("fixture-gpu").expect("gpu backend"),
        RmsNormSpec {
            width: 4,
            epsilon: 1.0e-6,
        },
        SpecializedFixture {
            preparations: Cell::new(0),
        },
        ReferenceFixture,
    )
}

/// Deterministic rows of input so shape variation changes the data, not the test.
fn rows_input(rows: usize, width: usize) -> Vec<f32> {
    (0..rows * width)
        .map(|index| {
            let value = f32::from(u16::try_from(index % 7).expect("small index")) - 3.0;
            value * 0.5
        })
        .collect()
}

const WEIGHT: [f32; 4] = [1.0, 0.5, 2.0, 1.5];

#[test]
fn preparation_selects_backend_implementation_once() {
    let (gpu, spec, specialized, reference) = fixture();
    let cpu = BackendId::new("cpu").expect("cpu backend");
    let factories: [&dyn RmsNormFactory; 2] = [&specialized, &reference];

    let gpu_op = prepare_rmsnorm(&gpu, spec, &factories).expect("gpu preparation");
    let cpu_op = prepare_rmsnorm(&cpu, spec, &factories).expect("cpu preparation");
    assert_eq!(gpu_op.implementation, "fixture-gpu-rmsnorm");
    assert_eq!(cpu_op.implementation, "reference-rmsnorm");
    assert_eq!(specialized.preparations.get(), 1);

    let input = rows_input(1, spec.width);
    let expected = reference_kernel(&spec, 1, &input, &WEIGHT).expect("reference output");
    assert_close(
        &gpu_op.execute(1, &input, &WEIGHT).expect("gpu output"),
        &expected,
    );
    assert_close(
        &cpu_op.execute(1, &input, &WEIGHT).expect("cpu output"),
        &expected,
    );
}

#[test]
fn unsupported_width_falls_back_without_changing_semantics() {
    let (gpu, _, specialized, reference) = fixture();
    let spec = RmsNormSpec {
        width: 3,
        epsilon: 1.0e-6,
    };
    let factories: [&dyn RmsNormFactory; 2] = [&specialized, &reference];

    let prepared = prepare_rmsnorm(&gpu, spec, &factories).expect("fallback preparation");
    assert_eq!(prepared.implementation, "reference-rmsnorm");
}

/// The optimized path claims a set of runtime shapes. Every claimed shape must
/// agree with the reference implementation, which is what qualifies it; the shapes
/// it does not claim must not reach it at all.
#[test]
fn optimized_shapes_are_qualified_against_the_reference() {
    let (gpu, spec, specialized, reference) = fixture();
    let factories: [&dyn RmsNormFactory; 2] = [&specialized, &reference];
    let selected = prepare_rmsnorm(&gpu, spec, &factories).expect("preparation");
    let fallback = reference.prepare(spec);

    for rows in 1..=8 {
        let input = rows_input(rows, spec.width);
        let step = execute_step(&selected, &fallback, rows, &input, &WEIGHT).expect("step");
        let expected = reference_kernel(&spec, rows, &input, &WEIGHT).expect("reference");
        assert_close(&step.output, &expected);
        let qualified = rows == 1 || rows >= 4;
        assert_eq!(
            step.implementation,
            if qualified {
                "fixture-gpu-rmsnorm"
            } else {
                "reference-rmsnorm"
            },
            "row count {rows} routed to the wrong implementation"
        );
    }
}

/// Shape changes are runtime events. They must be routed per step, and they must
/// not cause re-selection - and an unqualified shape asked of the optimized
/// implementation must fail loudly rather than compute anyway.
#[test]
fn runtime_shape_changes_neither_reprepare_nor_silently_use_unqualified_paths() {
    let (gpu, spec, specialized, reference) = fixture();
    let factories: [&dyn RmsNormFactory; 2] = [&specialized, &reference];
    let selected = prepare_rmsnorm(&gpu, spec, &factories).expect("preparation");
    let fallback = reference.prepare(spec);

    let mut routed = Vec::new();
    for rows in [4, 1, 2, 3, 8, 2, 1] {
        let input = rows_input(rows, spec.width);
        let step = execute_step(&selected, &fallback, rows, &input, &WEIGHT).expect("step");
        routed.push((rows, step.implementation));
    }
    assert_eq!(
        routed,
        vec![
            (4, "fixture-gpu-rmsnorm"),
            (1, "fixture-gpu-rmsnorm"),
            (2, "reference-rmsnorm"),
            (3, "reference-rmsnorm"),
            (8, "fixture-gpu-rmsnorm"),
            (2, "reference-rmsnorm"),
            (1, "fixture-gpu-rmsnorm"),
        ]
    );
    assert_eq!(
        specialized.preparations.get(),
        1,
        "shape changes must not re-run preparation"
    );

    let input = rows_input(2, spec.width);
    assert!(matches!(
        selected.execute(2, &input, &WEIGHT),
        Err(OpError::UnsupportedShape { rows: 2 })
    ));
}
