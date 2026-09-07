use super::*;

#[test]
#[ignore = "requires a CUDA device"]
fn converts_all_f16_bit_patterns() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let source = format!(
        r#"{MODEL_OPS_SOURCE}
extern "C" __global__ void convert_f16_patterns(float* output) {{
    unsigned int bits = blockIdx.x * blockDim.x + threadIdx.x;
    if (bits < 65536u) output[bits] = f16_bits_to_f32((unsigned short)bits);
}}
"#
    );
    let ptx = compile_ptx(source).expect("compile conversion regression kernel");
    let module = context.load_module(ptx).expect("load conversion module");
    let kernel = module
        .load_function("convert_f16_patterns")
        .expect("load conversion kernel");
    let mut output = stream.alloc_zeros::<f32>(65536).expect("allocate output");
    // Safety: the kernel writes exactly one element for each of the 65536
    // half-precision bit patterns into the equally sized device allocation.
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&mut output)
            .launch(LaunchConfig::for_num_elems(65536))
            .expect("launch conversion");
    }
    let actual = stream.clone_dtoh(&output).expect("read converted values");
    for bits in 0..=u16::MAX {
        let exponent = i32::from((bits >> 10) & 31);
        let fraction = f32::from(bits & 1023);
        let magnitude = if exponent == 0 {
            fraction * 2.0_f32.powi(-24)
        } else if exponent == 31 {
            if fraction == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15)
        };
        let expected = if bits & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        };
        let actual = actual[usize::from(bits)];
        if expected.is_nan() {
            assert!(actual.is_nan(), "F16 {bits:#06x}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "F16 {bits:#06x}");
        }
    }
}
