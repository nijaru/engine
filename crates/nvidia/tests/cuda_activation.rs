#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use engine_nvidia::{CudaQ8_1Quantizer, CudaQuantizedKernelError};

#[test]
#[ignore = "requires CUDA GPU and NVRTC"]
fn packs_q8_1_layout_rounding_and_partial_launch() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let quantizer = CudaQ8_1Quantizer::new(stream.clone()).unwrap();
    // Five blocks exercise a partially occupied final thread block. Exact
    // powers of two make half scale/sum expectations independent of a decoder.
    let mut values = vec![0.0_f32; 160];
    for block in 1..5 {
        let scale = [0.0, 1.0, 0.5, 2.0, 0.25][block];
        values[block * 32] = 127.0 * scale;
        values[block * 32 + 1] = -127.0 * scale;
        values[block * 32 + 2] = 0.5 * scale;
        values[block * 32 + 3] = -0.5 * scale;
        values[block * 32 + 4] = 1.5 * scale;
        values[block * 32 + 5] = -1.5 * scale;
        values[block * 32 + 31] = 2.0 * scale;
    }
    let input = stream.clone_htod(&values).unwrap();
    let mut output = stream.clone_htod(&[u32::MAX; 45]).unwrap();
    quantizer.execute(&input, &mut output).unwrap();
    let actual = stream.clone_dtoh(&output).unwrap();
    assert_eq!(&actual[..9], &[0; 9]);
    for (block, header) in [0x4000_3c00, 0x3c00_3800, 0x4400_4000, 0x3800_3400]
        .into_iter()
        .enumerate()
    {
        let words = &actual[(block + 1) * 9..(block + 2) * 9];
        assert_eq!(words[0], header);
        assert_eq!(words[1], 0xff01_817f);
        assert_eq!(words[2], 0x0000_fe02);
        assert_eq!(&words[3..8], &[0; 5]);
        assert_eq!(words[8], 0x0200_0000);
    }
    // Caller-owned scratch can be reused on the same stream.
    quantizer.execute(&input, &mut output).unwrap();
    assert_eq!(stream.clone_dtoh(&output).unwrap(), actual);
}

#[test]
#[ignore = "requires CUDA GPU and NVRTC"]
fn rejects_invalid_q8_1_storage_before_launch() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let quantizer = CudaQ8_1Quantizer::new(stream.clone()).unwrap();
    let partial = stream.clone_htod(&[0.0_f32; 31]).unwrap();
    let input = stream.clone_htod(&[0.0_f32; 32]).unwrap();
    let mut output = stream.alloc_zeros::<u32>(9).unwrap();
    assert!(matches!(
        quantizer.execute(&partial, &mut output),
        Err(CudaQuantizedKernelError::InputLength { .. })
    ));
    let mut short = stream.alloc_zeros::<u32>(8).unwrap();
    assert!(matches!(
        quantizer.execute(&input, &mut short),
        Err(CudaQuantizedKernelError::OutputLength { .. })
    ));
    let empty = stream.alloc_zeros::<f32>(0).unwrap();
    assert!(matches!(
        quantizer.execute(&empty, &mut output),
        Err(CudaQuantizedKernelError::InputLength { .. })
    ));
}
