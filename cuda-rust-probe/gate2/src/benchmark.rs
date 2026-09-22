//! Bounded measurements; event intervals can include host launch gaps.
use std::{error::Error, sync::Arc, time::Instant};

pub type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;
const LAUNCHES: usize = 100;

pub fn rust(
    stream: &Arc<cuda_core::CudaStream>,
    mut launch: impl FnMut() -> Result,
) -> Result<f32> {
    let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT as u32;
    let start = stream.record_event(Some(flags))?;
    for _ in 0..LAUNCHES {
        launch()?;
    }
    let end = stream.record_event(Some(flags))?;
    end.synchronize()?;
    Ok(start.elapsed_ms(&end)? * 1000.0 / LAUNCHES as f32)
}

pub fn cpp(
    stream: &Arc<cudarc::driver::CudaStream>,
    mut launch: impl FnMut() -> Result,
) -> Result<f32> {
    let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let start = stream.record_event(Some(flags))?;
    for _ in 0..LAUNCHES {
        launch()?;
    }
    let end = stream.record_event(Some(flags))?;
    end.synchronize()?;
    Ok(start.elapsed_ms(&end)? * 1000.0 / LAUNCHES as f32)
}

pub fn paired(
    label: &str,
    mut rust: impl FnMut() -> Result<f32>,
    mut cpp: impl FnMut() -> Result<f32>,
) -> Result {
    for repetition in 0..7 {
        let (rust_us, cpp_us) = if repetition % 2 == 0 {
            (rust()?, cpp()?)
        } else {
            let cpp_us = cpp()?;
            (rust()?, cpp_us)
        };
        println!("TIMING {label} rep={repetition} rust_us={rust_us:.3} cpp_us={cpp_us:.3}");
    }
    Ok(())
}

// Run in a fresh process per component, with CUDA_CACHE_DISABLE=1 when measuring
// without persistent driver caching. Context creation and Rust AOT compilation
// are outside this interval; C++ first-call NVRTC compilation is inside it.
// Each iteration creates a fresh wrapper, not just a cached function handle.
pub fn preparation<T>(label: &str, mut prepare: impl FnMut() -> Result<T>) -> Result {
    for repetition in 0..8 {
        let start = Instant::now();
        let ready = prepare()?;
        let us = start.elapsed().as_secs_f64() * 1e6;
        println!("PREPARATION {label} rep={repetition} module_and_prepare_us={us:.3}");
        drop(std::hint::black_box(ready));
    }
    Ok(())
}
