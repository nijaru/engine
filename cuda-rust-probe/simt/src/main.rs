//! Gate-1 SIMT-track probe: an Engine-authored cuda-oxide kernel on the 4090.
//!
//! The tile probe exercises cuTile borrowing someone else's allocation. This one
//! exercises the migration's target state instead: cuda-core owns the context,
//! the stream, and every buffer, and the kernel launch borrows them. There is no
//! second allocator and no second runtime to reconcile.
//!
//! Run from this directory:
//!
//! ```text
//! cargo oxide run
//! ```

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

const N: usize = 1024;
const BLOCK: u32 = 256;
const EXPECTED: f32 = 7.0;

#[cuda_module]
mod kernels {
    use super::*;

    /// `z[i] = 2 * x[i] + y[i]`, one thread per element.
    #[kernel]
    #[launch_bounds(BLOCK)]
    #[launch_contract(domain = 1, block = (BLOCK, 1, 1))]
    pub fn double_add(x: &[f32], y: &[f32], mut z: DisjointSlice<f32>) {
        let index = thread::index_1d();
        if let Some(slot) = z.get_mut(index) {
            let i = index.get();
            *slot = 2.0 * x[i] + y[i];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let x_host = vec![3.0f32; N];
    let y_host = vec![1.0f32; N];
    let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
    let y_dev = DeviceBuffer::from_host(&stream, &y_host)?;
    let mut z_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // `kernels` module above.
    let module = unsafe { kernels::load(&ctx)? };
    let geometry = LaunchConfig1D::new(N.div_ceil(BLOCK as usize) as u32, BLOCK, 0);
    let prepared = module.prepare_double_add(geometry)?;
    module.double_add(&stream, &prepared, &x_dev, &y_dev, &mut z_dev)?;

    let z_host = z_dev.to_host_vec(&stream)?;
    let wrong = z_host.iter().filter(|value| **value != EXPECTED).count();
    if wrong != 0 {
        return Err(format!("simt kernel left {wrong} of {N} elements wrong").into());
    }

    println!("simt probe: cuda-oxide kernel wrote {N} f32 through cuda-core buffers — ok");
    Ok(())
}
