//! Gate-1 tile-track probe: an Engine-authored cuTile kernel running over an
//! allocation that the probe owns.
//!
//! cuTile's `Tensor::from_foreign` is its documented interop entry point: it
//! wraps device memory owned by an external framework, holds the owner alive so
//! the mapping outlives every use, and takes no copy and no ownership transfer.
//! This probe holds it to that: cuda-core allocates and frees, cuTile borrows,
//! and the buffer's contents are read back through cuda-core after the kernel.

use std::sync::Arc;

use cuda_async::device_buffer::DeviceAllocation;
use cuda_async::device_operation::DeviceOp;
use cutile::prelude::*;

const N: usize = 1024;
const TILE: usize = 128;
const EXPECTED: f32 = 2.0;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// `z = x + x`, one program per tile of `z`.
    #[cutile::entry()]
    fn double<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let tx = x.load_like(z);
        z.store(tx + tx);
    }
}

/// A raw cuda-core allocation exposed to cuTile as a foreign owner.
///
/// cuTile holds this alive for as long as any tensor borrows it and never frees
/// it, so `free_async` at the end of `main` is the allocation's only
/// deallocation.
struct ProbeAllocation {
    dptr: cuda_core::sys::CUdeviceptr,
    len_bytes: usize,
    device_id: usize,
}

// SAFETY: `dptr` is a live `cuMemAllocAsync` allocation on `device_id` for
// `len_bytes` bytes, and this struct outlives every tensor built from it.
unsafe impl DeviceAllocation for ProbeAllocation {
    fn device_ptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.dptr
    }

    fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    fn device_id(&self) -> usize {
        self.device_id
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;
    let bytes = N * std::mem::size_of::<f32>();

    // One owner: cuda-core allocates the buffer cuTile will write.
    let dptr = unsafe { cuda_core::malloc_async(bytes, &stream)? };
    let owner = Arc::new(ProbeAllocation {
        dptr,
        len_bytes: bytes,
        device_id: 0,
    });

    let borrowed: Arc<dyn DeviceAllocation> = owner.clone();
    let z = unsafe { Tensor::<f32>::from_foreign(borrowed, vec![N as i32], vec![1]) };
    let x = cutile::api::ones::<f32>(&[N]);

    // Launch on the same stream the owner allocated on, and let every tensor
    // drop before the owner reads or frees anything.
    let (z, x) = kernels::double(z.partition([TILE]), x).sync_on(&stream)?;
    drop(z);
    drop(x);

    let mut host = vec![0f32; N];
    unsafe { cuda_core::memcpy_dtoh_async(host.as_mut_ptr(), dptr, N, &stream)? };
    stream.synchronize()?;
    if !host.iter().all(|value| *value == EXPECTED) {
        let wrong = host.iter().filter(|value| **value != EXPECTED).count();
        return Err(format!("tile kernel left {wrong} of {N} elements wrong").into());
    }

    // Dropping the tensors above must have released the borrow; only then can
    // the owner be reclaimed and the allocation freed exactly once.
    let allocation =
        Arc::try_unwrap(owner).map_err(|_| "a tensor still holds the foreign owner alive")?;
    unsafe { cuda_core::free_async(allocation.dptr, &stream)? };
    stream.synchronize()?;

    println!("tile probe: cuTile wrote {N} f32 into a cuda-core allocation it borrowed — ok");
    Ok(())
}
