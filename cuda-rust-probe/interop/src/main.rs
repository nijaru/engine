//! Gate-1 resource-interoperability smoke test.
//!
//! Two CUDA libraries, one context, one stream, one allocation, one
//! deallocation owner. See `../README.md` and `docs/cuda-rust-migration.md`.

use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;

/// Elements in every case, not bytes.
const N: usize = 1024;
/// Word pattern written through cudarc's raw bindings in case A.
const WORD_PATTERN: u32 = 0x1122_3344;
/// Byte fill written through cuda-core's API in case B.
const BYTE_FILL: u8 = 0x5a;

type Fallible = Result<(), Box<dyn std::error::Error>>;

fn main() -> Fallible {
    let name = device_line()?;
    println!("device: {name}\n");

    case_a_cuda_core_owns()?;
    case_b_cudarc_owns()?;

    println!("\ninterop smoke: both ownership directions passed");
    Ok(())
}

fn device_line() -> Result<String, Box<dyn std::error::Error>> {
    let ctx = cuda_core::CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    Ok(format!("{} (sm_{major}{minor})", ctx.device_name()?))
}

/// Case A: cuda-core owns the context, stream, and allocation.
///
/// cudarc's raw driver entry points — the binding Engine uses today — write
/// into that allocation on that stream. cudarc never allocates and never
/// frees; cuda-core frees exactly once, when `buf` drops.
fn case_a_cuda_core_owns() -> Fallible {
    use cuda_core::{CudaContext, DeviceBuffer};

    let ctx = CudaContext::new(0)?;
    ctx.bind_to_thread()?;
    let stream = ctx.default_stream();
    let buf: DeviceBuffer<u32> = DeviceBuffer::zeroed(&stream, N)?;

    let words = vec![WORD_PATTERN; N];
    let bytes = N * size_of::<u32>();
    let dptr = buf.cu_deviceptr() as cudarc::driver::sys::CUdeviceptr;
    let cu_stream = stream.cu_stream() as *mut c_void
        as *mut cudarc::driver::sys::CUstream_st
        as cudarc::driver::sys::CUstream;

    // Same context, same stream, same allocation: no host round trip between
    // the library that owns the resources and the one that borrows them.
    let result = unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dptr, words.as_ptr().cast(), bytes, cu_stream)
    };
    ensure(result, "cuMemcpyHtoDAsync_v2")?;

    // One synchronization, after both libraries' work is enqueued.
    stream.synchronize()?;

    let host = buf.to_host_vec(&stream)?;
    if host.len() != N || !host.iter().all(|word| *word == WORD_PATTERN) {
        return Err("case A: cudarc's write did not land in cuda-core's allocation".into());
    }

    println!("case A: cuda-core owns, cudarc wrote async over the borrowed stream — ok");
    drop(buf); // the single free, by the single owner
    Ok(())
}

/// Case B: cudarc owns the context, stream, and allocation.
///
/// cuda-core adopts all three through non-owning `borrow_with_owner` handles
/// and writes through them. The probe drops those handles before reading the
/// memory again, so a borrower that released what it does not own fails the
/// run rather than passing silently.
fn case_b_cudarc_owns() -> Fallible {
    use cudarc::driver::{CudaContext, DevicePtr};

    let ctx = CudaContext::new(0)?;
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let buf = stream.alloc_zeros::<u32>(N)?;

    let core_device = unsafe {
        cuda_core::Device::borrow_with_owner(
            ctx.cu_ctx() as *mut c_void,
            ctx.cu_device(),
            0,
            Arc::clone(&ctx) as Arc<dyn cuda_core::ForeignOwner>,
        )
    };
    let core_stream = unsafe {
        cuda_core::Stream::borrow_with_owner(
            stream.cu_stream() as *mut c_void,
            &core_device,
            Arc::clone(&stream) as Arc<dyn cuda_core::ForeignOwner>,
        )
    };

    let (dptr, _guard) = buf.device_ptr(&stream);
    let bytes = N * size_of::<u32>();
    let dptr = dptr as cuda_core::sys::CUdeviceptr;

    unsafe { cuda_core::memset_d8_async(dptr, BYTE_FILL, bytes, &core_stream)? };
    unsafe { cuda_core::memcpy_htod_async(dptr, [WORD_PATTERN; N].as_ptr(), N, &core_stream)? };
    stream.synchronize()?;

    let host = stream.memcpy_dtov(&buf)?;
    if host.len() != N || !host.iter().all(|word| *word == WORD_PATTERN) {
        return Err("case B: cuda-core's write did not land in cudarc's allocation".into());
    }

    // Release the borrower before the owner. If the borrowed handles really
    // are foreign, the allocation survives and still reads correctly.
    drop(core_stream);
    drop(core_device);
    let still_there = stream.memcpy_dtov(&buf)?;
    if still_there != host {
        return Err("case B: dropping cuda-core's borrowed handles disturbed cudarc's allocation".into());
    }

    println!("case B: cudarc owns, cuda-core borrowed and wrote, release order safe — ok");
    drop(buf); // the single free, by the single owner
    Ok(())
}

fn ensure(result: cudarc::driver::sys::CUresult, call: &str) -> Fallible {
    if result == 0 {
        return Ok(());
    }
    Err(format!("{call} failed with CUresult {result}").into())
}
