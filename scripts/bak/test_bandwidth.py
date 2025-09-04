import torch
import time
import numpy as np
import ctypes

def measure_time(fn, iters=10):
    # Warm up once
    fn()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    end = time.time()
    return (end - start) / iters


def benchmark(size_mb=256):
    print(f"\n--- Benchmark with {size_mb} MB ---")

    numel = (size_mb * 1024 * 1024) // 4  # float32
    size_bytes = numel * 4

    # ---------- 1. Default .cuda() ----------
    src = torch.randn(numel, dtype=torch.float32)
    dst = torch.empty_like(src, device="cuda")

    def run_default():
        dst.copy_(src, non_blocking=True)

    t = measure_time(run_default)
    bw = size_bytes / t / 1e9
    print(f"[Default .cuda()]   {bw:.2f} GB/s")

    # ---------- 2. UVM + Prefetch ----------
    uvm_pool = torch.cuda.MemPool(torch.cuda.UvmAllocator())
    with torch.cuda.use_mem_pool(uvm_pool):
        uvm_tensor = torch.randn(numel, dtype=torch.float32)

    # Prefetch helper
    libcudart = ctypes.CDLL("libcudart.so")
    cudaMemPrefetchAsync = libcudart.cudaMemPrefetchAsync
    cudaMemPrefetchAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    cudaMemPrefetchAsync.restype = int

    def run_uvm_prefetch():
        ptr = uvm_tensor.data_ptr()
        cudaMemPrefetchAsync(ctypes.c_void_p(ptr), size_bytes, torch.cuda.current_device(), None)
        torch.cuda._sleep(100)  # simulate GPU touch

    t = measure_time(run_uvm_prefetch)
    bw = size_bytes / t / 1e9
    print(f"[UVM + Prefetch]    {bw:.2f} GB/s")

    # ---------- 3. Direct mapped HBM ----------
    gpu_tensor = torch.empty(numel, device="cuda", dtype=torch.float32)
    addr = gpu_tensor.data_ptr()

    # Build numpy array view pointing into GPU HBM
    cpu_view = np.ctypeslib.as_array((ctypes.c_float * numel).from_address(addr))
    host_data = np.random.randn(numel).astype(np.float32)

    def run_direct_map():
        cpu_view[:] = host_data  # CPU writes directly into GPU HBM
        _ = gpu_tensor.sum()    # GPU consumes

    t = measure_time(run_direct_map)
    bw = size_bytes / t / 1e9
    print(f"[Direct mapped HBM] {bw:.2f} GB/s")


if __name__ == "__main__":
    torch.cuda.init()
    for mb in [64, 256, 1024]:  # try different sizes
        benchmark(mb)

