import cupy as cp
import numpy as np
import time

# --- Key Parameters (Adjustable) ---
DATA_SIZE_GB = 8
NUM_STREAMS = 8
NUM_ITERATIONS = 10
NUM_WARMUP = 3

# --------------------------

def test_multistream_bandwidth_cupy_corrected():
    """Tests CPU->GPU bandwidth using CuPy and multiple CUDA streams (Corrected Version)"""
    total_size_bytes = int(DATA_SIZE_GB * (1024**3))
    chunk_size_bytes = total_size_bytes // NUM_STREAMS
    
    if total_size_bytes % NUM_STREAMS != 0:
        raise ValueError("Total data size must be divisible by the number of streams")

    print("="*50)
    print(f"CuPy Multi-Stream Bandwidth Test (Corrected)")
    print(f"Total Data Size: {DATA_SIZE_GB} GB")
    print(f"Number of Streams: {NUM_STREAMS}")
    print(f"Chunk Size per Stream: {chunk_size_bytes / (1024**2):.2f} MB")
    print("="*50)

    # ## --- FIX 1: Use high-level API to allocate a CuPy array in pinned memory ---
    # This is safer and more idiomatic than using alloc_pinned_memory.
    h_data_pinned = cp.empty_pinned(total_size_bytes, dtype=cp.uint8)
    # You can fill it with data just like a regular array
    h_data_pinned[:] = cp.random.randint(0, 255, size=total_size_bytes, dtype=cp.uint8)

    # Allocate device memory for the destination
    d_data = cp.empty(total_size_bytes, dtype=cp.uint8)

    # Create CUDA streams
    streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]

    # Create CUDA events for accurate timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    timings = []

    print(f"Starting test ({NUM_WARMUP} warmup, {NUM_ITERATIONS} timed iterations)...")
    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        cp.cuda.Device().synchronize()
        
        start_event.record()

        # Issue copy commands asynchronously in their respective streams
        for j in range(NUM_STREAMS):
            with streams[j]:
                offset = j * chunk_size_bytes
                
                # ## --- FIX 2: Use high-level cp.copyto for the asynchronous transfer ---
                # cp.copyto respects the stream context, making the copy asynchronous.
                # We copy a "slice" or "view" of the host array to the device array.
                src_chunk = h_data_pinned[offset : offset + chunk_size_bytes]
                dst_chunk = d_data[offset : offset + chunk_size_bytes]
                cp.copyto(dst_chunk, src_chunk)
        
        end_event.record()

        # Wait for all operations in all streams to complete
        cp.cuda.Device().synchronize()

        # Record timing
        if i >= NUM_WARMUP:
            elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            timings.append(elapsed_ms)
            print(f"  Iteration {i - NUM_WARMUP + 1}/{NUM_ITERATIONS}: {elapsed_ms:.2f} ms")

    # Calculate and report results
    avg_time_ms = sum(timings) / len(timings)
    avg_time_s = avg_time_ms / 1000.0
    bandwidth_gbps = DATA_SIZE_GB / avg_time_s

    print("-"*50)
    print(f"Average Time: {avg_time_ms:.2f} ms")
    print(f"Calculated Bandwidth: {bandwidth_gbps:.2f} GB/s")
    print("-"*50)

if __name__ == "__main__":
    test_multistream_bandwidth_cupy_corrected()
