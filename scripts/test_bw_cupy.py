import cupy as cp
import numpy as np
import time

# --- 关键参数 (可调整) ---
DATA_SIZE_GB = 8
NUM_STREAMS = 8  # <--- 尝试调整这个值，例如 2, 4, 8, 16
NUM_ITERATIONS = 10
NUM_WARMUP = 3

# --------------------------

def test_multistream_bandwidth():
    """使用 CuPy 和多个 CUDA 流测试 CPU->GPU 的带宽"""
    total_size_bytes = int(DATA_SIZE_GB * (1024**3))
    chunk_size_bytes = total_size_bytes // NUM_STREAMS
    
    if total_size_bytes % NUM_STREAMS != 0:
        raise ValueError("数据总大小必须能被流的数量整除")

    print("="*50)
    print(f"CuPy 多流带宽测试")
    print(f"总数据大小: {DATA_SIZE_GB} GB")
    print(f"流 (Streams) 数量: {NUM_STREAMS}")
    print(f"每个流的数据块大小: {chunk_size_bytes / (1024**2):.2f} MB")
    print("="*50)

    # 1. 在 CPU 上分配固定内存 (Pinned Memory)
    # 这是实现高带宽的关键
    try:
        h_data_ptr = cp.cuda.alloc_pinned_memory(total_size_bytes)
        # 创建一个 numpy 视图来填充数据
        h_data_np = np.frombuffer(h_data_ptr, dtype=np.uint8, count=total_size_bytes)
        h_data_np[:] = np.random.randint(0, 255, size=total_size_bytes, dtype=np.uint8)
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"分配固定内存失败: {e}")
        print("请检查系统是否有足够的内存，或者是否权限不足。")
        return

    # 2. 在 GPU 上为每个流分配目标内存
    d_chunks = [cp.empty(chunk_size_bytes, dtype=cp.uint8) for _ in range(NUM_STREAMS)]

    # 3. 创建 CUDA 流
    streams = [cp.cuda.Stream() for _ in range(NUM_STREAMS)]

    # 4. 创建 CUDA 事件用于精确计时
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    timings = []

    print(f"开始执行测试 ({NUM_WARMUP} 次预热, {NUM_ITERATIONS} 次计时)...")
    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        # 将CPU执行同步，确保上一轮的清理工作完成
        cp.cuda.Device().synchronize()
        
        start_event.record()

        # 5. 在各自的流中异步地发出拷贝命令
        for j in range(NUM_STREAMS):
            offset = j * chunk_size_bytes
            # 使用 with stream[j] 确保操作在正确的流中
            with streams[j]:
                # 从固定内存的正确偏移处拷贝对应的数据块
                d_chunks[j].data.copy_from_host_async(h_data_ptr + offset, chunk_size_bytes)
        
        end_event.record()

        # 6. 等待所有流中的所有操作完成
        cp.cuda.Device().synchronize()

        # 记录时间
        if i >= NUM_WARMUP:
            elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            timings.append(elapsed_ms)
            print(f"  迭代 {i - NUM_WARMUP + 1}/{NUM_ITERATIONS}: {elapsed_ms:.2f} ms")

    # 7. 清理固定内存
    cp.cuda.free_pinned_memory(h_data_ptr)

    # 8. 计算并报告结果
    avg_time_ms = sum(timings) / len(timings)
    avg_time_s = avg_time_ms / 1000.0
    bandwidth_gbps = DATA_SIZE_GB / avg_time_s

    print("-"*50)
    print(f"平均时间: {avg_time_ms:.2f} ms")
    print(f"计算带宽: {bandwidth_gbps:.2f} GB/s")
    print("-"*50)

if __name__ == "__main__":
    test_multistream_bandwidth()
