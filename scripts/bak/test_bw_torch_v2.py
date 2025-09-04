import torch
import time

# --- 关键参数 (可调整) ---
DATA_SIZE_GB = 64
NUM_STREAMS = 64
NUM_ITERATIONS = 15
NUM_WARMUP = 3

# --------------------------

def test_multistream_bandwidth_pytorch_corrected():
    """使用 PyTorch 和多个 CUDA 流测试 CPU->GPU 的带宽 (修正版)"""
    if not torch.cuda.is_available():
        print("CUDA 不可用。")
        return

    total_size_bytes = int(DATA_SIZE_GB * (1024**3))
    
    print("="*50)
    print(f"PyTorch 多流带宽测试 (修正版)")
    print(f"总数据大小: {DATA_SIZE_GB} GB")
    print(f"流 (Streams) 数量: {NUM_STREAMS}")
    print("="*50)

    # 1. 在 CPU 上创建张量并锁定到固定内存 (Pinned Memory)
    h_data = torch.empty(total_size_bytes, dtype=torch.uint8, device='cpu').pin_memory()
    h_chunks = h_data.chunk(NUM_STREAMS)

    # 2. 在 GPU 上为每个流准备目标张量
    d_chunks = [
        torch.empty_like(chunk, device='cuda') for chunk in h_chunks
    ]

    # 3. 创建 CUDA 流
    streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]

    # 4. 创建 CUDA 事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    timings = []

    print(f"开始执行测试 ({NUM_WARMUP} 次预热, {NUM_ITERATIONS} 次计时)...")
    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        # 预热和迭代之间同步，确保状态干净
        torch.cuda.synchronize()
        
        # 在默认流上记录开始事件
        start_event.record()

        # 5. 在各自的侧边流中异步地发出拷贝命令
        for j in range(NUM_STREAMS):
            with torch.cuda.stream(streams[j]):
                d_chunks[j].copy_(h_chunks[j], non_blocking=True)
        
        ## --- 修正部分 START ---
        # 核心修正: 在记录结束事件之前，让默认流等待所有侧边流完成。
        # 这样可以确保 end_event 的时间戳是在所有拷贝都结束后才被记录的。
        default_stream = torch.cuda.default_stream()
        for s in streams:
            default_stream.wait_stream(s)
        ## --- 修正部分 END ---

        # 现在，当默认流执行到这里时，所有侧边流的工作都已完成
        end_event.record()

        # 等待 end_event 记录完成
        torch.cuda.synchronize()

        # 记录时间
        if i >= NUM_WARMUP:
            elapsed_ms = start_event.elapsed_time(end_event)
            timings.append(elapsed_ms)
            print(f"  迭代 {i - NUM_WARMUP + 1}/{NUM_ITERATIONS}: {elapsed_ms:.2f} ms")

    # 8. 计算并报告结果
    # 过滤掉可能的异常值，取稳定的平均值
    if len(timings) > 2:
        timings.sort()
        stable_timings = timings[1:-1] # 去掉一个最大值和一个最小值
        avg_time_ms = sum(stable_timings) / len(stable_timings)
    else:
        avg_time_ms = sum(timings) / len(timings)
        
    avg_time_s = avg_time_ms / 1000.0
    bandwidth_gbps = DATA_SIZE_GB / avg_time_s

    print("-"*50)
    print(f"平均时间 (稳定): {avg_time_ms:.2f} ms")
    print(f"计算带宽: {bandwidth_gbps:.2f} GB/s")
    print("-"*50)

if __name__ == "__main__":
    test_multistream_bandwidth_pytorch_corrected()
