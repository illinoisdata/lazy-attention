#!/usr/bin/env python3
"""
GH200 C2C传输性能测试脚本
目标：接近理论峰值900GB/s
"""

import cupy as cp
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

class GH200C2CTest:
    def __init__(self):
        self.gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"检测到 {self.gpu_count} 个GPU设备")
        
        # 设置CUDA环境变量以优化性能
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '32'
        os.environ['CUDA_MEMORY_POOL_DISABLED'] = '0'
        
        # 初始化所有GPU
        self.devices = []
        self.streams = []
        for i in range(self.gpu_count):
            self.devices.append(cp.cuda.Device(i))
            with cp.cuda.Device(i):
                self.streams.append(cp.cuda.Stream(non_blocking=True))
    
    def setup_memory_pools(self):
        """设置内存池以减少分配开销"""
        for i in range(self.gpu_count):
            with cp.cuda.Device(i):
                # 预分配大块内存池
                pool = cp.get_default_memory_pool()
                pool.set_limit(size=32 * 1024**3)  # 32GB限制
    
    def create_test_data(self, size_gb=8):
        """创建测试数据"""
        # 计算每个buffer的大小（字节）
        buffer_size = int(size_gb * 1024**3)
        element_count = buffer_size // 4  # float32
        
        print(f"创建 {size_gb}GB 测试数据...")
        
        # 在每个GPU上创建数据
        gpu_arrays = []
        for i in range(self.gpu_count):
            with cp.cuda.Device(i):
                # 创建随机数据
                arr = cp.random.random((element_count,), dtype=cp.float32)
                gpu_arrays.append(arr)
        
        return gpu_arrays, buffer_size
    
    def peer_to_peer_test(self, arrays, buffer_size, iterations=10):
        """点对点传输测试"""
        print(f"开始P2P传输测试，缓冲区大小: {buffer_size/1024**3:.2f}GB")
        
        # 启用P2P访问
        for i in range(self.gpu_count):
            for j in range(self.gpu_count):
                if i != j:
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(j, 0)
                    except:
                        pass  # 可能已经启用
        
        total_bytes = 0
        total_time = 0
        
        for iteration in range(iterations):
            start_time = time.perf_counter()
            
            # 并行执行多个传输
            futures = []
            with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
                for i in range(self.gpu_count):
                    for j in range(self.gpu_count):
                        if i != j:
                            future = executor.submit(
                                self._transfer_data, 
                                arrays[i], i, j, self.streams[i]
                            )
                            futures.append(future)
                
                # 等待所有传输完成
                for future in futures:
                    future.result()
            
            # 同步所有流
            for stream in self.streams:
                stream.synchronize()
            
            end_time = time.perf_counter()
            iteration_time = end_time - start_time
            iteration_bytes = buffer_size * len(futures)
            
            total_bytes += iteration_bytes
            total_time += iteration_time
            
            bandwidth_gbps = (iteration_bytes / iteration_time) / 1024**3
            print(f"迭代 {iteration+1}: {bandwidth_gbps:.2f} GB/s")
        
        avg_bandwidth = (total_bytes / total_time) / 1024**3
        print(f"平均带宽: {avg_bandwidth:.2f} GB/s")
        return avg_bandwidth
    
    def _transfer_data(self, src_array, src_gpu, dst_gpu, stream):
        """执行单个数据传输"""
        with cp.cuda.Device(src_gpu):
            # 创建目标缓冲区
            with cp.cuda.Device(dst_gpu):
                dst_array = cp.empty_like(src_array)
            
            # 异步传输
            with stream:
                dst_array[:] = src_array[:]
    
    def all_to_all_test(self, arrays, buffer_size, iterations=5):
        """全对全传输测试（模拟AllReduce等集合通信）"""
        print(f"开始All-to-All传输测试")
        
        total_bytes = 0
        total_time = 0
        
        for iteration in range(iterations):
            start_time = time.perf_counter()
            
            # 创建传输任务列表
            transfer_tasks = []
            for i in range(self.gpu_count):
                for j in range(self.gpu_count):
                    if i != j:
                        transfer_tasks.append((i, j))
            
            # 并行执行所有传输
            with ThreadPoolExecutor(max_workers=len(transfer_tasks)) as executor:
                futures = [
                    executor.submit(
                        self._transfer_data,
                        arrays[src], src, dst, self.streams[src % len(self.streams)]
                    )
                    for src, dst in transfer_tasks
                ]
                
                # 等待完成
                for future in futures:
                    future.result()
            
            # 同步所有GPU
            for i in range(self.gpu_count):
                with cp.cuda.Device(i):
                    cp.cuda.Stream.null.synchronize()
            
            end_time = time.perf_counter()
            iteration_time = end_time - start_time
            iteration_bytes = buffer_size * len(transfer_tasks)
            
            total_bytes += iteration_bytes
            total_time += iteration_time
            
            bandwidth_gbps = (iteration_bytes / iteration_time) / 1024**3
            print(f"All-to-All 迭代 {iteration+1}: {bandwidth_gbps:.2f} GB/s")
        
        avg_bandwidth = (total_bytes / total_time) / 1024**3
        print(f"All-to-All 平均带宽: {avg_bandwidth:.2f} GB/s")
        return avg_bandwidth
    
    def streaming_test(self, arrays, buffer_size, iterations=10):
        """流水线传输测试"""
        print("开始流水线传输测试")
        
        # 将每个数组分成多个块进行流水线传输
        chunk_count = 8
        chunk_size = len(arrays[0]) // chunk_count
        
        total_bytes = 0
        total_time = 0
        
        for iteration in range(iterations):
            start_time = time.perf_counter()
            
            # 创建多个流进行并行传输
            streams_per_gpu = []
            for i in range(self.gpu_count):
                with cp.cuda.Device(i):
                    gpu_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(chunk_count)]
                    streams_per_gpu.append(gpu_streams)
            
            # 并行传输所有块
            with ThreadPoolExecutor(max_workers=self.gpu_count * chunk_count) as executor:
                futures = []
                for i in range(self.gpu_count):
                    for j in range(self.gpu_count):
                        if i != j:
                            for chunk_idx in range(chunk_count):
                                start_idx = chunk_idx * chunk_size
                                end_idx = min((chunk_idx + 1) * chunk_size, len(arrays[i]))
                                
                                future = executor.submit(
                                    self._transfer_chunk,
                                    arrays[i][start_idx:end_idx],
                                    i, j, streams_per_gpu[i][chunk_idx]
                                )
                                futures.append(future)
                
                # 等待所有传输完成
                for future in futures:
                    future.result()
            
            # 同步所有流
            for gpu_streams in streams_per_gpu:
                for stream in gpu_streams:
                    stream.synchronize()
            
            end_time = time.perf_counter()
            iteration_time = end_time - start_time
            iteration_bytes = (buffer_size // chunk_count) * len(futures)
            
            total_bytes += iteration_bytes
            total_time += iteration_time
            
            bandwidth_gbps = (iteration_bytes / iteration_time) / 1024**3
            print(f"流水线迭代 {iteration+1}: {bandwidth_gbps:.2f} GB/s")
        
        avg_bandwidth = (total_bytes / total_time) / 1024**3
        print(f"流水线平均带宽: {avg_bandwidth:.2f} GB/s")
        return avg_bandwidth
    
    def _transfer_chunk(self, src_chunk, src_gpu, dst_gpu, stream):
        """传输数据块"""
        with cp.cuda.Device(dst_gpu):
            dst_chunk = cp.empty_like(src_chunk)
            with stream:
                dst_chunk[:] = src_chunk[:]
    
    def system_info(self):
        """显示系统信息"""
        print("=== 系统信息 ===")
        print(f"CPU: {psutil.cpu_count()} 核心")
        print(f"内存: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        
        for i in range(self.gpu_count):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                mem_info = cp.cuda.runtime.memGetInfo()
                print(f"GPU {i}: {props['name'].decode()}")
                print(f"  显存: {mem_info[1] / 1024**3:.1f} GB")
                print(f"  计算能力: {props['major']}.{props['minor']}")
        print()
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("=== GH200 C2C 传输性能测试 ===\n")
        
        self.system_info()
        self.setup_memory_pools()
        
        # 测试不同大小的数据
        test_sizes = [2, 4, 8, 16]  # GB
        results = {}
        
        for size_gb in test_sizes:
            print(f"\n=== 测试数据大小: {size_gb} GB ===")
            arrays, buffer_size = self.create_test_data(size_gb)
            
            # P2P测试
            p2p_bw = self.peer_to_peer_test(arrays, buffer_size)
            
            # All-to-All测试
            all_to_all_bw = self.all_to_all_test(arrays, buffer_size)
            
            # 流水线测试
            streaming_bw = self.streaming_test(arrays, buffer_size)
            
            results[size_gb] = {
                'p2p': p2p_bw,
                'all_to_all': all_to_all_bw,
                'streaming': streaming_bw
            }
            
            # 清理显存
            del arrays
            cp.get_default_memory_pool().free_all_blocks()
        
        # 显示最终结果
        print("\n=== 最终结果汇总 ===")
        print("数据大小(GB) | P2P(GB/s) | All-to-All(GB/s) | 流水线(GB/s)")
        print("-" * 60)
        for size, res in results.items():
            print(f"{size:11} | {res['p2p']:8.2f} | {res['all_to_all']:12.2f} | {res['streaming']:10.2f}")
        
        # 找出最佳性能
        max_bw = 0
        best_config = ""
        for size, res in results.items():
            for test_type, bw in res.items():
                if bw > max_bw:
                    max_bw = bw
                    best_config = f"{size}GB {test_type}"
        
        print(f"\n最佳性能: {max_bw:.2f} GB/s ({best_config})")
        print(f"理论峰值利用率: {max_bw/900*100:.1f}%")

def main():
    try:
        # 检查CuPy是否可用
        import cupy as cp
        print(f"CuPy版本: {cp.__version__}")
        print(f"CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
        
        # 运行测试
        tester = GH200C2CTest()
        tester.run_comprehensive_test()
        
    except ImportError:
        print("错误: 需要安装CuPy库")
        print("安装命令: pip install cupy-cuda12x")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
