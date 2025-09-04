#!/usr/bin/env python3
"""
GH200 CPU-GPU C2C传输性能测试脚本
测试Grace CPU与Hopper GPU之间的高速互连
理论峰值: 900GB/s
"""

import cupy as cp
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import gc

class GH200CPUGPUTest:
    def __init__(self):
        self.gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"检测到 {self.gpu_count} 个GPU设备")
        
        if self.gpu_count == 0:
            raise RuntimeError("未检测到GPU设备")
        
        # 设置CUDA环境变量优化传输性能
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '32'
        os.environ['CUDA_MEMORY_POOL_DISABLED'] = '0'
        
        # 使用第一个GPU
        self.device = cp.cuda.Device(0)
        with self.device:
            # 创建多个并行流以最大化带宽利用率
            self.num_streams = 16
            self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(self.num_streams)]
    
    def setup_memory_pool(self):
        """设置内存池"""
        with self.device:
            pool = cp.get_default_memory_pool()
            pool.set_limit(size=80 * 1024**3)  # 80GB限制
    
    def cpu_to_gpu_test(self, size_gb=8, iterations=10, use_pinned=True):
        """CPU到GPU传输测试"""
        print(f"\n=== CPU → GPU 传输测试 ({size_gb}GB) ===")
        print(f"使用固定内存: {'是' if use_pinned else '否'}")
        
        buffer_size = int(size_gb * 1024**3)
        element_count = buffer_size // 4  # float32
        
        # 创建CPU数据
        if use_pinned:
            # 使用CUDA固定内存 (pinned memory) 加速传输
            byte_size = element_count * 4  # float32 = 4 bytes
            cpu_data = cp.cuda.alloc_pinned_memory(byte_size)
            cpu_array = np.frombuffer(cpu_data, dtype=np.float32).reshape((element_count,))
            # 填充随机数据
            cpu_array[:] = np.random.random(element_count).astype(np.float32)
        else:
            # 普通CPU内存
            cpu_array = np.random.random(element_count).astype(np.float32)
        
        with self.device:
            # 预分配GPU内存
            gpu_array = cp.empty((element_count,), dtype=cp.float32)
            
            total_bytes = 0
            total_time = 0
            
            # 预热传输
            gpu_array[:] = cp.asarray(cpu_array)
            cp.cuda.Stream.null.synchronize()
            
            print("开始传输测试...")
            for i in range(iterations):
                start_time = time.perf_counter()
                
                # 异步传输
                with self.streams[0]:
                    gpu_array[:] = cp.asarray(cpu_array)
                
                self.streams[0].synchronize()
                end_time = time.perf_counter()
                
                iteration_time = end_time - start_time
                iteration_bytes = buffer_size
                
                total_bytes += iteration_bytes
                total_time += iteration_time
                
                bandwidth_gbps = (iteration_bytes / iteration_time) / 1024**3
                print(f"迭代 {i+1}: {bandwidth_gbps:.2f} GB/s")
            
            avg_bandwidth = (total_bytes / total_time) / 1024**3
            print(f"CPU→GPU 平均带宽: {avg_bandwidth:.2f} GB/s")
            
            # 清理
            del gpu_array
            if use_pinned:
                cp.cuda.free_pinned_memory(cpu_data)
            del cpu_array
            
            return avg_bandwidth
    
    def gpu_to_cpu_test(self, size_gb=8, iterations=10, use_pinned=True):
        """GPU到CPU传输测试"""
        print(f"\n=== GPU → CPU 传输测试 ({size_gb}GB) ===")
        print(f"使用固定内存: {'是' if use_pinned else '否'}")
        
        buffer_size = int(size_gb * 1024**3)
        element_count = buffer_size // 4
        
        with self.device:
            # 创建GPU数据
            gpu_array = cp.random.random((element_count,), dtype=cp.float32)
            
            # 准备CPU目标内存
            if use_pinned:
                byte_size = element_count * 4  # float32 = 4 bytes
                cpu_data = cp.cuda.alloc_pinned_memory(byte_size)
                cpu_array = np.frombuffer(cpu_data, dtype=np.float32).reshape((element_count,))
            else:
                cpu_array = np.empty((element_count,), dtype=np.float32)
            
            total_bytes = 0
            total_time = 0
            
            # 预热
            cpu_array[:] = cp.asnumpy(gpu_array)
            cp.cuda.Stream.null.synchronize()
            
            print("开始传输测试...")
            for i in range(iterations):
                start_time = time.perf_counter()
                
                # 传输到CPU
                with self.streams[0]:
                    cpu_array[:] = cp.asnumpy(gpu_array)
                
                self.streams[0].synchronize()
                end_time = time.perf_counter()
                
                iteration_time = end_time - start_time
                iteration_bytes = buffer_size
                
                total_bytes += iteration_bytes
                total_time += iteration_time
                
                bandwidth_gbps = (iteration_bytes / iteration_time) / 1024**3
                print(f"迭代 {i+1}: {bandwidth_gbps:.2f} GB/s")
            
            avg_bandwidth = (total_bytes / total_time) / 1024**3
            print(f"GPU→CPU 平均带宽: {avg_bandwidth:.2f} GB/s")
            
            # 清理
            del gpu_array
            if use_pinned:
                cp.cuda.free_pinned_memory(cpu_data)
            del cpu_array
            
            return avg_bandwidth
    
    def parallel_bidirectional_test(self, size_gb=4, iterations=10):
        """并行双向传输测试"""
        print(f"\n=== 并行双向传输测试 ({size_gb}GB) ===")
        
        buffer_size = int(size_gb * 1024**3)
        element_count = buffer_size // 4
        half_streams = self.num_streams // 2
        
        # 准备数据
        cpu_data_list = []
        cpu_arrays = []
        for i in range(half_streams):
            byte_size = element_count * 4  # float32 = 4 bytes
            cpu_data = cp.cuda.alloc_pinned_memory(byte_size)
            cpu_array = np.frombuffer(cpu_data, dtype=np.float32).reshape((element_count,))
            cpu_array[:] = np.random.random(element_count).astype(np.float32)
            cpu_data_list.append(cpu_data)
            cpu_arrays.append(cpu_array)
        
        with self.device:
            # GPU数据
            gpu_arrays_down = []  # CPU→GPU
            gpu_arrays_up = []    # GPU→CPU
            
            for i in range(half_streams):
                gpu_down = cp.empty((element_count,), dtype=cp.float32)
                gpu_up = cp.random.random((element_count,), dtype=cp.float32)
                gpu_arrays_down.append(gpu_down)
                gpu_arrays_up.append(gpu_up)
            
            # CPU接收数据
            cpu_recv_data_list = []
            cpu_recv_arrays = []
            for i in range(half_streams):
                byte_size = element_count * 4  # float32 = 4 bytes
                cpu_recv_data = cp.cuda.alloc_pinned_memory(byte_size)
                cpu_recv_array = np.frombuffer(cpu_recv_data, dtype=np.float32).reshape((element_count,))
                cpu_recv_data_list.append(cpu_recv_data)
                cpu_recv_arrays.append(cpu_recv_array)
            
            total_bytes = 0
            total_time = 0
            
            print("开始并行双向传输...")
            for iteration in range(iterations):
                start_time = time.perf_counter()
                
                # 启动并行传输任务
                def transfer_task(stream_idx):
                    # CPU → GPU (下行)
                    with self.streams[stream_idx]:
                        gpu_arrays_down[stream_idx][:] = cp.asarray(cpu_arrays[stream_idx])
                    
                    # GPU → CPU (上行)
                    with self.streams[stream_idx + half_streams]:
                        cpu_recv_arrays[stream_idx][:] = cp.asnumpy(gpu_arrays_up[stream_idx])
                
                # 并行执行传输
                with ThreadPoolExecutor(max_workers=half_streams) as executor:
                    futures = [executor.submit(transfer_task, i) for i in range(half_streams)]
                    for future in futures:
                        future.result()
                
                # 同步所有流
                for stream in self.streams:
                    stream.synchronize()
                
                end_time = time.perf_counter()
                
                iteration_time = end_time - start_time
                # 双向传输：下行 + 上行
                iteration_bytes = buffer_size * 2 * half_streams
                
                total_bytes += iteration_bytes
                total_time += iteration_time
                
                bandwidth_gbps = (iteration_bytes / iteration_time) / 1024**3
                print(f"迭代 {iteration+1}: {bandwidth_gbps:.2f} GB/s")
            
            avg_bandwidth = (total_bytes / total_time) / 1024**3
            print(f"并行双向平均带宽: {avg_bandwidth:.2f} GB/s")
            
            # 清理内存
            for gpu_down, gpu_up in zip(gpu_arrays_down, gpu_arrays_up):
                del gpu_down, gpu_up
            
            for cpu_data, cpu_recv_data in zip(cpu_data_list, cpu_recv_data_list):
                cp.cuda.free_pinned_memory(cpu_data)
                cp.cuda.free_pinned_memory(cpu_recv_data)
            
            return avg_bandwidth
    
    def streaming_transfer_test(self, size_gb=8, chunk_count=16, iterations=10):
        """流水线传输测试 - 将大数据分块并行传输"""
        print(f"\n=== 流水线传输测试 ({size_gb}GB, {chunk_count}块) ===")
        
        total_size = int(size_gb * 1024**3)
        chunk_size = total_size // chunk_count
        chunk_elements = chunk_size // 4
        
        # 准备分块数据
        cpu_chunks = []
        cpu_data_list = []
        for i in range(chunk_count):
            byte_size = chunk_elements * 4  # float32 = 4 bytes
            cpu_data = cp.cuda.alloc_pinned_memory(byte_size)
            cpu_chunk = np.frombuffer(cpu_data, dtype=np.float32).reshape((chunk_elements,))
            cpu_chunk[:] = np.random.random(chunk_elements).astype(np.float32)
            cpu_chunks.append(cpu_chunk)
            cpu_data_list.append(cpu_data)
        
        with self.device:
            # GPU接收缓冲区
            gpu_chunks = [cp.empty((chunk_elements,), dtype=cp.float32) for _ in range(chunk_count)]
            
            total_bytes = 0
            total_time = 0
            
            print("开始流水线传输...")
            for iteration in range(iterations):
                start_time = time.perf_counter()
                
                # 并行传输所有块
                def transfer_chunk(chunk_idx):
                    stream_idx = chunk_idx % self.num_streams
                    with self.streams[stream_idx]:
                        gpu_chunks[chunk_idx][:] = cp.asarray(cpu_chunks[chunk_idx])
                
                with ThreadPoolExecutor(max_workers=min(chunk_count, self.num_streams)) as executor:
                    futures = [executor.submit(transfer_chunk, i) for i in range(chunk_count)]
                    for future in futures:
                        future.result()
                
                # 同步所有流
                for stream in self.streams:
                    stream.synchronize()
                
                end_time = time.perf_counter()
                
                iteration_time = end_time - start_time
                iteration_bytes = total_size
                
                total_bytes += iteration_bytes
                total_time += iteration_time
                
                bandwidth_gbps = (iteration_bytes / iteration_time) / 1024**3
                print(f"迭代 {iteration+1}: {bandwidth_gbps:.2f} GB/s")
            
            avg_bandwidth = (total_bytes / total_time) / 1024**3
            print(f"流水线平均带宽: {avg_bandwidth:.2f} GB/s")
            
            # 清理
            for gpu_chunk in gpu_chunks:
                del gpu_chunk
            for cpu_data in cpu_data_list:
                cp.cuda.free_pinned_memory(cpu_data)
            
            return avg_bandwidth
    
    def system_info(self):
        """显示系统信息"""
        print("=== 系统信息 ===")
        print(f"CPU: {psutil.cpu_count()} 核心")
        print(f"内存: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        
        with self.device:
            props = cp.cuda.runtime.getDeviceProperties(0)
            mem_info = cp.cuda.runtime.memGetInfo()
            print(f"GPU: {props['name'].decode()}")
            print(f"  显存: {mem_info[1] / 1024**3:.1f} GB")
            print(f"  计算能力: {props['major']}.{props['minor']}")
        
        # 检查是否支持统一内存
        try:
            unified_addressing = cp.cuda.runtime.deviceGetAttribute(
                cp.cuda.runtime.deviceAttribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 0
            )
            print(f"  统一寻址支持: {'是' if unified_addressing else '否'}")
        except:
            pass
        
        print()
    
    def run_comprehensive_test(self):
        """运行全面的CPU-GPU传输测试"""
        print("=== GH200 CPU-GPU C2C 传输性能测试 ===\n")
        
        self.system_info()
        self.setup_memory_pool()
        
        results = {}
        
        # 测试不同大小的数据传输
        test_sizes = [1, 2, 4, 8, 16]  # GB
        
        for size_gb in test_sizes:
            print(f"\n{'='*50}")
            print(f"测试数据大小: {size_gb} GB")
            print(f"{'='*50}")
            
            try:
                # CPU → GPU 传输 (固定内存)
                cpu_to_gpu_pinned = self.cpu_to_gpu_test(size_gb, use_pinned=True)
                
                # CPU → GPU 传输 (普通内存)
                cpu_to_gpu_normal = self.cpu_to_gpu_test(size_gb, use_pinned=False)
                
                # GPU → CPU 传输 (固定内存)
                gpu_to_cpu_pinned = self.gpu_to_cpu_test(size_gb, use_pinned=True)
                
                # GPU → CPU 传输 (普通内存)
                gpu_to_cpu_normal = self.gpu_to_cpu_test(size_gb, use_pinned=False)
                
                # 并行双向传输
                bidirectional = self.parallel_bidirectional_test(size_gb // 2)
                
                # 流水线传输
                streaming = self.streaming_transfer_test(size_gb)
                
                results[size_gb] = {
                    'cpu_to_gpu_pinned': cpu_to_gpu_pinned,
                    'cpu_to_gpu_normal': cpu_to_gpu_normal,
                    'gpu_to_cpu_pinned': gpu_to_cpu_pinned,
                    'gpu_to_cpu_normal': gpu_to_cpu_normal,
                    'bidirectional': bidirectional,
                    'streaming': streaming
                }
                
                # 强制垃圾回收
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                
            except Exception as e:
                print(f"测试 {size_gb}GB 时出现错误: {e}")
                continue
        
        # 显示最终结果汇总
        print(f"\n{'='*80}")
        print("最终结果汇总")
        print(f"{'='*80}")
        print(f"{'大小(GB)':<8} | {'CPU→GPU固定':<12} | {'CPU→GPU普通':<12} | {'GPU→CPU固定':<12} | {'GPU→CPU普通':<12} | {'双向':<8} | {'流水线':<10}")
        print("-" * 80)
        
        for size, res in results.items():
            print(f"{size:<8} | {res['cpu_to_gpu_pinned']:<12.2f} | {res['cpu_to_gpu_normal']:<12.2f} | "
                  f"{res['gpu_to_cpu_pinned']:<12.2f} | {res['gpu_to_cpu_normal']:<12.2f} | "
                  f"{res['bidirectional']:<8.2f} | {res['streaming']:<10.2f}")
        
        # 找出最佳性能
        max_bandwidth = 0
        best_config = ""
        for size, res in results.items():
            for test_type, bandwidth in res.items():
                if bandwidth > max_bandwidth:
                    max_bandwidth = bandwidth
                    best_config = f"{size}GB {test_type}"
        
        print(f"\n最佳性能: {max_bandwidth:.2f} GB/s ({best_config})")
        print(f"理论峰值利用率: {max_bandwidth/900*100:.1f}% (目标: 900GB/s)")
        
        # 性能分析和建议
        print(f"\n{'='*50}")
        print("性能分析")
        print(f"{'='*50}")
        if max_bandwidth < 100:
            print("⚠️  传输性能较低，建议检查：")
            print("   - 确认使用固定内存 (pinned memory)")
            print("   - 增加并行流数量")
            print("   - 检查系统NUMA配置")
        elif max_bandwidth < 500:
            print("✅ 传输性能良好，可进一步优化：")
            print("   - 调整数据块大小")
            print("   - 优化内存访问模式")
        else:
            print("🚀 传输性能优秀！接近理论峰值")

def main():
    try:
        print(f"CuPy版本: {cp.__version__}")
        print(f"CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
        
        tester = GH200CPUGPUTest()
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
