# This script is used to test the gpu loading latency and recomputation latency on vllm
# Specifically optimized for GH200 chipset

# Date: 2025-08-26

import os
import time
import json
import argparse
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import vllm
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Data class to store benchmark results"""
    test_name: str
    model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    seq_length: int
    batch_size: int
    
    # Loading metrics
    model_loading_time: float
    gpu_memory_allocated: float
    gpu_memory_reserved: float
    
    # Inference metrics
    first_token_latency: float
    throughput_tokens_per_sec: float
    total_inference_time: float
    
    # Recomputation vs Loading metrics
    load_and_first_inference_time: Optional[float] = None
    warm_recompute_time: Optional[float] = None
    cached_computation_time: Optional[float] = None
    variation_recomputation_time: Optional[float] = None
    loading_overhead: Optional[float] = None
    
    # Legacy recomputation metrics (for backward compatibility)
    cache_hit_rate: Optional[float] = None
    recomputation_time: Optional[float] = None
    
    # GH200 specific metrics
    cpu_memory_usage: float = 0.0
    grace_cpu_utilization: float = 0.0


class GH200Benchmark:
    """Benchmark suite for GPU loading and recomputation latency on GH200"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 device: str = "cuda"):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.device = device
        self.llm = None
        self.results: List[BenchmarkResult] = []
    
    def check_gh200_environment(self) -> Dict[str, str]:
        """Check if running on GH200 and gather system info"""
        info = {}
        
        # Check GPU info
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
            info['torch_version'] = torch.__version__
            
            # Check if this is Grace Hopper (GH200)
            gpu_name = torch.cuda.get_device_name(0).lower()
            info['is_gh200'] = 'hopper' in gpu_name or 'grace' in gpu_name or 'gh200' in gpu_name
        
        # Check CPU info (Grace CPU in GH200)
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                info['cpu_model'] = 'unknown'
                for line in cpu_info.split('\n'):
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
        except:
            info['cpu_model'] = 'unknown'
        
        # Memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()
                for line in mem_info.split('\n'):
                    if 'MemTotal:' in line:
                        info['total_memory_gb'] = int(line.split()[1]) / 1024 / 1024
                        break
        except:
            info['total_memory_gb'] = 'unknown'
            
        return info
    
    def measure_model_loading(self) -> Tuple[float, float, float]:
        """Measure model loading time and memory usage"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        
        # NOTE(haocheng): here we disable CUDA graph, is it possible to enable it?
        # A: Disabling CUDA graphs is correct for this type of latency benchmark.
        # Load model with vLLM
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            device=self.device,
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs for more predictable timing
        )
        
        end_time = time.perf_counter()
        loading_time = end_time - start_time
        
        # Measure GPU memory usage
        gpu_memory_allocated = 0
        gpu_memory_reserved = 0
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB

        logger.info(f"Model loaded in {loading_time:.2f} seconds")
        logger.info(f"GPU Memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")

        return loading_time, gpu_memory_allocated, gpu_memory_reserved
    
    def measure_cpu_memory_usage(self) -> float:
        """Measure current CPU memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024**3  # GB
        except ImportError:
            logger.warning("Warning: psutil not available, CPU memory usage not measured")
            return 0.0
    
    def benchmark_inference_latency(self, prompts: List[str], 
                                  max_tokens: int = 100) -> Tuple[float, float, float]:
        """Benchmark inference latency and throughput"""
        if self.llm is None:
            raise ValueError("Model not loaded. Call measure_model_loading() first.")
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True
        )
        
        # Warmup run
        logger.info("Warming up...")
        _ = self.llm.generate(prompts[:1], sampling_params)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Actual benchmark
        logger.info(f"Benchmarking inference with {len(prompts)} prompts...")
        start_time = time.perf_counter()
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / total_time
        
        # Estimate first token latency (rough approximation)
        first_token_latency = total_time / len(prompts)

        logger.info(f"Total inference time: {total_time:.2f} seconds")
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"Average first token latency: {first_token_latency:.4f} seconds")

        return first_token_latency, throughput, total_time
    
    def benchmark_kv_cache_vs_recomputation(self, token_lengths: List[int], 
                                           max_tokens: int = 50, num_runs: int = 10) -> Dict[int, Dict[str, float]]:
        """Benchmark KV cache loading from CPU DRAM vs recomputation across different token lengths
        
        Uses simplified dummy tensors of the same size as KV cache for transfer measurements.
        """
        results = {}
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True
        )
        
        # Load model once at the beginning (model stays loaded)
        if self.llm is None:
            logger.info("Loading model for KV cache analysis...")
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                device=self.device,
                trust_remote_code=True,
                enforce_eager=True,
            )
        
        # Get model configuration for realistic KV cache simulation
        model_config = self.llm.llm_engine.model_config
        hidden_size = getattr(model_config.hf_config, 'hidden_size', 4096)
        num_attention_heads = getattr(model_config.hf_config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(model_config.hf_config, 'num_key_value_heads', num_attention_heads)
        head_dim = hidden_size // num_attention_heads
        
        logger.info(f"Model config: hidden_size={hidden_size}, num_kv_heads={num_key_value_heads}, head_dim={head_dim}")
        
        for token_length in token_lengths:
            logger.info(f"\n--- Testing KV cache vs recomputation for {token_length} tokens ---")
            
            # Generate different prompts for each measurement to avoid caching
            prompts = self.generate_test_prompts(token_length, 3)  # Generate 3 different prompts
            base_prompt = prompts[0]  # For warm measurements (consistent)
            cold_prompt = prompts[1]  # For cold measurements (unique)
            partial_prompt = prompts[2]  # For partial measurements (different)
            
            # Step 1: Cold recomputation (no KV cache available) - single measurement with unique prompt
            logger.info("Measuring cold recomputation (no KV cache)...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache to simulate cold start
            
            cold_start = time.perf_counter()
            outputs = self.llm.generate([cold_prompt], sampling_params)  # Use unique prompt for cold
            cold_recomputation_time = time.perf_counter() - cold_start
            
            # Step 2: Warm recomputation (KV cache available in GPU memory) - multiple runs for stability
            logger.info(f"Measuring warm recomputation (KV cache in GPU) - {num_runs} runs...")
            warm_times = []
            for run in range(num_runs):
                warm_start = time.perf_counter()
                _ = self.llm.generate([base_prompt], sampling_params)
                warm_time = time.perf_counter() - warm_start
                warm_times.append(warm_time)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            warm_recomputation_time = np.mean(warm_times)
            warm_std = np.std(warm_times)
            logger.info(f"  Warm recomputation: {warm_recomputation_time:.4f}s ± {warm_std:.4f}s")
            
            # Step 3: Simplified KV cache transfer simulation - multiple runs for stability
            logger.info(f"Measuring simplified KV cache transfer (CPU DRAM ↔ GPU) - {num_runs} runs...")
            kv_transfer_times = []
            
            # Calculate KV cache size based on model architecture
            # KV cache shape: [batch_size, num_kv_heads, seq_len, head_dim] for both key and value
            batch_size = 1
            kv_shape = (batch_size, num_key_value_heads, token_length, head_dim)
            total_elements = 2 * np.prod(kv_shape)  # 2 tensors (key + value)
            total_bytes = total_elements * 2  # fp16 = 2 bytes per element
            
            logger.info(f"  Simulating KV cache transfer: Key{kv_shape} + Value{kv_shape}")
            logger.info(f"  Total KV cache size: ~{total_bytes / 1024**2:.1f} MB (fp16)")
            
            for run in range(num_runs):
                # Create dummy tensors of same size as KV cache
                dummy_tensor = torch.randn(total_elements, dtype=torch.float16, device='cuda')
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Measure CPU ↔ GPU transfer time
                transfer_start = time.perf_counter()
                
                # Transfer to CPU and back to GPU to simulate KV cache reload
                dummy_cpu = dummy_tensor.cpu()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                dummy_gpu = dummy_cpu.cuda(non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for transfer completion
                
                transfer_time = time.perf_counter() - transfer_start
                kv_transfer_times.append(transfer_time)
                
                # Clean up
                del dummy_tensor, dummy_cpu, dummy_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if run == 0:  # Log details for first run
                    logger.info(f"    Run {run+1}: Transfer time={transfer_time:.4f}s")
            
            kv_transfer_time = np.mean(kv_transfer_times)
            kv_transfer_std = np.std(kv_transfer_times)
            logger.info(f"  KV cache transfer (CPU→GPU): {kv_transfer_time:.4f}s ± {kv_transfer_std:.4f}s")
            
            # Calculate transfer bandwidth
            bandwidth_gbps = (total_bytes / 1024**3) / kv_transfer_time if kv_transfer_time > 0 else 0
            logger.info(f"  Transfer bandwidth: ~{bandwidth_gbps:.1f} GB/s")
            
            # Step 4: Full recomputation vs partial recomputation - multiple runs for stability
            logger.info(f"Measuring partial recomputation (different prompt) - {num_runs} runs...")
            
            partial_times = []
            for run in range(num_runs):
                partial_start = time.perf_counter()
                _ = self.llm.generate([partial_prompt], sampling_params)  # Use different prompt
                partial_time = time.perf_counter() - partial_start
                partial_times.append(partial_time)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            partial_recomputation_time = np.mean(partial_times)
            partial_std = np.std(partial_times)
            
            # Calculate KV cache transfer overhead and pure recomputation time
            pure_recomputation_time = cold_recomputation_time - warm_recomputation_time  # This is the actual recomputation cost
            
            results[token_length] = {
                'cold_recomputation': cold_recomputation_time,
                'warm_recomputation': warm_recomputation_time,  # Fully cached inference time
                'warm_recomputation_std': warm_std,
                'kv_cache_transfer_time': kv_transfer_time,  # Real transfer time from CPU DRAM to GPU
                'kv_cache_transfer_std': kv_transfer_std,
                'kv_cache_transfer_bandwidth_gbps': bandwidth_gbps,
                'partial_recomputation': partial_recomputation_time,
                'partial_recomputation_std': partial_std,
                'kv_cache_transfer_overhead': kv_transfer_time,  # For backward compatibility
                'pure_recomputation_time': max(0, pure_recomputation_time),  # Key metric: actual recomputation cost
                'recomputation_vs_transfer_ratio': pure_recomputation_time / kv_transfer_time if kv_transfer_time > 0 else float('inf'),
                'cache_efficiency': (cold_recomputation_time - warm_recomputation_time) / cold_recomputation_time if cold_recomputation_time > 0 else 0,
                'num_runs': num_runs,
                'kv_cache_size_mb': (total_bytes / 1024**2),
                'measurement_stability': {
                    'warm_cv': (warm_std / warm_recomputation_time * 100) if warm_recomputation_time > 0 else 0,  # Coefficient of variation
                    'kv_transfer_cv': (kv_transfer_std / kv_transfer_time * 100) if kv_transfer_time > 0 else 0,
                    'partial_cv': (partial_std / partial_recomputation_time * 100) if partial_recomputation_time > 0 else 0,
                }
            }
            
            logger.info(f"Results for {token_length} tokens (averaged over {num_runs} runs):")
            logger.info(f"  Cold Recomputation (unique prompt): {cold_recomputation_time:.4f}s")
            logger.info(f"  Warm Recomputation (repeated prompt): {warm_recomputation_time:.4f}s ± {warm_std:.4f}s (CV: {results[token_length]['measurement_stability']['warm_cv']:.1f}%)")
            logger.info(f"  Pure Recomputation Time: {results[token_length]['pure_recomputation_time']:.4f}s")
            logger.info(f"  KV Cache Transfer (CPU→GPU): {kv_transfer_time:.4f}s ± {kv_transfer_std:.4f}s (CV: {results[token_length]['measurement_stability']['kv_transfer_cv']:.1f}%)")
            logger.info(f"  KV Cache Size: {results[token_length]['kv_cache_size_mb']:.1f} MB")
            logger.info(f"  Transfer Bandwidth: {bandwidth_gbps:.1f} GB/s")
            logger.info(f"  Partial Recomputation: {partial_recomputation_time:.4f}s ± {partial_std:.4f}s (CV: {results[token_length]['measurement_stability']['partial_cv']:.1f}%)")
            logger.info(f"  Recomputation vs Transfer Ratio: {results[token_length]['recomputation_vs_transfer_ratio']:.2f}")
            logger.info(f"  --> Decision: {'Recompute' if results[token_length]['pure_recomputation_time'] < results[token_length]['kv_cache_transfer_overhead'] else 'Load KV Cache'}")
            
            # Warn if measurements are unstable
            if any(cv > 10 for cv in results[token_length]['measurement_stability'].values()):
                logger.warning(f"  ⚠️  High measurement variability detected! Consider increasing num_runs for more stable results.")
            
            # GH200-specific insights
            if bandwidth_gbps > 500:  # High bandwidth suggests NVLink/unified memory efficiency
                logger.info(f"  🚀 Excellent GH200 unified memory performance: {bandwidth_gbps:.1f} GB/s")
            elif bandwidth_gbps > 200:
                logger.info(f"  ✅ Good GH200 memory transfer performance: {bandwidth_gbps:.1f} GB/s")
            elif bandwidth_gbps > 100:
                logger.info(f"  ⚠️  Moderate transfer performance: {bandwidth_gbps:.1f} GB/s - check for bottlenecks")
            else:
                logger.warning(f"  🐌 Low transfer performance: {bandwidth_gbps:.1f} GB/s - potential memory bandwidth bottleneck")
        
        return results

    def analyze_kv_cache_intersection(self, kv_cache_results: Dict[int, Dict[str, float]]) -> Dict[str, any]:
        """Analyze the intersection point where KV cache loading becomes more efficient than recomputation"""
        analysis = {
            'intersection_points': {},
            'efficiency_recommendations': {},
            'performance_trends': {},
            'gh200_insights': {}
        }
        
        token_lengths = sorted(kv_cache_results.keys())
        
        # Analyze performance trends for each token length
        for token_length in token_lengths:
            results = kv_cache_results[token_length]
            
            cold_recompute = results['cold_recomputation']
            kv_transfer_time = results['kv_cache_transfer_time']
            warm_recompute = results['warm_recomputation']
            partial_recompute = results['partial_recomputation']
            pure_recomputation = results['pure_recomputation_time']  # The actual recomputation cost
            
            # Key comparison: pure recomputation time vs KV cache transfer time
            reload_vs_recompute_ratio = kv_transfer_time / pure_recomputation if pure_recomputation > 0 else float('inf')
            cache_efficiency = results['cache_efficiency']
            transfer_bandwidth = results.get('kv_cache_transfer_bandwidth_gbps', 0)
            
            # The main decision criterion: is pure recomputation faster than KV cache transfer?
            is_recompute_faster = pure_recomputation < kv_transfer_time
            
            analysis['performance_trends'][token_length] = {
                'is_recompute_faster_than_transfer': is_recompute_faster,
                'is_transfer_faster_than_cold': kv_transfer_time < cold_recompute,
                'pure_recomputation_time': pure_recomputation,
                'kv_transfer_time': kv_transfer_time,
                'transfer_bandwidth_gbps': transfer_bandwidth,
                'reload_vs_recompute_ratio': reload_vs_recompute_ratio,
                'cache_efficiency_pct': cache_efficiency * 100,
                'transfer_overhead_pct': (kv_transfer_time / cold_recompute * 100) if cold_recompute > 0 else 0, # FIXED: Added calculation
                'memory_bandwidth_critical': transfer_bandwidth < 100,  # Low bandwidth indicates bottleneck
                'decision': 'recompute' if is_recompute_faster else 'load_kv_cache'
            }
        
        # Find intersection points
        prev_recompute_faster = None
        
        for token_length in token_lengths:
            trends = analysis['performance_trends'][token_length]
            
            # Main intersection: pure recomputation vs KV cache transfer overhead
            is_recompute_faster = trends['is_recompute_faster_than_transfer']
            if prev_recompute_faster is not None and prev_recompute_faster != is_recompute_faster:
                analysis['intersection_points'][f'{token_length}_main'] = {
                    'type': 'recomputation_vs_kv_transfer',
                    'transition': 'transfer_to_recompute' if is_recompute_faster else 'recompute_to_transfer',
                    'token_length': token_length,
                    'description': f'Pure recomputation vs KV cache transfer overhead intersection'
                }
            prev_recompute_faster = is_recompute_faster
        
        # Generate strategy recommendations
        for token_length in token_lengths:
            trends = analysis['performance_trends'][token_length]
            results = kv_cache_results[token_length]
            
            pure_recomputation = results['pure_recomputation_time']
            kv_transfer_overhead = results['kv_cache_transfer_overhead']
            
            # Primary decision: pure recomputation vs KV cache transfer
            if pure_recomputation < kv_transfer_overhead:
                best_strategy = 'recomputation'
                best_time = pure_recomputation
                if pure_recomputation > 0:
                    speedup = kv_transfer_overhead / pure_recomputation
                    recommendation = f'Recompute faster by {speedup:.2f}x than KV transfer'
                else:
                    recommendation = 'Recomputation faster than KV transfer (near-zero recomputation time)'
            else:
                best_strategy = 'kv_cache_transfer'
                best_time = kv_transfer_overhead
                if kv_transfer_overhead > 0:
                    speedup = pure_recomputation / kv_transfer_overhead
                    recommendation = f'KV cache transfer faster by {speedup:.2f}x than recomputation'
                else:
                    recommendation = 'KV cache transfer faster than recomputation (near-zero transfer overhead)'
            
            # Additional context
            # FIXED: Corrected logical error by using the newly calculated key
            if trends['transfer_overhead_pct'] > 50:
                recommendation += ' (High transfer overhead detected)'
            elif trends['cache_efficiency_pct'] > 80:
                recommendation += ' (Excellent cache efficiency)'
            
            analysis['efficiency_recommendations'][token_length] = {
                'best_strategy': best_strategy,
                'best_time': best_time,
                'recommendation': recommendation,
                'pure_recomputation_time': pure_recomputation,
                'kv_transfer_overhead': kv_transfer_overhead,
                'speedup_ratio': (max(pure_recomputation, kv_transfer_overhead) / min(pure_recomputation, kv_transfer_overhead)) 
                                if min(pure_recomputation, kv_transfer_overhead) > 0 else float('inf')
            }
        
        # GH200-specific insights
        analysis['gh200_insights'] = {
            'unified_memory_benefit': 'High' if any(
                trends['transfer_overhead_pct'] < 20 for trends in analysis['performance_trends'].values()
            ) else 'Moderate',
            'optimal_token_ranges': {},
            'memory_bandwidth_bottleneck': any(
                trends['memory_bandwidth_critical'] for trends in analysis['performance_trends'].values()
            )
        }
        
        # Categorize optimal strategies by token range
        recompute_optimal = [t for t, trends in analysis['performance_trends'].items() 
                           if trends['is_recompute_faster_than_transfer']]
        transfer_optimal = [t for t, trends in analysis['performance_trends'].items() 
                          if not trends['is_recompute_faster_than_transfer']]
        
        if recompute_optimal:
            analysis['gh200_insights']['optimal_token_ranges']['pure_recomputation'] = {
                'token_range': f"{min(recompute_optimal)}-{max(recompute_optimal)}",
                'reason': 'Pure recomputation faster than KV cache transfer overhead'
            }
        
        if transfer_optimal:
            analysis['gh200_insights']['optimal_token_ranges']['kv_cache_transfer'] = {
                'token_range': f"{min(transfer_optimal)}-{max(transfer_optimal)}" if transfer_optimal else "None",
                'reason': 'KV cache transfer overhead less than recomputation cost'
            }
        
        return analysis

    def benchmark_recomputation_scenario(self, base_prompt: str, 
                                       variations: List[str],
                                       max_tokens: int = 50) -> Tuple[float, float]:
        """Benchmark recomputation scenarios with shared prefixes"""
        if self.llm is None:
            raise ValueError("Model not loaded. Call measure_model_loading() first.")
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True
        )
        
        # First, run the base prompt
        print("Running base prompt...")
        start_time = time.perf_counter()
        base_output = self.llm.generate([base_prompt], sampling_params)
        base_time = time.perf_counter() - start_time
        
        # Then run variations (should benefit from any caching/recomputation optimizations)
        print(f"Running {len(variations)} variations...")
        start_time = time.perf_counter()
        variation_outputs = self.llm.generate(variations, sampling_params)
        variation_time = time.perf_counter() - start_time
        
        # Estimate cache hit rate (simplified)
        avg_variation_time = variation_time / len(variations)
        cache_hit_rate = max(0, (base_time - avg_variation_time) / base_time)
        
        print(f"Base prompt time: {base_time:.4f} seconds")
        print(f"Average variation time: {avg_variation_time:.4f} seconds")
        print(f"Estimated cache benefit: {cache_hit_rate:.2%}")
        
        return cache_hit_rate, variation_time
    
    def run_comprehensive_benchmark(self, test_configs: List[Dict]) -> List[BenchmarkResult]:
        """Run comprehensive benchmark with multiple configurations"""
        system_info = self.check_gh200_environment()
        print("System Information:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        print()
        
        for config in test_configs:
            print(f"\n{'='*60}")
            print(f"Running test: {config['name']}")
            print(f"Config: {config}")
            print(f"{'='*60}")
            
            # Generate test prompts
            prompts = self.generate_test_prompts(
                config['seq_length'], 
                config['batch_size']
            )
            
            # Measure loading (for non-intersection tests)
            if not config.get('is_intersection_test', False):
                loading_time, gpu_allocated, gpu_reserved = self.measure_model_loading()
                cpu_memory = self.measure_cpu_memory_usage()
                
                # Measure inference
                first_token_lat, throughput, total_time = self.benchmark_inference_latency(
                    prompts, config.get('max_tokens', 100)
                )
                
                # Legacy recomputation test
                cache_hit_rate = None
                recomputation_time = None
                if config.get('test_recomputation', False):
                    base_prompt = prompts[0]
                    variations = [base_prompt + f" Additional context {i}" for i in range(5)]
                    cache_hit_rate, recomputation_time = self.benchmark_recomputation_scenario(
                        base_prompt, variations
                    )
                
                # Store results
                result = BenchmarkResult(
                    test_name=config['name'],
                    model_name=self.model_name,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    seq_length=config['seq_length'],
                    batch_size=config['batch_size'],
                    model_loading_time=loading_time,
                    gpu_memory_allocated=gpu_allocated,
                    gpu_memory_reserved=gpu_reserved,
                    first_token_latency=first_token_lat,
                    throughput_tokens_per_sec=throughput,
                    total_inference_time=total_time,
                    cache_hit_rate=cache_hit_rate,
                    recomputation_time=recomputation_time,
                    cpu_memory_usage=cpu_memory
                )
                
                self.results.append(result)
            
            # Clean up for next test
            if self.llm is not None:
                del self.llm
                self.llm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self.results

    def run_intersection_analysis(self, token_lengths: List[int] = None, 
                                 max_tokens: int = 50, num_runs: int = 10) -> Dict:
        """Run intersection analysis for KV cache loading vs recomputation across token lengths"""
        if token_lengths is None:
            token_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        
        print(f"\n{'='*80}")
        print("KV CACHE INTERSECTION ANALYSIS: Loading from CPU DRAM vs Recomputation")
        print(f"Testing token lengths: {token_lengths}")
        print(f"Measurements repeated {num_runs} times for stability")
        print(f"{'='*80}")
        
        # Run the KV cache benchmark across different token lengths
        kv_cache_results = self.benchmark_kv_cache_vs_recomputation(
            token_lengths, max_tokens, num_runs
        )
        
        # Analyze intersection points
        analysis = self.analyze_kv_cache_intersection(kv_cache_results)
        
        # Store results for each token length
        for token_length, results in kv_cache_results.items():
            result = BenchmarkResult(
                test_name=f'kv_cache_analysis_{token_length}_tokens',
                model_name=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                seq_length=token_length,
                batch_size=1,
                model_loading_time=0.0,  # Model stays loaded
                gpu_memory_allocated=0.0,  # Not measured in this specific test
                gpu_memory_reserved=0.0,
                first_token_latency=results['warm_recomputation'],
                throughput_tokens_per_sec=0.0,  # Not applicable
                total_inference_time=results['warm_recomputation'],
                load_and_first_inference_time=results['cold_recomputation'],
                warm_recompute_time=results['warm_recomputation'],
                # FIXED: Corrected key from 'kv_cache_reload_from_cpu' to 'kv_cache_transfer_time'
                cached_computation_time=results['kv_cache_transfer_time'],
                variation_recomputation_time=results['partial_recomputation'],
                loading_overhead=results['kv_cache_transfer_overhead'],
                cpu_memory_usage=self.measure_cpu_memory_usage()
            )
            self.results.append(result)
        
        return {
            'kv_cache_results': kv_cache_results,
            'analysis': analysis
        }
    
    def generate_test_prompts(self, seq_length: int, batch_size: int) -> List[str]:
        """Generate test prompts of specified length with different content and random numbers"""
        import random
        
        base_texts = [
            "The quick brown fox jumps over the lazy dog. ",
            "In a hole in the ground there lived a hobbit. ",
            "It was the best of times, it was the worst of times. ",
            "To be or not to be, that is the question. ",
            "Call me Ishmael. Some years ago never mind how long precisely. ",
        ]
        
        # Create prompts of approximately the target sequence length
        target_chars = seq_length * 4  # Rough estimate: 4 chars per token
        
        # Create batch of unique prompts with different base texts and random numbers
        prompts = []
        for i in range(batch_size):
            # Generate random numbers to make each prompt unique
            random_seed = random.randint(100000, 999999)
            random_numbers = [random.randint(1, 1000) for _ in range(5)]
            random_suffix = f" Random seed: {random_seed}. Numbers: {', '.join(map(str, random_numbers))}. "
            
            base_text = base_texts[i % len(base_texts)] * 10  # Multiply for longer text
            repeats = max(1, target_chars // len(base_text))
            prompt_text = base_text * repeats
            
            # Add random suffix to ensure uniqueness
            prompt = f"Context {i}{random_suffix}{prompt_text[:target_chars-len(random_suffix)]}"
            prompts.append(prompt)
        
        return prompts
    
    def save_results(self, output_file: str = "gh200_benchmark_results.json", 
                    intersection_analysis: Dict = None):
        """Save benchmark results to JSON file"""
        results_dict = [asdict(result) for result in self.results]
        
        output_data = {
            'system_info': self.check_gh200_environment(),
            'benchmark_config': {
                'model_name': self.model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'gpu_memory_utilization': self.gpu_memory_utilization,
            },
            'results': results_dict,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if intersection_analysis:
            output_data['intersection_analysis'] = intersection_analysis
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")

    def print_intersection_summary(self, intersection_analysis: Dict):
        """Print detailed summary of KV cache intersection analysis"""
        print(f"\n{'='*80}")
        print("KV CACHE INTERSECTION ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        kv_cache_results = intersection_analysis['kv_cache_results']
        analysis = intersection_analysis['analysis']
        
        print("\nPerformance by Token Length (with measurement stability):")
        print("-" * 105)
        print(f"{'Tokens':<8} {'Cold':<10} {'Warm±std':<15} {'Pure Recomp':<12} {'KV Transfer±std':<18} {'Decision':<15}")
        print("-" * 105)
        
        for token_length in sorted(kv_cache_results.keys()):
            results = kv_cache_results[token_length]
            trends = analysis['performance_trends'][token_length]
            
            warm_str = f"{results['warm_recomputation']:.4f}±{results.get('warm_recomputation_std', 0):.4f}"
            # FIXED: Corrected key from 'kv_cache_reload_std' to 'kv_cache_transfer_std'
            kv_transfer_str = f"{results['kv_cache_transfer_overhead']:.4f}±{results.get('kv_cache_transfer_std', 0):.4f}"
            
            print(f"{token_length:<8} "
                  f"{results['cold_recomputation']:<10.4f} "
                  f"{warm_str:<15} "
                  f"{results['pure_recomputation_time']:<12.4f} "
                  f"{kv_transfer_str:<18} "
                  f"{trends['decision']:<15}")
        
        # Show measurement stability summary
        print(f"\nMeasurement Stability (Coefficient of Variation %):")
        print("-" * 60)
        print(f"{'Tokens':<8} {'Warm CV%':<10} {'KV Transfer CV%':<15} {'Partial CV%':<12} {'Quality':<10}")
        print("-" * 60)
        
        for token_length in sorted(kv_cache_results.keys()):
            results = kv_cache_results[token_length]
            stability = results.get('measurement_stability', {})
            
            warm_cv = stability.get('warm_cv', 0)
            # FIXED: Corrected key from 'kv_reload_cv' to 'kv_transfer_cv'
            kv_cv = stability.get('kv_transfer_cv', 0)
            partial_cv = stability.get('partial_cv', 0)
            
            # Determine quality based on CV
            max_cv = max(warm_cv, kv_cv, partial_cv)
            if max_cv < 5:
                quality = "Excellent"
            elif max_cv < 10:
                quality = "Good"
            elif max_cv < 20:
                quality = "Fair"
            else:
                quality = "Poor"
            
            print(f"{token_length:<8} "
                  f"{warm_cv:<10.1f} "
                  f"{kv_cv:<15.1f} "
                  f"{partial_cv:<12.1f} "
                  f"{quality:<10}")
        
        print(f"\nNote: CV (Coefficient of Variation) < 5% = Excellent, < 10% = Good, < 20% = Fair, >= 20% = Poor")
        
        print("\nIntersection Points (Pure Recomputation vs KV Cache Transfer):")
        print("-" * 65)
        if analysis['intersection_points']:
            for key, intersection in analysis['intersection_points'].items():
                print(f"  At {intersection['token_length']} tokens: {intersection['transition']}")
                print(f"    {intersection['description']}")
        else:
            print("  No clear intersection points found in tested range")
        
        print("\nGH200-Specific Insights:")
        print("-" * 25)
        gh200_insights = analysis['gh200_insights']
        print(f"  • Unified Memory Benefit: {gh200_insights['unified_memory_benefit']}")
        print(f"  • Memory Bandwidth Bottleneck: {'Yes' if gh200_insights['memory_bandwidth_bottleneck'] else 'No'}")
        
        if gh200_insights['optimal_token_ranges']:
            print("  • Optimal Token Ranges:")
            for strategy, info in gh200_insights['optimal_token_ranges'].items():
                print(f"    - {strategy}: {info['token_range']} tokens ({info['reason']})")
        
        print("\nKey Findings:")
        print("-" * 15)
        
        # Find where pure recomputation is faster than KV cache transfer
        recompute_wins = []
        transfer_wins = []
        
        for token_length in sorted(kv_cache_results.keys()):
            trends = analysis['performance_trends'][token_length]
            if trends['is_recompute_faster_than_transfer']:
                recompute_wins.append(token_length)
            else:
                transfer_wins.append(token_length)
        
        if recompute_wins:
            print(f"  • Pure recomputation faster than KV cache transfer for tokens: {recompute_wins}")
            # Ensure we don't divide by zero if ratio is inf
            valid_ratios = [1 / analysis['performance_trends'][t]['reload_vs_recompute_ratio'] 
                            for t in recompute_wins if analysis['performance_trends'][t]['reload_vs_recompute_ratio'] != float('inf')]
            if valid_ratios:
                avg_speedup = np.mean(valid_ratios)
                print(f"  • Average speedup when recomputation wins: {avg_speedup:.2f}x")
        
        if transfer_wins:
            print(f"  • KV cache transfer faster than pure recomputation for tokens: {transfer_wins}")
            if transfer_wins:
                avg_speedup = np.mean([
                    analysis['performance_trends'][t]['reload_vs_recompute_ratio'] for t in transfer_wins
                ])
                print(f"  • Average speedup when transfer wins: {avg_speedup:.2f}x")
        
        # Find best efficiency points
        best_recomputation = min(
            [(t, results['pure_recomputation_time']) for t, results in kv_cache_results.items()],
            key=lambda x: x[1]
        )
        best_transfer = min(
            [(t, results['kv_cache_transfer_overhead']) for t, results in kv_cache_results.items()],
            key=lambda x: x[1]
        )
        
        print(f"  • Fastest recomputation: {best_recomputation[1]:.4f}s at {best_recomputation[0]} tokens")
        print(f"  • Fastest KV transfer: {best_transfer[1]:.4f}s at {best_transfer[0]} tokens")
        
        # Memory bandwidth insights
        print(f"\nMemory Bandwidth Analysis:")
        print("-" * 30)
        for token_length in sorted(kv_cache_results.keys()):
            trends = analysis['performance_trends'][token_length]
            bandwidth_gbps = trends.get('transfer_bandwidth_gbps', 0)
            kv_size_mb = kv_cache_results[token_length].get('kv_cache_size_mb', 0)
            transfer_time = trends.get('kv_transfer_time', 0)
            
            if transfer_time > 0:
                print(f"  • {token_length:<5} tokens: ~{kv_size_mb:<5.1f}MB KV cache, "
                      f"transfer time {transfer_time:.4f}s, ~{bandwidth_gbps:.1f} GB/s")
        
        print(f"\nRecommendations for GH200:")
        print("-" * 30)
        print("  • Leverage unified memory for efficient KV cache management")
        print("  • Consider token length when deciding between reload vs recomputation")
        print("  • Monitor memory bandwidth utilization for optimal performance")
        if gh200_insights['memory_bandwidth_bottleneck']:
            print("  • Memory bandwidth appears to be a bottleneck - consider compression or recomputation")
        print("  • Use partial recomputation for sequences with shared prefixes")

    def print_summary(self):
        """Print a summary of benchmark results"""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        # Separate KV cache analysis results from regular benchmark results
        kv_cache_results = [r for r in self.results if 'kv_cache_analysis' in r.test_name]
        regular_results = [r for r in self.results if 'kv_cache_analysis' not in r.test_name and 'intersection_analysis' not in r.test_name]
        
        if regular_results:
            print("\nRegular Benchmark Results:")
            print("-" * 30)
            for result in regular_results:
                print(f"\nTest: {result.test_name}")
                print(f"  Model Loading Time: {result.model_loading_time:.2f}s")
                print(f"  GPU Memory Allocated: {result.gpu_memory_allocated:.2f} GB")
                print(f"  First Token Latency: {result.first_token_latency:.4f}s")
                print(f"  Throughput: {result.throughput_tokens_per_sec:.2f} tokens/s")
                if result.cache_hit_rate is not None:
                    print(f"  Cache Hit Rate: {result.cache_hit_rate:.2%}")
                if result.recomputation_time is not None:
                    print(f"  Recomputation Time: {result.recomputation_time:.4f}s")
        
        if kv_cache_results:
            print(f"\nKV Cache Analysis Results ({len(kv_cache_results)} token lengths tested):")
            print("-" * 75)
            print(f"{'Tokens':<8} {'Cold Recomp (s)':<18} {'KV Transfer (s)':<18} {'Partial (s)':<15} {'Transfer OH (s)':<15}")
            print("-" * 75)
            
            for result in sorted(kv_cache_results, key=lambda x: x.seq_length):
                if (result.load_and_first_inference_time is not None and result.cached_computation_time is not None and 
                    result.variation_recomputation_time is not None and result.loading_overhead is not None):
                    print(f"{result.seq_length:<8} "
                          f"{result.load_and_first_inference_time:<18.4f} "
                          f"{result.cached_computation_time:<18.4f} "
                          f"{result.variation_recomputation_time:<15.4f} "
                          f"{result.loading_overhead:<15.4f}")
            
            # Calculate and show intersection point for KV cache vs recomputation
            intersection_found = False
            for result in sorted(kv_cache_results, key=lambda x: x.seq_length):
                if (result.cached_computation_time is not None and result.load_and_first_inference_time is not None and
                    result.cached_computation_time < result.load_and_first_inference_time):
                    print(f"\n  → KV Cache Intersection: Around {result.seq_length} tokens")
                    print(f"    (KV cache transfer from CPU becomes faster than cold recomputation)")
                    if result.cached_computation_time > 0:
                        speedup = result.load_and_first_inference_time / result.cached_computation_time
                        print(f"    Speedup: {speedup:.2f}x")
                    intersection_found = True
                    break
            
            if not intersection_found:
                print(f"\n  → No intersection found in tested range")
                print(f"    (Cold recomputation remains faster than KV cache transfer from CPU)")
            
            # Analyze transfer overhead impact
            high_overhead_tokens = []
            for result in kv_cache_results:
                if (result.loading_overhead is not None and result.cached_computation_time is not None and 
                    result.cached_computation_time > 0 and
                    result.loading_overhead / result.cached_computation_time > 0.3):
                    high_overhead_tokens.append(result.seq_length)
            
            if high_overhead_tokens:
                print(f"\n  → High transfer overhead (>30%) at: {high_overhead_tokens} tokens")
                print(f"    Consider recomputation for these token lengths")


def main():
    parser = argparse.ArgumentParser(description="GH200 KV Cache Loading vs Recomputation Latency Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name to benchmark")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization fraction")
    parser.add_argument("--output-file", type=str, default="gh200_kv_cache_benchmark_results.json",
                        help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run a quick test with smaller configurations")
    parser.add_argument("--intersection-only", type=bool, default=True,
                        help="Run only KV cache intersection analysis (CPU DRAM loading vs recomputation)")
    parser.add_argument("--token-lengths", type=int, nargs='+', 
                        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
                        help="Token lengths to test for intersection analysis")
    parser.add_argument("--max-tokens", type=int, default=1,
                        help="Maximum tokens to generate in each test")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of runs for repeated measurements (default: 100)")

    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = GH200Benchmark(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device=args.device
    )
    
    print("Starting GH200 KV Cache Loading vs Recomputation Benchmark...")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    
    # FIXED: Corrected syntax error
    print("Focus: KV cache loading from CPU DRAM vs recomputation analysis (simulated transfer)")
    intersection_analysis = None
    
    try:
        if args.intersection_only:
            # Run only KV cache intersection analysis
            print(f"Running KV cache intersection analysis for token lengths: {args.token_lengths}")
            print(f"Each measurement repeated {args.num_runs} times for stability")
            intersection_analysis = benchmark.run_intersection_analysis(
                args.token_lengths, args.max_tokens, args.num_runs
            )
            benchmark.print_intersection_summary(intersection_analysis)
            
        else:
            # Define test configurations for regular benchmarks
            if args.quick_test:
                test_configs = [
                    {
                        'name': 'quick_short_sequence',
                        'seq_length': 128,
                        'batch_size': 1,
                        'max_tokens': 1,
                        'test_recomputation': False
                    },
                    {
                        'name': 'quick_with_recomputation',
                        'seq_length': 256,
                        'batch_size': 2,
                        'max_tokens': 1,
                        'test_recomputation': True
                    }
                ]
            else:
                test_configs = [
                    {
                        'name': 'short_sequence_single',
                        'seq_length': 128,
                        'batch_size': 1,
                        'max_tokens': 1,
                        'test_recomputation': False
                    },
                ]
            
            # Run regular benchmark
            results = benchmark.run_comprehensive_benchmark(test_configs)
            
            # Also run KV cache intersection analysis unless it's a quick test
            if not args.quick_test:
                print(f"\nRunning additional KV cache intersection analysis...")
                intersection_analysis = benchmark.run_intersection_analysis(
                    args.token_lengths, args.max_tokens, args.num_runs
                )
                benchmark.print_intersection_summary(intersection_analysis)
        
        # Print summary
        benchmark.print_summary()
        
        # Save results
        benchmark.save_results(args.output_file, intersection_analysis)
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())