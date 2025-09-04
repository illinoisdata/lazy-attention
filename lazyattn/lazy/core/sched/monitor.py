import os
import time
import json
import logging
from typing import Dict, Any, Optional

class LazySchedulerMonitor:
    """
    独立的监控类，负责收集、统计、日志和报告 LazyScheduler 的运行数据。
    """
    def __init__(self, time_str: Optional[str] = None):
        self.enabled = os.environ.get("LAZY_CACHE_LOG") == "1"
        if not self.enabled:
            return
        self.time_str = time_str or time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.hit_sum = 0
        self.query_sum = 0
        self.stats = {
            'schedule_steps': 0,
            'total_requests_processed': 0,
            'total_tokens_processed': 0,
            'cache_hit_ratio': 0.0,
            'memory_usage_history': [],
            'request_latency': [],
            'throughput_history': [],
            'start_time': time.time(),
            'last_log_time': time.time(),
        }
        self.logger = logging.getLogger(f'lazy_scheduler_monitor_{self.time_str}')
        self.logger.setLevel(logging.INFO)
        log_file = f"lazy_scheduler_{self.time_str}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.info(f"LazySchedulerMonitor started at {self.time_str}")

    def update_stats(self, scheduler_output, kv_cache_manager, running, waiting):
        if not self.enabled:
            return
        current_time = time.time()
        self.stats['schedule_steps'] += 1
        # 内存使用率
        current_memory_usage = kv_cache_manager.usage
        self.stats['memory_usage_history'].append({
            'timestamp': current_time,
            'usage': current_memory_usage
        })
        # 处理的 token 数
        total_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        self.stats['total_tokens_processed'] += total_scheduled_tokens
        # 缓存命中率
        if hasattr(kv_cache_manager, 'prefix_cache_stats') and kv_cache_manager.prefix_cache_stats:
            stats = kv_cache_manager.prefix_cache_stats
            self.hit_sum += stats.hits
            self.query_sum += stats.queries
            if self.query_sum > 0:
                self.stats['cache_hit_ratio'] = self.hit_sum / self.query_sum
        # 吞吐量
        elapsed_time = current_time - self.stats['start_time']
        if elapsed_time > 0:
            throughput = self.stats['total_tokens_processed'] / elapsed_time
            self.stats['throughput_history'].append({
                'timestamp': current_time,
                'throughput': throughput
            })
        # 定期日志
        if (self.stats['schedule_steps'] % 100 == 0 or 
            current_time - self.stats['last_log_time'] > 10):
            self.log_stats(running, waiting, kv_cache_manager)
            self.stats['last_log_time'] = current_time

    def log_stats(self, running, waiting, kv_cache_manager):
        if not self.enabled:
            return
        stats = self.stats
        current_time = time.time()
        elapsed_time = current_time - stats['start_time']
        log_data = {
            'timestamp': current_time,
            'elapsed_time': elapsed_time,
            'schedule_steps': stats['schedule_steps'],
            'total_tokens_processed': stats['total_tokens_processed'],
            'current_memory_usage': kv_cache_manager.usage,
            'cache_hit_ratio': stats['cache_hit_ratio'],
            'running_requests': len(running),
            'waiting_requests': len(waiting),
            'throughput_tokens_per_sec': stats['total_tokens_processed'] / elapsed_time if elapsed_time > 0 else 0,
        }
        self.logger.info(f"Monitoring stats: {json.dumps(log_data, indent=2)}")

    def get_detailed_stats(self, running, waiting, kv_cache_manager) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        current_time = time.time()
        elapsed_time = current_time - self.stats['start_time']
        memory_history = self.stats['memory_usage_history']
        avg_memory_usage = sum(h['usage'] for h in memory_history) / len(memory_history) if memory_history else 0
        max_memory_usage = max(h['usage'] for h in memory_history) if memory_history else 0
        min_memory_usage = min(h['usage'] for h in memory_history) if memory_history else 0
        throughput_history = self.stats['throughput_history']
        avg_throughput = sum(h['throughput'] for h in throughput_history) / len(throughput_history) if throughput_history else 0
        max_throughput = max(h['throughput'] for h in throughput_history) if throughput_history else 0
        return {
            'summary': {
                'total_runtime_seconds': elapsed_time,
                'total_schedule_steps': self.stats['schedule_steps'],
                'total_tokens_processed': self.stats['total_tokens_processed'],
                'average_throughput_tokens_per_sec': avg_throughput,
                'max_throughput_tokens_per_sec': max_throughput,
                'average_memory_usage': avg_memory_usage,
                'max_memory_usage': max_memory_usage,
                'min_memory_usage': min_memory_usage,
                'cache_hit_ratio': self.stats['cache_hit_ratio'],
            },
            'current_state': {
                'running_requests': len(running),
                'waiting_requests': len(waiting),
                'current_memory_usage': kv_cache_manager.usage,
                'current_throughput': self.stats['total_tokens_processed'] / elapsed_time if elapsed_time > 0 else 0,
            }
        }

    def get_final_stats_report(self, running, waiting, kv_cache_manager) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        self.log_stats(running, waiting, kv_cache_manager)
        detailed_stats = self.get_detailed_stats(running, waiting, kv_cache_manager)
        final_report = {
            **detailed_stats,
            'performance_metrics': {
                'tokens_per_schedule_step': (
                    detailed_stats['summary']['total_tokens_processed'] /
                    detailed_stats['summary']['total_schedule_steps']
                    if detailed_stats['summary']['total_schedule_steps'] > 0 else 0
                ),
                'memory_efficiency': (
                    detailed_stats['summary']['total_tokens_processed'] /
                    (detailed_stats['summary']['average_memory_usage'] + 1e-6)
                ),
                'cache_efficiency': detailed_stats['summary']['cache_hit_ratio'],
            },
            'monitoring_config': {
                'monitoring_enabled': self.enabled,
                'start_time': self.stats['start_time'],
                'time_str': self.time_str,
            }
        }
        report_file = f"lazy_scheduler_final_report_{self.time_str}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        self.logger.info(f"Final monitoring report saved to {report_file}")
        return final_report

    def print_summary(self, running, waiting, kv_cache_manager) -> None:
        if not self.enabled:
            print("Monitoring is not enabled. Set LAZY_CACHE_LOG=1 to enable.")
            return
        stats = self.get_detailed_stats(running, waiting, kv_cache_manager)
        if not stats:
            return
        print("\n" + "="*60)
        print("LAZY SCHEDULER MONITORING SUMMARY")
        print("="*60)
        summary = stats['summary']
        current = stats['current_state']
        print(f"Runtime: {summary['total_runtime_seconds']:.2f} seconds")
        print(f"Schedule Steps: {summary['total_schedule_steps']}")
        print(f"Total Tokens Processed: {summary['total_tokens_processed']:,}")
        print(f"Average Throughput: {summary['average_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Max Throughput: {summary['max_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Cache Hit Ratio: {summary['cache_hit_ratio']:.4f}")
        print(f"Memory Usage - Avg: {summary['average_memory_usage']:.4f}, Max: {summary['max_memory_usage']:.4f}")
        print(f"Current State - Running: {current['running_requests']}, Waiting: {current['waiting_requests']}")
        print("="*60) 