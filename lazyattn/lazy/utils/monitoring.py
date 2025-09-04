#!/usr/bin/env python3
"""
LazyAttention 监控工具

提供实时监控和报告功能，帮助分析 LazyScheduler 的性能。
"""

import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


class LazySchedulerMonitor:
    """LazyScheduler 监控器"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logger = logging.getLogger('lazy_monitor')
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
    def parse_log_file(self) -> Dict[str, Any]:
        """解析日志文件，提取监控数据"""
        if not Path(self.log_file).exists():
            self.logger.error(f"Log file not found: {self.log_file}")
            return {}
            
        monitoring_data = {
            'timestamps': [],
            'memory_usage': [],
            'throughput': [],
            'cache_hit_ratio': [],
            'running_requests': [],
            'waiting_requests': [],
            'schedule_steps': [],
            'total_tokens_processed': []
        }
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if 'Monitoring stats:' in line:
                        # 提取 JSON 数据
                        json_start = line.find('{')
                        if json_start != -1:
                            json_str = line[json_start:]
                            try:
                                data = json.loads(json_str)
                                
                                monitoring_data['timestamps'].append(data.get('timestamp', 0))
                                monitoring_data['memory_usage'].append(data.get('current_memory_usage', 0))
                                monitoring_data['throughput'].append(data.get('throughput_tokens_per_sec', 0))
                                monitoring_data['cache_hit_ratio'].append(data.get('cache_hit_ratio', 0))
                                monitoring_data['running_requests'].append(data.get('running_requests', 0))
                                monitoring_data['waiting_requests'].append(data.get('waiting_requests', 0))
                                monitoring_data['schedule_steps'].append(data.get('schedule_steps', 0))
                                monitoring_data['total_tokens_processed'].append(data.get('total_tokens_processed', 0))
                                
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            self.logger.error(f"Error parsing log file: {e}")
            
        return monitoring_data
    
    def generate_plots(self, data: Dict[str, Any], output_dir: str = "monitoring_plots"):
        """生成监控图表"""
        if not data or not data['timestamps']:
            self.logger.warning("No data to plot")
            return
            
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 转换时间戳为相对时间（秒）
        start_time = min(data['timestamps'])
        relative_times = [(t - start_time) for t in data['timestamps']]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LazyScheduler 监控数据', fontsize=16)
        
        # 1. 内存使用率
        axes[0, 0].plot(relative_times, data['memory_usage'], 'b-', linewidth=2)
        axes[0, 0].set_title('内存使用率')
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('使用率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 吞吐量
        axes[0, 1].plot(relative_times, data['throughput'], 'g-', linewidth=2)
        axes[0, 1].set_title('吞吐量 (tokens/sec)')
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel('tokens/sec')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 缓存命中率
        axes[1, 0].plot(relative_times, data['cache_hit_ratio'], 'r-', linewidth=2)
        axes[1, 0].set_title('缓存命中率')
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel('命中率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 请求数量
        axes[1, 1].plot(relative_times, data['running_requests'], 'b-', label='运行中', linewidth=2)
        axes[1, 1].plot(relative_times, data['waiting_requests'], 'r-', label='等待中', linewidth=2)
        axes[1, 1].set_title('请求数量')
        axes[1, 1].set_xlabel('时间 (秒)')
        axes[1, 1].set_ylabel('请求数量')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f"{output_dir}/lazy_scheduler_monitoring_{int(time.time())}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_file}")
        
    def generate_summary_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要报告"""
        if not data or not data['timestamps']:
            return {}
            
        # 计算统计信息
        memory_usage = data['memory_usage']
        throughput = data['throughput']
        cache_hit_ratio = data['cache_hit_ratio']
        
        summary = {
            'runtime_seconds': max(data['timestamps']) - min(data['timestamps']),
            'total_schedule_steps': max(data['schedule_steps']) if data['schedule_steps'] else 0,
            'total_tokens_processed': max(data['total_tokens_processed']) if data['total_tokens_processed'] else 0,
            'memory_usage': {
                'average': np.mean(memory_usage) if memory_usage else 0,
                'max': np.max(memory_usage) if memory_usage else 0,
                'min': np.min(memory_usage) if memory_usage else 0,
                'std': np.std(memory_usage) if memory_usage else 0,
            },
            'throughput': {
                'average': np.mean(throughput) if throughput else 0,
                'max': np.max(throughput) if throughput else 0,
                'min': np.min(throughput) if throughput else 0,
                'std': np.std(throughput) if throughput else 0,
            },
            'cache_hit_ratio': {
                'average': np.mean(cache_hit_ratio) if cache_hit_ratio else 0,
                'max': np.max(cache_hit_ratio) if cache_hit_ratio else 0,
                'min': np.min(cache_hit_ratio) if cache_hit_ratio else 0,
            },
            'requests': {
                'max_running': np.max(data['running_requests']) if data['running_requests'] else 0,
                'max_waiting': np.max(data['waiting_requests']) if data['waiting_requests'] else 0,
                'avg_running': np.mean(data['running_requests']) if data['running_requests'] else 0,
                'avg_waiting': np.mean(data['waiting_requests']) if data['waiting_requests'] else 0,
            }
        }
        
        # 计算效率指标
        if summary['total_tokens_processed'] > 0 and summary['memory_usage']['average'] > 0:
            summary['efficiency_metrics'] = {
                'tokens_per_second': summary['total_tokens_processed'] / summary['runtime_seconds'],
                'memory_efficiency': summary['total_tokens_processed'] / (summary['memory_usage']['average'] + 1e-6),
                'cache_efficiency': summary['cache_hit_ratio']['average'],
            }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """打印摘要信息"""
        if not summary:
            self.logger.warning("No summary data available")
            return
            
        print("\n" + "="*60)
        print("LAZY SCHEDULER 监控摘要")
        print("="*60)
        
        print(f"运行时间: {summary['runtime_seconds']:.2f} 秒")
        print(f"调度步数: {summary['total_schedule_steps']}")
        print(f"处理令牌数: {summary['total_tokens_processed']:,}")
        
        if 'efficiency_metrics' in summary:
            eff = summary['efficiency_metrics']
            print(f"平均吞吐量: {eff['tokens_per_second']:.2f} tokens/sec")
            print(f"内存效率: {eff['memory_efficiency']:.2f}")
            print(f"缓存效率: {eff['cache_efficiency']:.4f}")
        
        mem = summary['memory_usage']
        print(f"内存使用率 - 平均: {mem['average']:.4f}, 最大: {mem['max']:.4f}, 最小: {mem['min']:.4f}")
        
        thr = summary['throughput']
        print(f"吞吐量 - 平均: {thr['average']:.2f}, 最大: {thr['max']:.2f}, 最小: {thr['min']:.2f}")
        
        cache = summary['cache_hit_ratio']
        print(f"缓存命中率 - 平均: {cache['average']:.4f}, 最大: {cache['max']:.4f}")
        
        req = summary['requests']
        print(f"请求统计 - 运行中(最大/平均): {req['max_running']}/{req['avg_running']:.1f}")
        print(f"请求统计 - 等待中(最大/平均): {req['max_waiting']}/{req['avg_waiting']:.1f}")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LazyScheduler 监控工具')
    parser.add_argument('log_file', help='日志文件路径')
    parser.add_argument('--output-dir', default='monitoring_plots', help='图表输出目录')
    parser.add_argument('--no-plots', action='store_true', help='不生成图表')
    parser.add_argument('--summary-only', action='store_true', help='只显示摘要')
    
    args = parser.parse_args()
    
    monitor = LazySchedulerMonitor(args.log_file)
    
    # 解析日志文件
    data = monitor.parse_log_file()
    
    if not data or not data['timestamps']:
        print("未找到有效的监控数据")
        return
    
    # 生成摘要报告
    summary = monitor.generate_summary_report(data)
    monitor.print_summary(summary)
    
    # 生成图表
    if not args.no_plots:
        monitor.generate_plots(data, args.output_dir)
    
    # 保存摘要报告
    if summary:
        report_file = f"lazy_scheduler_summary_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"摘要报告已保存到: {report_file}")


if __name__ == "__main__":
    main() 