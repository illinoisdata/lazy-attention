# LazyAttention 监控指南

本文档介绍如何使用 LazyAttention 的监控系统来分析调度器性能。

## 概述

LazyAttention 提供了一个完善的监控系统，可以实时跟踪和分析调度器的性能指标，包括：

- 内存使用率
- 吞吐量 (tokens/sec)
- 缓存命中率
- 请求队列状态
- 调度效率

## 启用监控

### 1. 环境变量设置

要启用监控，设置环境变量：

```bash
export LAZY_CACHE_LOG=1
```

### 2. 运行程序

正常启动你的 LazyAttention 程序，监控系统会自动开始收集数据。

## 监控指标

### 实时指标

监控系统会定期记录以下指标：

- **内存使用率**: KV缓存的使用情况
- **吞吐量**: 每秒处理的令牌数
- **缓存命中率**: 前缀缓存的命中率
- **运行中请求数**: 当前正在处理的请求数量
- **等待中请求数**: 队列中等待的请求数量
- **调度步数**: 已执行的调度步骤数
- **总处理令牌数**: 累计处理的令牌数

### 日志输出

监控数据会输出到以下位置：

1. **控制台**: 实时显示监控信息
2. **日志文件**: `lazy_scheduler_YYYYMMDD_HHMMSS.log`
3. **最终报告**: `lazy_scheduler_final_report_YYYYMMDD_HHMMSS.json`

## 使用监控工具

### 1. 实时监控

在程序运行时，监控信息会定期输出到控制台：

```
2024-01-15 10:30:15 - lazy_scheduler - INFO - Monitoring stats: {
  "timestamp": 1705312215.123,
  "elapsed_time": 45.67,
  "schedule_steps": 150,
  "total_tokens_processed": 15000,
  "current_memory_usage": 0.75,
  "cache_hit_ratio": 0.85,
  "running_requests": 8,
  "waiting_requests": 3,
  "throughput_tokens_per_sec": 328.45
}
```

### 2. 最终统计报告

程序结束时，会生成详细的统计报告：

```python
# 在程序结束时调用
scheduler.print_summary_stats()
```

输出示例：
```
============================================================
LAZY SCHEDULER MONITORING SUMMARY
============================================================
Runtime: 120.45 seconds
Schedule Steps: 500
Total Tokens Processed: 50,000
Average Throughput: 415.23 tokens/sec
Max Throughput: 520.15 tokens/sec
Cache Hit Ratio: 0.8745
Memory Usage - Avg: 0.7234, Max: 0.8912
Current State - Running: 5, Waiting: 2
============================================================
```

### 3. 使用监控工具脚本

使用提供的监控工具来分析日志文件：

```bash
# 基本用法
python lazyattn/lazy/utils/monitoring.py lazy_scheduler_20240115_103000.log

# 只显示摘要
python lazyattn/lazy/utils/monitoring.py lazy_scheduler_20240115_103000.log --summary-only

# 不生成图表
python lazyattn/lazy/utils/monitoring.py lazy_scheduler_20240115_103000.log --no-plots

# 指定图表输出目录
python lazyattn/lazy/utils/monitoring.py lazy_scheduler_20240115_103000.log --output-dir my_plots
```

## 性能分析

### 关键指标解释

1. **吞吐量 (Throughput)**
   - 单位：tokens/sec
   - 表示系统处理令牌的速度
   - 越高越好，但需要考虑内存使用

2. **内存使用率 (Memory Usage)**
   - 范围：0.0 - 1.0
   - 表示KV缓存的使用情况
   - 过高可能导致OOM，过低可能浪费资源

3. **缓存命中率 (Cache Hit Ratio)**
   - 范围：0.0 - 1.0
   - 表示前缀缓存的有效性
   - 越高越好，可以减少重复计算

4. **请求队列状态**
   - 运行中请求：正在处理的请求数
   - 等待中请求：队列中的请求数
   - 平衡这两个指标以获得最佳性能

### 性能优化建议

1. **高内存使用率 + 低吞吐量**
   - 可能原因：内存碎片化、缓存效率低
   - 建议：调整缓存策略、优化内存分配

2. **低缓存命中率**
   - 可能原因：请求模式变化、缓存大小不足
   - 建议：增加缓存大小、优化缓存算法

3. **高等待请求数**
   - 可能原因：处理能力不足、请求积压
   - 建议：增加并发处理能力、优化调度算法

4. **吞吐量波动大**
   - 可能原因：请求大小不均、调度不均匀
   - 建议：实现更智能的调度策略

## 图表分析

监控工具会生成包含4个子图的综合图表：

1. **内存使用率趋势**: 显示内存使用随时间的变化
2. **吞吐量趋势**: 显示处理速度随时间的变化
3. **缓存命中率趋势**: 显示缓存效率随时间的变化
4. **请求数量趋势**: 显示运行中和等待中请求数量的变化

### 图表解读

- **平稳的曲线**: 表示系统运行稳定
- **剧烈波动**: 可能表示系统负载不均或配置问题
- **上升趋势**: 表示性能在改善
- **下降趋势**: 表示性能在恶化

## 故障排除

### 常见问题

1. **监控未启用**
   ```
   解决方案: 确保设置了 LAZY_CACHE_LOG=1
   ```

2. **日志文件过大**
   ```
   解决方案: 定期清理旧日志文件，或调整日志级别
   ```

3. **图表生成失败**
   ```
   解决方案: 安装 matplotlib 和 numpy: pip install matplotlib numpy
   ```

4. **性能数据异常**
   ```
   解决方案: 检查系统资源使用情况，确认监控数据准确性
   ```

## 高级用法

### 自定义监控指标

你可以扩展监控系统来跟踪额外的指标：

```python
# 在 _update_monitoring_stats 方法中添加自定义指标
def _update_monitoring_stats(self, scheduler_output: SchedulerOutput) -> None:
    # ... 现有代码 ...
    
    # 添加自定义指标
    custom_metric = self.calculate_custom_metric()
    self.monitoring_stats['custom_metric'] = custom_metric
```

### 实时监控集成

可以将监控数据集成到外部监控系统：

```python
# 发送数据到 Prometheus
def send_to_prometheus(self, stats):
    # 实现 Prometheus 推送逻辑
    pass

# 发送数据到 Grafana
def send_to_grafana(self, stats):
    # 实现 Grafana 推送逻辑
    pass
```

## 总结

LazyAttention 的监控系统提供了全面的性能分析工具，帮助你：

1. **实时监控**系统性能
2. **识别瓶颈**和性能问题
3. **优化配置**参数
4. **生成报告**用于分析和展示

通过合理使用这些监控工具，你可以显著提升 LazyAttention 的性能和稳定性。 