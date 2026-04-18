# HunYuanImage Performance Dashboard

性能监控仪表板 - 用于追踪 HunYuanImage 扩散模型的性能变化

## 版权声明
MIT License | Copyright (c) 2026 思捷娅科技 (SJYKJ)

---

## 概述

本模块提供 HunYuanImage 扩散模型的性能监控和回归检测功能，包括：

- ✅ CI 基准测试自动化
- ✅ 性能回归检测
- ✅ 基线对比报告
- ✅ GitHub Actions 集成
- ✅ 性能趋势追踪

---

## 文件结构

```
performance_dashboard/
├── baseline.json                      # 性能基线
├── check_regression.py                # 回归检测脚本
├── compare_benchmarks.py              # 对比报告生成
└── README.md                          # 本文档

../
├── hunyuan_ci_benchmark.py            # CI 基准测试主程序
└── diffusion_benchmark_serving.py     # 服务基准测试
```

---

## 快速开始

### 1. 运行基准测试

```bash
python benchmarks/diffusion/hunyuan_ci_benchmark.py \
  --model ./HunyuanImage-3.0 \
  --tensor-parallel-size 4 \
  --num-steps 50 \
  --guidance-scale 5.0 \
  --seed 1234 \
  --output-dir benchmark_results
```

### 2. 与基线对比

```bash
python benchmarks/diffusion/performance_dashboard/compare_benchmarks.py \
  --current benchmark_results/benchmark_20260323_093000.json \
  --baseline benchmarks/diffusion/performance_dashboard/baseline.json \
  --output benchmark_results/comparison_report.md
```

### 3. 检查性能回归

```bash
python benchmarks/diffusion/performance_dashboard/check_regression.py \
  --current benchmark_results/benchmark_20260323_093000.json \
  --baseline benchmarks/diffusion/performance_dashboard/baseline.json \
  --threshold 0.10
```

---

## CI/CD 集成

### GitHub Actions

工作流程文件：`.github/workflows/hunyuan-ci-benchmark.yml`

**触发条件：**
- Push 到 main/master 分支（相关路径变更）
- Pull Request（相关路径变更）
- 每天 UTC 00:00 定时运行
- 手动触发（带参数）

**运行步骤：**
1. 检出代码
2. 安装依赖
3. 下载模型
4. 运行基准测试
5. 上传结果
6. 与基线对比
7. PR 评论（如果是 PR）
8. 回归检测（失败则阻断）

---

## 性能基线

### 当前基线（A100）

| 配置 | 值 |
|------|-----|
| Tensor Parallel | 4 |
| Expert Parallel | 4 |
| 推理步数 | 50 |
| 引导系数 | 5.0 |
| 平均延迟 | 27.05s |

### 更新基线

当有以下情况时，应更新基线：

1. **硬件升级** - 更换 GPU 型号
2. **重大优化** - 性能提升超过 20%
3. **架构变更** - 模型结构重大调整

更新方法：

```bash
# 1. 运行基准测试
python benchmarks/diffusion/hunyuan_ci_benchmark.py \
  --model ./HunyuanImage-3.0 \
  --tensor-parallel-size 4 \
  --expert-parallel-size 4 \
  --output-dir benchmark_results

# 2. 复制结果到基线
cp benchmark_results/benchmark_*.json \
   benchmarks/diffusion/performance_dashboard/baseline.json

# 3. 编辑 baseline.json，更新 notes 字段
```

---

## 回归检测

### 阈值设置

默认回归阈值：**10%**

| 变化范围 | 状态 | 处理 |
|---------|------|------|
| < -10% | 🟢 性能提升 | 记录优化成果 |
| -10% ~ +10% | 🟡 稳定 | 正常波动 |
| > +10% | 🔴 性能回退 | 需要调查 |
| > +20% | 🔴 严重回退 | 阻断 PR |

### 回归处理流程

1. **自动检测** - CI 自动运行回归检测
2. **标记 PR** - 在 PR 中添加性能回退评论
3. **调查原因** - 分析代码变更、环境因素
4. **修复或调整** - 修复性能问题或调整基线
5. **重新验证** - 重新运行基准测试确认

---

## 性能优化建议

根据性能分析，Attention 和 MoE 模块占总执行时间的 70-80%：

### Attention 优化（~30% runtime）

1. **Flash Attention 2**
   - 减少内存访问
   - 提升计算效率
   - 预期提升：15-25%

2. **Paged Attention**
   - 优化 KV Cache 管理
   - 减少内存碎片
   - 预期提升：10-20%

3. **Multi-Query Attention**
   - 减少 KV 头数量
   - 降低显存占用
   - 预期提升：5-15%

### MoE 优化（~70% runtime）

1. **Expert Parallel 优化**
   - 改进负载均衡
   - 减少通信开销
   - 预期提升：20-30%

2. **动态路由**
   - 智能选择 Expert
   - 减少不必要计算
   - 预期提升：10-15%

3. **量化**
   - INT8/FP8 量化
   - 减少显存带宽
   - 预期提升：30-50%

---

## 监控指标

### Prometheus 指标（待实现）

```
hunyuan_inference_latency_seconds{step="total"}
hunyuan_inference_latency_seconds{step="attention"}
hunyuan_inference_latency_seconds{step="moe"}
hunyuan_inference_memory_usage_bytes
hunyuan_inference_throughput_images_per_second
```

### Grafana 仪表板（待实现）

- 实时性能监控
- 历史趋势图
- 异常告警

---

## 测试

运行单元测试：

```bash
pytest benchmarks/diffusion/performance_dashboard/test_*.py -v
```

---

## 故障排查

### 常见问题

**Q: 基准测试结果波动大？**
A: 确保测试环境稳定，关闭其他占用 GPU 的进程，多次测试取平均值。

**Q: CI 测试超时？**
A: 增加 timeout-minutes 或减少测试样本数（num-prompts）。

**Q: 回归检测误报？**
A: 调整 threshold 参数，或检查基线是否过时。

---

## 贡献指南

1. Fork 仓库
2. 创建功能分支
3. 运行基准测试确认性能无回退
4. 提交 PR（会自动触发 CI 基准测试）

---

*Last updated: 2026-03-23*
*Version: 1.0.0*
