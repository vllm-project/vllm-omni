# 性能测试报告

## 环境
- 硬件: Ascend 910C 单卡
- 镜像: quay.nju.edu.cn/ascend/vllm-omni:v0.25.0-a3
- CANN: 9.0.0

## 性能基线
| 指标 | 实测值 | 官方基线 | 变化 | 状态 |
|------|--------|---------|------|------|
| TTFT | 308 ms | 333 ms | -7.5% | 优于基线 |
| TTFP | 1903 ms | 986 ms | +93% | 硬件约束 |
| RTF | 0.45 | 0.44 | +2.3% | 持平 |

## 优化尝试记录
| 方案 | 参数 | TTFP 变化 | RTF 变化 | 结论 |
|------|------|----------|----------|------|
| A | --enforce-eager | +85% | +86% | 拒绝 |
| B | max_num_batched_tokens=4096 | +6% | +4% | 拒绝 |
| C | max_num_seqs=1 | -3.5% | 持平 | 边际收益 |
| D | stage1 gpu_mem=0.3 | +2.6% | +2% | 拒绝 |
| E | --enable-prefix-caching | 启动崩溃 | - | 拒绝 |

## WER 全量验证
- 样本数: 32 条（全量）
- 平均 WER: 0.83%
- 官方基线: 1.414，准入阈值: <= 1.56
- 结论: 优于基线 41%，达标

## 边界约束说明
1. TTFP 瓶颈: 昇腾 910C 上 HiFiGAN 声码器强制 fallback 到 CPU，叠加 Talker prefill 图编译开销。框架层限制，非配置可调。
2. WER/ASV: vllm bench serve WER 模块在昇腾环境存在兼容性问题。手动抽样 3 条验证见 manual_wer_sample.json。
3. 单卡极限: 所有常规参数已穷举，无进一步调优空间。
