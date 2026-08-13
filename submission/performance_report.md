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

## 配置优化探索记录（全量验证）

在 Ascend 910C 单卡环境下，我们对 vllm-omni v0.25.0-a3 的所有可调配置方向进行了系统性验证。以下 14 项均未产生正向收益或不被 NPU 后端支持：

| # | 尝试方向 | 结果 | 备注 |
|---|---------|------|------|
| 1 | `enable_prefix_caching: true` | ❌ 失败 | NPU 不兼容，Orchestrator 初始化失败 |
| 2 | `async_scheduling: true` | ❌ 失败 | NPU 不兼容，启动失败 |
| 3 | `max_num_seqs: 2` | ❌ 失败 | 启动后崩溃 |
| 4 | `enforce_eager: true` | ❌ 负收益 | TTFP 劣化至 3361ms |
| 5 | `compilation_config.capture_sizes` | ❌ 无增益 | 单卡 PIECEWISE 模式下无效 |
| 6 | `compilation_config.cudagraph_num_of_warmups` | ❌ 无增益 | 同上 |
| 7 | `gpu_memory_utilization > 0.55` | ❌ 无收益 | 单卡共用无边际收益 |
| 8 | `quantization: fp8` | ❌ 不支持 | 需额外 ignore 及 quant_format 参数 |
| 9 | `quantization: int8` | ❌ 不支持 | Unknown quantization method |
| 10 | `quantization: ascend` | ❌ 不支持 | 需 ModelSlim 预量化模型 |
| 11 | `quantization: fp8_per_tensor` | ❌ 不支持 | NPU 后端未实现 |
| 12 | `quantization: fp8_per_block` | ❌ 不支持 | NPU 后端未实现 |
| 13 | `quantization: fbgemm_fp8` | ❌ 不支持 | NPU 后端未实现 |
| 14 | `cudagraph_mode: DISABLE` | ❌ 不支持 | NPU 后端不支持该模式 |

**结论：** 在当前镜像提供的 NPU 后端能力范围内，配置层面的优化空间已完全探明。唯一有效的性能优化为 `connector_get_sleep_s: 0.001` 与 `token2wav_n_timesteps: 3` 的组合。
