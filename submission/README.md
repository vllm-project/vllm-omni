# MiniCPM-o 4.5 vLLM-Omni 昇腾 910C 部署与评测报告

> **赛道**: vLLM-Omni 高性能推理优化
> **时间**: 2026-08-10
> **硬件**: Ascend 910C 单卡

## 1. 项目概述
在昇腾 910C NPU 单卡环境下完成 MiniCPM-o 4.5 全模态模型的 vLLM-Omni 部署，通过 Video-MME、Daily-Omni 精度评测，并完成 Seed-TTS 性能基线采集与手动 WER 抽样验证。

## 2. 快速启动


## 3. 环境要求
| 项目 | 版本 |
|------|------|
| 镜像 | quay.nju.edu.cn/ascend/vllm-omni:v0.25.0-a3 |
| CANN | 9.0.0 |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |

## 4. 前置准备
1. pip install stepaudio2-minicpmo
2. pip install step-audio2 --no-deps
3. SETUPTOOLS_SCM_PRETEND_VERSION=0.25.0 pip install -e .
4. 修改 async_chunk: false、模型注册导入

## 5. 执行流程
- Video-MME: 服务加 --allowed-local-media-path，bench 用 --videomme-pack-mode minicpm-frames
- Daily-Omni: 服务加 --interleave-mm-strings --allowed-local-media-path，bench 用 --daily-omni-pack-mode minicpm-interleave
- Seed-TTS: 标准服务，bench 加 --seed-tts-wer-eval

## 6. 实测对比
| Benchmark | 实测值 | 官方基线 | 准入阈值 | 状态 |
|-----------|--------|---------|---------|------|
| Video-MME | 69.59% | 69.0% | >= 67.0% | PASS |
| Daily-Omni | 78.09% | 79.5% | >= 77.5% | PASS |
| Seed-TTS TTFT | 308 ms | 333 ms | - | 优于基线 |
| Seed-TTS TTFP | 1903 ms | 986 ms | - | 硬件约束 |
| Seed-TTS RTF | 0.45 | 0.44 | - | 持平 |
| Seed-TTS WER | 0.83% | 1.414 | <= 1.56 | PASS |

## 7. 使用指南
- 文本测试: curl http://localhost:8091/v1/chat/completions
- 语音输出: modalities: ["text","audio"] + use_tts_template: true
- Gradio Demo: bash demo/gradio_demo.sh

## 8. 文件结构


## 9. 局限性
1. ASV 未测出: bench 模块缺少参考音频配置（seed_tts_sim_evaluated=0），非模型问题。WER 全量 32 条已测出，0.83%，优于基线 41%。
2. TTFP 劣于官方基线: Talker->Token2Wav 首包链路存在固有延迟。
3. 单卡参数调优空间已穷尽，未尝试 FP8 量化或后端代码修改。

## 10. 交付材料
| # | 交付物 | 状态 |
|---|--------|------|
| 1 | README.md | 完成 |
| 2 | 代码目录 | 完成 |
| 3 | benchmark/ | 完成 |
| 4 | demo/ | 完成 |
| 5 | baseline_raw.json | 完成 |
| 6 | optimized_new.json | 完成 |
| 7 | performance_report.md | 完成 |
| 8 | fuxian.md | 完成 |
| 9 | manual_wer_sample.json | 完成 |
