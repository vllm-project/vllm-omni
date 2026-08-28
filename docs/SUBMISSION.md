# 提交信息总览

## 队伍信息
- 队伍名：小蚂蚁看世界
- 参赛报名名：zhongwen-code-framework
- GitHub：worldant001
- Fork 地址：https://github.com/worldant001/vllm-omni
- 分支：minicpm-challenge

## 提交定位
- 模型权重：未修改 MiniCPM-o 4.5 原始权重
基于官方 minicpm-challenge 分支的基线复现与稳定性验证，未提交额外推理优化代码。

## 精度结果
| Benchmark | 指标 | 实测值 | 准入阈值 | 状态 |
|-----------|------|--------|----------|------|
| Daily-Omni | Accuracy | 77.93% | ≥77.5% | 通过 |
| Video-MME | Accuracy | 69.59% | ≥67.0% | 通过 |
| Seed-TTS | WER | 1.42% | ≤1.56% | 通过 |
| Seed-TTS | SIM | 0.8484 | ≥0.689 | 通过 |

## 性能结果
| 指标 | 实测值 | 官方基线 |
|------|--------|----------|
| RTF | 0.36 | 0.4423 |
| TTFT | 372.10 ms | 333.27 ms |
| TTFP | 1105.19 ms | 986.47 ms |

## Demo
- Gradio Demo 已验证启动，端口 7862
- 演示视频：https://www.bilibili.com/video/BV1ZLtT6pEvE/
- 使用说明：demo/README.md

## 复现
完整步骤见 docs/REPRODUCE.md
