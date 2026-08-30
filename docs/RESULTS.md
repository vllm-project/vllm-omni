# 评测结果汇总

## 精度结果

| Benchmark | 指标 | 本次实测值 | 上次官方评测 | 准入阈值 | 状态 |
|-----------|------|-----------|-------------|----------|------|
| Daily-Omni | Accuracy | 78.18% | 78.2% | ≥77.5% | 通过 |
| Video-MME | Accuracy | 69.59% | 69.48% | ≥67.0% | 通过 |
| Seed-TTS | WER | 1.37% | 1.379% | ≤1.56% | 通过 |
| Seed-TTS | SIM | 0.8459 | 0.8485 | ≥0.689 | 通过 |

## 性能结果

| 指标 | 本次实测值 | 上次官方评测 | 官方基线 | 状态 |
|------|-----------|-------------|----------|------|
| RTF | 0.30 | 0.357 | 0.4423 | 优于基线 19.3% |
| TTFT | 361.29 ms | 351.056 ms | 333.27 ms | 略高 |
| TTFP | 1490.54 ms | 1769.757 ms | 986.47 ms | 优于上次评测 |

## 本轮优化说明

本次提交在官方 minicpm-challenge 分支基础上，通过调整 deploy 配置参数 `token2wav_n_timesteps` 从默认值 10 降至 3，在保持全部四项精度达标的前提下，将 Seed-TTS RTF 从 0.357 优化至 0.30。

## 测试命令与参数

- Seed-TTS：中文数据集 2020 条，单并发，num-warmups 2
- Daily-Omni：全量 1196 条，并发 10，output-len 512，temperature 0
- Video-MME：全量 2700 条，并发 4，96 帧，无字幕，temperature 0

## 原始输出摘要

### Daily-Omni
Overall Accuracy: 935/1196 = 78.18%
Submitted: 1196, Successful HTTP: 1196, Failed: 0

### Video-MME
Overall Accuracy: 1879/2700 = 69.59%
Submitted: 2700, Successful HTTP: 2700, Failed: 0

### Seed-TTS
Evaluated (WER): 2020, Mean WER: 0.0137
SIM evaluated: 2020, Mean SIM: 0.8459
Mean AUDIO_RTF: 0.30, Mean TTFT: 361.29ms, Mean AUDIO_TTFP: 1490.54ms
