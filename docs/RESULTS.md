# 评测结果汇总

## 精度结果

| Benchmark | 指标 | 实测值 | 官方基线 | 准入阈值 | 状态 |
|-----------|------|--------|----------|----------|------|
| Daily-Omni | Accuracy | 77.93% | 79.5% | ≥77.5% | 通过 |
| Video-MME | Accuracy | 69.59% | 69.0% | ≥67.0% | 通过 |
| Seed-TTS | WER | 1.42% | 1.414% | ≤1.56% | 通过 |
| Seed-TTS | SIM | 0.8484 | 0.709 | ≥0.689 | 通过 |


## 性能结果

| 指标 | 实测值 | 官方基线 | 状态 |
|------|--------|----------|------|
| RTF | 0.36 | 0.4423 | 略高 |
| TTFT | 372.10 ms | 333.27 ms | 略高 |
| TTFP | 1105.19 ms | 986.47 ms | 略高 |


## 测试命令与参数

完整命令见 run_benchmarks.sh。主要参数：
- Seed-TTS：中文数据集，2020条，单并发
- Daily-Omni：全量1196条，并发10，output-len 512
- Video-MME：全量2700条，并发4，96帧，无字幕


## 原始输出摘要

### Daily-Omni
Overall Accuracy: 932/1196 = 77.93%

### Video-MME
Overall Accuracy: 1879/2700 = 69.59%

### Seed-TTS
Evaluated (WER): 2020
Mean WER: 0.0142
SIM evaluated: 2020
Mean SIM: 0.8484
