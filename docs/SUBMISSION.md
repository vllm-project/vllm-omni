# 提交信息总览

## 队伍信息

- 队伍名：小蚂蚁看世界
- 参赛报名名：zhongwen-code-framework
- GitHub：worldant001
- Fork 地址：https://github.com/worldant001/vllm-omni
- 分支：minicpm-challenge

## 提交定位

基于官方 minicpm-challenge 分支，完成昇腾 910C 上的基线复现与一项配置级性能优化：token2wav_n_timesteps 从 10 降至 3。

未修改模型权重，未改动 vLLM-Omni 核心推理代码。

## 精度结果

| Benchmark | 指标 | 实测值 | 准入阈值 | 状态 |
|-----------|------|--------|----------|------|
| Daily-Omni | Accuracy | 78.18% | ≥77.5% | 通过 |
| Video-MME | Accuracy | 69.59% | ≥67.0% | 通过 |
| Seed-TTS | WER | 1.37% | ≤1.56% | 通过 |
| Seed-TTS | SIM | 0.8459 | ≥0.689 | 通过 |

## 性能结果

| 指标 | 实测值 | 官方基线 |
|------|--------|----------|
| RTF | 0.30 | 0.4423 |
| TTFT | 361.29 ms | 333.27 ms |
| TTFP | 1490.54 ms | 986.47 ms |

## Demo

- Gradio Demo 已验证启动，端口 7862
- 使用说明见 demo/README.md
- 演示视频：https://www.bilibili.com/video/BV1rLtp6BECv/

## 复现

完整步骤见 docs/REPRODUCE.md

## 提交文件清单

- 代码与配置：vllm_omni/deploy/minicpmo_4_5.yaml
- 服务启动脚本：start_serve.sh
- Demo 启动脚本：start_demo.sh
- Benchmark 执行记录：run_benchmarks.sh
- 评测原始日志：results/expB_n3_seed2020_full.log、results/expB_n3_daily1197_full.log、results/expB_n3_videomme2700_full.log
- 文档：docs/RESULTS.md、docs/OPTIMIZATION.md、docs/REPRODUCE.md、docs/SUBMISSION.md
