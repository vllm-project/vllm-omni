# 精度验证报告

## 测试环境
- 硬件：Ascend 910C
- 镜像：quay.nju.edu.cn/ascend/vllm-omni:v0.25.0-a3
- CANN：9.0.0

## Video-MME
- 实测准确率：69.59%
- 官方基线：69.0%
- 准入阈值：≥67.0%
- 状态：PASS

## Daily-Omni
- 实测准确率：78.09%
- 官方基线：79.5%
- 准入阈值：≥77.5%
- 状态：PASS

## Seed-TTS
- TTFT：308 ms
- TTFP：1903 ms
- RTF：0.45
- WER：0.83%
- 官方 WER 基线：1.414%
- 状态：PASS（WER 优于基线）

## 结论
三项 benchmark 精度全部达标，满足准入条件。
